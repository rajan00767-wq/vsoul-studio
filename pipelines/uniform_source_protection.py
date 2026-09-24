"""Experimental source-head composition, not enabled in the delivery pipeline.

Preserves source lighting as well as identity; this is not portrait restoration.
"""
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

HEAD = (1, 2, 4, 13)  # headwear, hair, glasses, face/neck
CLOTHES = (5, 6, 7, 10, 11, 12)


def compose_parts(source, candidate, source_labels, candidate_labels,
                  source_points, candidate_points, source_box, background):
    src = np.array(source.convert("RGBA"))
    gen = np.array(candidate.convert("RGBA"))
    h, w = src.shape[:2]
    if source_labels.shape != (h, w) or candidate_labels.shape != gen.shape[:2]:
        raise ValueError("Part masks must match their corresponding images")
    source_points = np.asarray(source_points, np.float32)
    candidate_points = np.asarray(candidate_points, np.float32)
    if (source_points.shape != (5, 2) or candidate_points.shape != (5, 2)
            or not np.isfinite(source_points).all()
            or not np.isfinite(candidate_points).all()):
        raise ValueError("Five valid facial landmarks are required")
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(candidate_points, np.float32),
                                          np.asarray(source_points, np.float32), method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Cannot align uniform to the source pose")
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = float(abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    mapped = cv2.transform(np.asarray(candidate_points, np.float32)[None], matrix)[0]
    if not .5 <= scale <= 2 or angle > 20 or np.max(np.linalg.norm(mapped - source_points, axis=1)) > (source_box[2] - source_box[0]) * .15:
        raise ValueError("Uniform pose differs too much for source-preserving composition")
    if np.count_nonzero(np.isin(source_labels, HEAD)) < h * w * .03:
        raise ValueError("Source head mask is missing")
    if np.count_nonzero(np.isin(candidate_labels, CLOTHES)) < candidate_labels.size * .03:
        raise ValueError("Generated uniform mask is missing")

    # Part labels select content; soft subject mattes preserve outer hair edges.
    head_region = np.isin(source_labels, HEAD).astype(np.uint8)
    head_region = cv2.dilate(head_region, np.ones((5, 5), np.uint8))
    head_region[np.isin(source_labels, CLOTHES)] = 0
    chin = int(source_box[3])
    # SCHP often labels white accessories as background and necklaces as
    # clothes. Above the jaw, retain the complete source foreground matte.
    head_region[:max(0, chin)] = 1
    face_width = source_box[2] - source_box[0]
    face_height = source_box[3] - source_box[1]
    center = (source_box[0] + source_box[2]) / 2
    yy, xx = np.mgrid[:h, :w]
    neck_region = (yy >= chin) & (yy < chin + face_height * .45) & (abs(xx - center) < face_width * .34)
    head_region[neck_region] = 1
    head_alpha = src[:, :, 3].astype(float) / 255 * head_region
    cloth_region = np.isin(candidate_labels, CLOTHES).astype(np.uint8)
    cloth_alpha = gen[:, :, 3].astype(float) / 255 * cloth_region
    aligned_rgb = cv2.warpAffine(gen[:, :, :3], matrix, (w, h), flags=cv2.INTER_LINEAR)
    cloth_alpha = cv2.warpAffine(cloth_alpha, matrix, (w, h), flags=cv2.INTER_LINEAR)
    # Keep the generated collar in front of the source neck, but never put
    # generated clothing over the face or long source hair.
    cloth_alpha[:max(0, chin)] = 0
    neck = ((source_labels == 13) | neck_region) & (np.arange(h)[:, None] >= chin)
    head_alpha[neck] *= 1 - cloth_alpha[neck]
    base = np.empty((h, w, 3), dtype=float)
    base[:] = background
    cloth_alpha = cloth_alpha[:, :, None]
    head_alpha = head_alpha[:, :, None]
    output = base * (1 - cloth_alpha) + aligned_rgb * cloth_alpha
    output = output * (1 - head_alpha) + src[:, :, :3] * head_alpha
    return Image.fromarray(np.rint(output).clip(0, 255).astype(np.uint8)), {
        "mode": "schp_source_head_qwen_clothes", "alignment_scale": scale,
        "alignment_angle": angle, "source_lighting_preserved": True,
    }


def protect_source_head(source, candidate, background, job_id):
    from pipelines.photo_restoration import _get_insight_app
    from pipelines.schp_service import parse
    from pipelines.birefnet_service import background_removal
    source = source.convert("RGB")
    # Keep facial proportions intact when source and output aspect ratios differ.
    source.thumbnail(candidate.size, Image.Resampling.LANCZOS)
    detector = _get_insight_app()
    faces = [detector.get(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))
             for img in (source, candidate)]
    if any(len(items) != 1 for items in faces):
        raise ValueError("Source protection requires one face in each image")
    layers, labels = [], []
    try:
        for suffix, img in (("source", source), ("candidate", candidate)):
            parts = parse(np.array(img.convert("RGB")))["labels"]
            labels.append(parts)
            Image.fromarray(parts).save(Path("outputs") / f"{job_id}_{suffix}_parts.png")
            stream = io.BytesIO()
            img.convert("RGB").save(stream, format="PNG")
            matte = Image.open(io.BytesIO(background_removal.remove_background(
                stream.getvalue(), job_id=f"{job_id}_{suffix}", use_schp=False))).getchannel("A")
            layer = img.convert("RGBA")
            layer.putalpha(matte)
            layers.append(layer)
    finally:
        background_removal.unload()
    return compose_parts(*layers, *labels, faces[0][0].kps, faces[1][0].kps,
                         faces[0][0].bbox, background)
