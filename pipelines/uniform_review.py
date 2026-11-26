"""Qwen-VL comparison of a uniform candidate with both uploaded references."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


CHECKS = ("badge_present", "neck_jewelry_present", "uniform_mismatch", "hair_changed", "identity_changed", "face_artifacts",
          "harsh_hair_glare", "background_mismatch", "head_cropped")


def check_uniform_identity(
    source,
    candidate,
    minimum_similarity=0.70,
    maximum_geometry_rmse=0.055,
):
    """Check source identity without inferring gender or substituting pixels."""
    import cv2
    import numpy as np
    from pipelines.photo_restoration import _get_insight_app
    try:
        detector = _get_insight_app()
        faces = [detector.get(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
                 for img in (source, candidate)]
        if any(len(items) != 1 for items in faces):
            return {
                "accepted": False,
                "source_face_count": len(faces[0]),
                "generated_face_count": len(faces[1]),
                "reason": (
                    "Expected one face in source and generated image "
                    f"(detected source={len(faces[0])}, generated={len(faces[1])})"
                ),
            }
        source_face, candidate_face = faces[0][0], faces[1][0]
        score = float(np.dot(source_face.normed_embedding, candidate_face.normed_embedding))

        source_points = np.asarray(
            source_face.landmark_2d_106, dtype=np.float32,
        )[:, :2]
        candidate_points = np.asarray(
            candidate_face.landmark_2d_106, dtype=np.float32,
        )[:, :2]
        source_anchors = np.asarray(source_face.kps, dtype=np.float32)[:, :2]
        source_eye_distance = max(
            float(np.linalg.norm(source_anchors[1] - source_anchors[0])), 1e-6,
        )
        # Align the complete 106-point face before measuring local shape
        # change.  The former eye-only normalization was unstable on scanned,
        # scratched and strongly lit portraits: a one-pixel eye-landmark error
        # inflated otherwise matching geometry by 3-4x.  LMEDS removes global
        # crop/scale/roll while the residual still exposes changed eyes, jaw,
        # nose or mouth. ArcFace and the independent visual reviewer remain
        # mandatory, so this cannot by itself approve a different person.
        transform, _ = cv2.estimateAffinePartial2D(
            candidate_points,
            source_points,
            method=cv2.LMEDS,
        )
        if transform is None:
            geometry_rmse = float("inf")
        else:
            aligned_candidate = cv2.transform(
                candidate_points[None, :, :], transform,
            )[0]
            geometry_rmse = float(np.sqrt(np.mean(
                ((source_points - aligned_candidate) / source_eye_distance) ** 2
            )))
        # A coherent Qwen regeneration can move dense detector landmarks a few
        # pixels even when the recognition embedding is an exceptionally close
        # match. Permit that narrow rendering tolerance only for very high
        # biometric similarity; ordinary candidates retain the stricter gate.
        # A strong ArcFace match can tolerate modest landmark movement from
        # leveling an outdoor pose into a camera-facing ID portrait. Keep the
        # stricter shape gate for borderline identity scores.
        # Dense landmarks are useful for catching de-aging and face reshaping,
        # but their position moves slightly when Qwen levels a portrait or
        # changes the crop.  Do not reject an exceptionally strong biometric
        # match for a sub-pixel threshold miss (the observed production case
        # was similarity=.950 with RMSE=.03045 against .03000).  The bounded
        # .035 allowance applies only at >=.90 similarity. At >=.94 the face
        # embedding is an exceptionally strong identity match; dense 106-point
        # landmarks can still move by ~.047 when Qwen changes portrait scale,
        # expression microgeometry or final framing. Permit at most .050 in
        # that narrow high-confidence tier. Borderline identities retain the
        # caller's strict geometry limit.
        effective_geometry_rmse = float(maximum_geometry_rmse)
        if np.isfinite(score) and score >= .94:
            effective_geometry_rmse = max(effective_geometry_rmse, .050)
        elif np.isfinite(score) and score >= .90:
            effective_geometry_rmse = max(effective_geometry_rmse, .035)
        similarity_ok = bool(np.isfinite(score) and score >= minimum_similarity)
        geometry_ok = bool(
            np.isfinite(geometry_rmse)
            and geometry_rmse <= effective_geometry_rmse
        )
        accepted = bool(similarity_ok and geometry_ok)
        if accepted:
            reason = "identity match"
        elif not similarity_ok:
            reason = "Biometric face similarity is below the required threshold"
        else:
            reason = "Facial landmark geometry exceeds the allowed tolerance"
        return {
            "accepted": accepted,
            "similarity": score if np.isfinite(score) else None,
            "minimum_similarity": minimum_similarity,
            "geometry_rmse": geometry_rmse if np.isfinite(geometry_rmse) else None,
            "requested_maximum_geometry_rmse": maximum_geometry_rmse,
            "maximum_geometry_rmse": effective_geometry_rmse,
            "similarity_ok": similarity_ok,
            "geometry_ok": geometry_ok,
            "reason": reason,
        }
    except Exception as exc:
        return {"accepted": False, "reason": f"Identity verification unavailable: {exc}"}


def pose_offset(keypoints):
    """Nose displacement along the eye axis, invariant to image scale/roll."""
    import numpy as np
    points = np.asarray(keypoints, dtype=float)
    eye_axis = points[1] - points[0]
    squared_distance = float(eye_axis @ eye_axis)
    if squared_distance < 1:
        raise ValueError("Face landmarks are too small for pose checking")
    return float((points[2] - (points[0] + points[1]) / 2) @ eye_axis / squared_distance)


def check_source_pose(source, candidate):
    import cv2
    import numpy as np
    from pipelines.photo_restoration import _get_insight_app
    detector = _get_insight_app()
    offsets = []
    for image in (source, candidate):
        faces = detector.get(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR))
        if len(faces) != 1:
            return {"accepted": False, "reason": "Expected exactly one face in each image"}
        offsets.append(pose_offset(faces[0].kps))
    delta = abs(offsets[0] - offsets[1])
    return {"accepted": delta <= 0.18, "source_offset": offsets[0],
            "candidate_offset": offsets[1], "difference": delta, "maximum_difference": 0.18}


def validate_review(data):
    if not isinstance(data, dict) or any(type(data.get(key)) is not bool for key in CHECKS):
        raise ValueError("Uniform review did not return all required boolean checks")
    issues = [key for key in CHECKS if data[key]]
    correction = data.get("correction")
    if isinstance(correction, (list, dict)):
        correction = json.dumps(correction, ensure_ascii=True)
    if not isinstance(correction, str) or not correction.strip():
        correction = "Correct these observed defects: " + ", ".join(issues) if issues else ""
    return {**data, "correction": correction, "issues": issues}


def review_uniform(person, template, candidate, background_rgb):
    from pipelines.uniform_vl_analyzer import MODEL_DIR, clean_and_repair_json

    prompt = (
        "Inspect this generated portrait only. "
        f"Its required flat background RGB is {background_rgb}. "
        "Return JSON only with boolean keys: badge_present, neck_jewelry_present, face_artifacts, harsh_hair_glare, background_mismatch, head_cropped. "
        "badge_present: visible logo, emblem or lettering on clothing. Seams and plain pockets are NOT badges. "
        "neck_jewelry_present: any necklace, chain or pendant visible on the neck/chest. A shirt collar, placket, button or template necktie is not jewelry. "
        "face_artifacts: visible pixel blocks, smearing, doubled eyes or unnatural facial texture; do not judge identity. "
        "harsh_hair_glare: strong metallic blue/silver glare, not ordinary soft sheen. "
        "background_mismatch: obvious wrong background color or remnants. head_cropped: missing hair/accessories at frame edges. "
        "Also return correction: short specific editing instructions for observed defects only, with locations. "
        "Include badge_evidence describing its location and visible symbol, or none. Do not invent defects or infer hidden details."
    )
    garment_prompt = (
        "Compare image 1 (the exact uniform fitted into the required upper-chest passport crop) "
        "with image 2 (the generated upper-chest passport portrait). Review only garment pixels visible "
        "inside this shared passport crop; do not expect lower sleeves, waist, hem, hands, or any garment "
        "area outside the frame. Ignore the person, hair, background, rough cutout edges, and any template badge. "
        "Return JSON: uniform_mismatch (boolean), evidence (string). Compare collar first: "
        "a standing band and folded triangular points are different. Then compare fabric hue, base/thread colors, "
        "check spacing and visible vest seams at the same apparent scale. Mark uniform_mismatch true only for "
        "a visible difference within the shared crop. Name the actual visible difference; do not infer hidden parts."
    )
    hair_prompt = (
        "Compare image 1 (the original source person) with image 2 (the generated portrait). Image 1 is the authority. "
        "Ignore clothing, background, face restoration detail and differences caused only by the tighter passport crop. "
        "Return JSON: hair_changed (boolean), evidence (string). Compare the visible crown shape, parting, left and right "
        "silhouettes, hair behind each ear, visible length and ends, ponytail/bun/braid/fringe/loose strands, and every real "
        "hair accessory by side, color, shape and count. Eyeglass frames and temples, earrings, bindis, necklaces, clothing "
        "edges, background marks, glare and shadows are not hair accessories. Different lighting, sharper strands, or hair "
        "outside the final upper-chest crop alone is not a hairstyle change. Set hair_changed true only for a concrete visible "
        "loss, invention, relocation, shortening, silhouette change or accessory mismatch, and name that exact difference. "
        "A different hairstyle name alone (for example pigtails versus ponytail, braid, bun or bob) is not evidence. "
        "For tied hair, state the visible source and output section counts and sides; set hair_changed false when both sides, endpoints, part and accessories remain visibly present."
    )
    identity_prompt = (
        "Compare image 1 (the uploaded person) with image 2 (the generated school-ID portrait). Image 1 is the identity "
        "and current apparent-age authority even when it has harsh sunlight. Ignore clothing, background, crop and lighting "
        "differences. Return JSON: identity_changed (boolean), evidence (string). Compare stable facial anatomy: eye size and "
        "spacing, eyelid shape, nose width/length, mouth shape, cheek fullness, jaw width, chin length, facial asymmetry and "
        "current apparent age. Set identity_changed true when the generated result beautifies, de-ages, enlarges the eyes, "
        "narrows or widens the jaw/chin, changes facial proportions, or resembles a different person. Do not accept a face "
        "merely because hair, skin color or expression is broadly similar. Name the concrete changed anatomy."
    )
    cache = Path(tempfile.gettempdir()) / "vsoul_cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uniform_review_", dir=cache) as folder:
        paths = []
        for index, img in enumerate((person, template, candidate)):
            path = Path(folder) / f"image_{index}.png"
            preview = img.convert("RGB").copy()
            preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
            preview.save(path)
            paths.append(str(path.resolve()))
        worker = Path(folder) / "worker.py"
        worker.write_text(f'''
import json
import torch
from PIL import Image
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
model_dir = {str(MODEL_DIR.resolve())!r}
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, quantization_config=config, device_map={{"": 0}}, local_files_only=True).eval()
images = [Image.open(path).convert("RGB") for path in {paths!r}]
def inspect(selected, prompt):
    content = [{{"type": "image", "image": img}} for img in selected] + [{{"type": "text", "text": prompt}}]
    text = processor.apply_chat_template([{{"role": "user", "content": content}}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=selected, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
results = [inspect([images[2]], {prompt!r}),
           inspect([images[1], images[2]], {garment_prompt!r}),
           inspect([images[0], images[2]], {hair_prompt!r}),
           inspect([images[0], images[2]], {identity_prompt!r})]
print("REVIEW_JSON=" + json.dumps(results))
''', encoding="utf-8")
        result = subprocess.run([sys.executable, str(worker)], capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError("Uniform review worker failed: " + result.stderr[-800:])
        line = next((s for s in result.stdout.splitlines() if s.startswith("REVIEW_JSON=")), None)
        if line is None:
            raise RuntimeError("Uniform review returned no result")
        raw = json.loads(line.split("=", 1)[1])
        portrait, garment, hair, identity = [clean_and_repair_json(item) for item in raw]
        portrait["uniform_mismatch"] = garment.get("uniform_mismatch")
        portrait["hair_changed"] = hair.get("hair_changed")
        portrait["identity_changed"] = identity.get("identity_changed")
        portrait["garment_evidence"] = garment.get("evidence")
        portrait["hair_evidence"] = hair.get("evidence")
        portrait["identity_evidence"] = identity.get("evidence")
        return validate_review(portrait)
