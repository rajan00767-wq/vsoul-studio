"""Deterministic one-image preparation for Qwen uniform refinement."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _face_box(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest frontal face, with a geometry fallback for portrait crops."""
    height, width = rgb.shape[:2]
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 4, minSize=(36, 36))
    if len(faces):
        return tuple(int(v) for v in max(faces, key=lambda box: box[2] * box[3]))
    face_width = max(48, int(width * 0.30))
    face_height = max(56, int(height * 0.28))
    return ((width - face_width) // 2, int(height * 0.13), face_width, face_height)


def _foreground_alpha(uniform: Image.Image) -> np.ndarray:
    rgba = np.array(uniform.convert("RGBA"))
    alpha = rgba[:, :, 3]
    # Proper product cutouts have a meaningful alpha channel.  Tiny nonzero
    # values around an otherwise transparent canvas must not expand the bounds.
    if int((alpha > 24).sum()) > int(alpha.size * 0.03):
        return np.where(alpha > 24, alpha, 0).astype(np.uint8)

    # Basic background fallback for non-transparent templates. Qwen-VL will
    # subsequently correct this rough placement; this stage must not invent cloth.
    rgb = rgba[:, :, :3]
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = np.where(distance > 20, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def build_uniform_analysis_board(
    person: Image.Image, uniform: Image.Image, rough_composite: Image.Image
) -> Image.Image:
    """Create one labeled visual input for Qwen-VL fit analysis."""
    panel_height = 512
    panels = []
    for image in (person, uniform, rough_composite):
        rgb = image.convert("RGB")
        scale = panel_height / max(1, rgb.height)
        panels.append(rgb.resize((max(1, int(rgb.width * scale)), panel_height), Image.LANCZOS))

    gap, header = 8, 34
    board_width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    board = Image.new("RGB", (board_width, panel_height + header), (20, 26, 34))
    draw = ImageDraw.Draw(board)
    labels = ("PERSON", "UNIFORM TEMPLATE", "ROUGH FIT")
    left = 0
    for panel, label in zip(panels, labels):
        board.paste(panel, (left, header))
        draw.text((left + 8, 9), label, fill=(235, 242, 248))
        left += panel.width + gap
    return board


def compose_uniform_on_person(
    person: Image.Image, uniform: Image.Image, fit_constraints: dict | None = None
) -> Image.Image:
    """Place the supplied garment under the source head as one Qwen input image."""
    person_rgb = np.array(person.convert("RGB"))
    height, width = person_rgb.shape[:2]
    uniform_rgba = np.array(uniform.convert("RGBA"))
    alpha = _foreground_alpha(uniform)
    ys, xs = np.where(alpha > 24)
    if len(xs) < 100:
        raise ValueError("Could not isolate visible uniform pixels from the template")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    garment = uniform_rgba[y0:y1, x0:x1].copy()
    garment[:, :, 3] = alpha[y0:y1, x0:x1]

    face_x, face_y, face_w, face_h = _face_box(person_rgb)
    # A portrait crop usually has shoulders around 3.4 face widths wide. Keep
    # margins so the generated uniform can naturally continue at both edges.
    constraints = fit_constraints or {}
    try:
        width_scale = float(constraints.get("uniform_scale", 1.0))
    except (TypeError, ValueError):
        width_scale = 1.0
    width_scale = float(np.clip(width_scale, 0.88, 1.10))
    target_width = int(np.clip(max(face_w * 3.4, width * 0.74) * width_scale, face_w * 2.7, width * 1.06))
    scale = target_width / max(1, garment.shape[1])
    target_height = max(1, int(garment.shape[0] * scale))
    garment = cv2.resize(garment, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    # Haar boxes include the forehead and hair on children.  Keep a visible
    # neck gap below the jawline. VL is allowed to make a small placement
    # adjustment, but it must not lift a shirt collar onto the chin.
    try:
        collar_ratio = float(constraints.get("collar_anchor_ratio", 1.10))
    except (TypeError, ValueError):
        collar_ratio = 1.10
    collar_y = int(np.clip(face_y + face_h * np.clip(collar_ratio, 1.04, 1.14), 0, height - 1))
    left = int(face_x + face_w * 0.5 - target_width * 0.5)
    top = collar_y
    canvas = person_rgb.copy()
    # Remove the source outfit before placing the template. Otherwise any
    # uncovered sleeve/collar pixels are treated by Qwen as intentional cloth
    # and leak into the finished uniform at the shoulders.
    try:
        from pipelines.schp_service import CLOTHING_LABELS, parse

        labels = parse(person_rgb)["labels"]
        old_clothes = np.isin(labels, tuple(CLOTHING_LABELS)).astype(np.uint8) * 255
        old_clothes[:collar_y, :] = 0
        old_clothes = cv2.dilate(
            old_clothes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1
        )
        old_clothes = cv2.GaussianBlur(old_clothes, (15, 15), 0).astype(np.float32)[:, :, None] / 255.0
        studio_blue = np.array((205, 230, 248), dtype=np.float32)
        canvas = (studio_blue * old_clothes + canvas.astype(np.float32) * (1.0 - old_clothes)).clip(0, 255).astype(np.uint8)
    except Exception:
        # The garment paste remains usable when the optional parser is absent.
        pass

    src_x0, src_y0 = max(0, -left), max(0, -top)
    dst_x0, dst_y0 = max(0, left), max(0, top)
    dst_x1, dst_y1 = min(width, left + target_width), min(height, top + target_height)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        raise ValueError("Uniform placement falls outside the portrait frame")
    layer = garment[src_y0:src_y0 + (dst_y1 - dst_y0), src_x0:src_x0 + (dst_x1 - dst_x0)]
    layer_alpha = cv2.GaussianBlur(layer[:, :, 3], (5, 5), 0).astype(np.float32) / 255.0
    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = (
        layer[:, :, :3].astype(np.float32) * layer_alpha[:, :, None]
        + region * (1.0 - layer_alpha[:, :, None])
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB")
