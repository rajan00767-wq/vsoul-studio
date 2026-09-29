"""Rough garment placement for a Qwen refinement experiment, not final output."""
import numpy as np
from PIL import Image

FABRIC_INSTRUCTION = (
    "Copy image 3's color, pattern scale, weave, layers, seams and collar exactly, with natural folds. "
)


def describe_fabric_color(garment, vl_color=None, vl_rgb=None):
    """Build color guidance from VL, retaining pixels only as supporting evidence."""
    color = str(vl_color or "").strip()
    target = tuple(vl_rgb) if isinstance(vl_rgb, (list, tuple)) and len(vl_rgb) == 3 else None
    if color and color.lower() not in {"none", "unknown", "uncertain"}:
        target_text = f", neutral-light RGB {target}" if target else ""
        return (
            f"VL reads image 3's outer fabric as {color}{target_text}; image 3 pixels remain authoritative."
        )
    return (
        "VL color is uncertain: copy image 3's fabric colors and pattern scale without a named-color preset."
    )


def describe_hair_correction(analyzed_hair_color):
    color = str(analyzed_hair_color or "").strip().lower()
    if "black" in color or "dark" in color:
        return (
            "Keep image 1's black/dark-brown hair, style, curls, strands, flowers and clips. Remove false colored glare; "
            "never recolor or redesign hair."
        )
    return (
        "Preserve the source person's natural hair color exactly and correct only obvious specular glare; "
        "do not darken genuinely light, gray or colored hair."
    )


def describe_lighting_correction(vl_plan):
    """Translate VL lighting evidence without inviting face or hair synthesis."""
    def truthy(value):
        return str(value or "").strip().lower() in {"true", "yes", "1", "present"}

    detected = truthy(vl_plan.get("direct_sunlight_present"))
    hotspot = truthy(vl_plan.get("head_hair_hotspot_present"))
    if not (detected or hotspot):
        return (
            "Use soft neutral frontal studio light; remove only harsh highlights and shadows while preserving natural detail."
        )

    # Region descriptions such as "reconstruct crown strands" made Qwen
    # reinterpret the child's face and hairstyle.  The VL result remains in
    # the audit, while generation receives a restrained global relighting
    # instruction that produced the last known-good portrait prompt.
    return (
        "Use neutral indoor studio light. Remove sunlight, hot spots, hard shadows and color cast; retain natural detail."
    )


def source_hair_is_dark(person, labels):
    """Measure base hair from shaded strands so sunlight cannot relabel it blonde."""
    rgb = np.asarray(person.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Source parsing dimensions do not match")
    hair = labels == 2
    if np.count_nonzero(hair) < labels.size * .01:
        return False
    pixels = rgb[hair].astype(np.float32)
    luma = pixels[:, 0] * .2126 + pixels[:, 1] * .7152 + pixels[:, 2] * .0722
    shaded = luma[luma <= np.percentile(luma, 50)]
    return bool(shaded.size and np.median(shaded) < 82.0)


def ensure_garment_cutout(garment):
    """Derive a soft alpha matte when a product template has a solid backdrop."""
    rgba = garment.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    if int(alpha.min()) < 255:
        return rgba

    rgb = np.asarray(rgba)[:, :, :3].astype(np.float32)
    height, width = rgb.shape[:2]
    patch = max(2, round(min(height, width) * .04))
    corners = np.concatenate((
        rgb[:patch, :patch].reshape(-1, 3),
        rgb[:patch, -patch:].reshape(-1, 3),
        rgb[-patch:, :patch].reshape(-1, 3),
        rgb[-patch:, -patch:].reshape(-1, 3),
    ))
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    # Solid catalog backdrops remain transparent while subtle white-shirt
    # folds survive as partial alpha. This matte is only a placement guide;
    # Qwen still sees the untouched template as its garment reference.
    derived = np.clip((distance - 2.0) * 28.0, 0, 255).astype(np.uint8)
    if np.count_nonzero(derived > 16) < derived.size * .02:
        raise ValueError("Cannot separate the garment from its background")
    result = Image.fromarray(np.dstack((rgb.astype(np.uint8), derived)), "RGBA")
    return result


def build_rough_composite(person, garment, labels, face_box, background):
    person = person.convert("RGB")
    if labels.shape != (person.height, person.width):
        raise ValueError("Source parsing dimensions do not match")
    rgba = ensure_garment_cutout(garment)
    alpha = rgba.getchannel("A")
    bounds = alpha.point(lambda v: 255 if v > 16 else 0).getbbox()
    if not bounds:
        raise ValueError("Empty garment cutout")
    rgba = rgba.crop(bounds)
    x1, y1, x2, y2 = map(float, face_box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid source face box")
    clothing = np.isin(labels, (5, 6, 7, 10, 11, 12))
    _, cols = np.where(clothing)
    if cols.size < labels.size * .03:
        raise ValueError("Cannot locate source clothing for placement")
    width = int(np.clip(np.percentile(cols, 98) - np.percentile(cols, 2),
                        (x2 - x1) * 1.6, person.width * 1.15))
    size = (max(1, width), max(1, round(rgba.height * width / rgba.width)))
    rgba = rgba.resize(size, Image.Resampling.LANCZOS)
    left = round((x1 + x2 - width) / 2)
    top = round(y2 + (y2 - y1) * .16)
    base = np.array(person)
    base[labels == 0] = background
    base[clothing] = background
    canvas = Image.fromarray(base).convert("RGBA")
    canvas.alpha_composite(rgba, (left, top))
    # Restore only parsed person regions over the uniform guide. Restoring the
    # whole top half also restored the outdoor scene, causing Qwen to ignore
    # the requested studio background.
    protect = np.isin(labels, (1, 2, 3, 4, 8, 9, 13, 14, 15, 16, 17, 18, 19)).astype(np.uint8) * 255
    canvas = Image.composite(person, canvas.convert("RGB"), Image.fromarray(protect))
    return canvas, {"garment_box": [left, top, left + size[0], top + size[1]],
                    "mode": "rough_placement_for_qwen_refinement", "not_final": True}


def refinement_prompt(
    background, fabric_color_instruction=None, hair_color_instruction=None,
    lighting_instruction=None, skin_tone_instruction=None, identity_instruction=None,
):
    fabric_color_instruction = fabric_color_instruction or (
        "Copy the outer fabric color directly from image 3 without increasing saturation."
    )
    hair_color_instruction = hair_color_instruction or (
        "Preserve the source person's natural hair color and correct only obvious lighting artifacts."
    )
    lighting_instruction = lighting_instruction or (
        "Use even neutral indoor studio light. Remove sunlight, hot spots, hard shadows and color cast while keeping "
        "real skin texture, eyelashes and hair strands."
    )
    skin_tone_instruction = skin_tone_instruction or (
        "Copy the person's natural facial, ear and neck skin color and undertone directly from image 1. Neutralize "
        "only uneven illumination; do not lighten, darken, whiten, tan, warm, cool or recolor the complexion."
    )
    identity_instruction = identity_instruction or (
        "Images 1 and 2 are the same uploaded person and the only identity references."
    )
    selected_rgb = tuple(int(value) for value in background)
    selected_hex = "#" + "".join(f"{value:02X}" for value in selected_rgb)
    return (
        f"EDIT IMAGE 1. REPLACE its entire scene with solid {selected_hex} in all background and hair gaps. "
        "No original scenery, gradient or texture. Output a photorealistic 35:45 school-ID: one centered child, "
        "whole head, headroom, upper chest, frontal gaze and level shoulders. IMAGES 1 AND 2 ARE THE SAME PERSON: "
        "preserve exact face geometry, ethnicity, skin, expression, hair, earrings and accessories. They override VL. "
        "Remove all necklaces, chains, pendants, lockets and neck ornaments. "
        + lighting_instruction + " " + skin_tone_instruction + " " + identity_instruction + " "
        + hair_color_instruction + " "
        "IMAGE 3 IS UNIFORM ONLY: copy its exact garment with natural neck/shoulder fit. "
        + FABRIC_INSTRUCTION + fabric_color_instruction + " "
        "Remove badges, logos and text. No haze, beauty filter or plastic fabric. "
        f"FINAL BACKGROUND: solid {selected_hex}."
    )
