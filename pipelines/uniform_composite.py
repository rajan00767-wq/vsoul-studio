"""Rough garment placement for a Qwen refinement experiment, not final output."""
import numpy as np
from PIL import Image

FABRIC_INSTRUCTION = (
    "Match the reference fabric's base color separately from its pattern-line color. "
    "Preserve check or stripe spacing, line thickness, weave, garment layers and seam placement. "
    "Use subtle woven texture at photographic scale, not embossed grids, denim, plastic or glossy fabric. "
    "Let the pattern follow small natural folds and shoulder curvature without changing its design. "
    "Use gentle collar contact shadows; do not tint the uniform with the background color. "
)


def describe_fabric_color(garment):
    """Describe dark cloth without mistaking cool reflected light for navy."""
    rgba = np.array(garment.convert("RGBA"))
    rgb = rgba[:, :, :3]
    dark = (rgba[:, :, 3] > 32) & (rgb.max(axis=2) < 90)
    if np.count_nonzero(dark) < rgba.shape[0] * rgba.shape[1] * .03:
        return "Copy the outer fabric color directly from image 2 without increasing saturation."
    median = np.median(rgb[dark], axis=0).round().astype(int)
    if median.max() < 75 and median.max() - median.min() < 40:
        return (
            "The outer garment is neutral dark black / near-black fabric, not navy blue. "
            "Any slight blue channel in image 2 is a cool photographic reflection, not the cloth color. "
            "Render the cloth as black with restrained neutral highlights; do not increase blue saturation."
        )
    return (
        f"Match the outer fabric's observed dark-pixel color near RGB {tuple(median)} while preserving "
        "neutral highlights; do not borrow color from the backdrop."
    )


def describe_hair_correction(analyzed_hair_color):
    color = str(analyzed_hair_color or "").strip().lower()
    if "black" in color or "dark" in color:
        return (
            "Source analysis identifies naturally dark/black hair. Treat pale gray, silver, cyan or blue patches "
            "on the crown and hairline as outdoor glare, not hair color; reconstruct only those affected strands "
            "using adjacent dark hair, curl direction and fine texture."
        )
    return (
        "Preserve the source person's natural hair color exactly and correct only obvious specular glare; "
        "do not darken genuinely light, gray or colored hair."
    )


def build_rough_composite(person, garment, labels, face_box, background):
    person = person.convert("RGB")
    if labels.shape != (person.height, person.width):
        raise ValueError("Source parsing dimensions do not match")
    rgba = garment.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] == 255:
        raise ValueError("This experiment requires a transparent garment cutout")
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
    base[clothing] = background
    canvas = Image.fromarray(base).convert("RGBA")
    canvas.alpha_composite(rgba, (left, top))
    # Restore source hair over the pasted shoulders, without copying old clothes.
    protect = np.isin(labels, (1, 2, 4)).astype(np.uint8) * 255
    protect[:max(0, int(y2))] = 255
    canvas = Image.composite(person, canvas.convert("RGB"), Image.fromarray(protect))
    return canvas, {"garment_box": [left, top, left + size[0], top + size[1]],
                    "mode": "rough_placement_for_qwen_refinement", "not_final": True}


def refinement_prompt(background, fabric_color_instruction=None, hair_color_instruction=None):
    fabric_color_instruction = fabric_color_instruction or (
        "Copy the outer fabric color directly from image 2 without increasing saturation."
    )
    hair_color_instruction = hair_color_instruction or (
        "Preserve the source person's natural hair color and correct only obvious lighting artifacts."
    )
    return (
        "Refine image 1, a rough clothing composite, into one natural school-ID photograph. "
        "The person already in image 1 is the only identity reference: preserve their face, "
        "expression, hairstyle, hair length, hair accessories and jewelry. Do not invent a new person. "
        "Image 2 is the authoritative uniform reference. Repair the rough pasted collar and "
        "shoulder boundaries in image 1, remove any remnants of old clothing, and fit this exact "
        "uniform to the existing neck and shoulders. Keep image 2's collar, buttons and panel construction. "
        + FABRIC_INSTRUCTION + fabric_color_instruction + " "
        "Remove badges, logos and lettering only, replacing them with matching plain fabric. "
        "Use soft neutral studio lighting with natural skin tones and restrained highlights. "
        + hair_color_instruction + " "
        "Keep the exact original hairline, parting, length, curl silhouette and both source hair clips. "
        "Preserve exact facial geometry, eye size, iris size, eyelid shape, gaze, nose, mouth opening, lip shape and expression. "
        "Restore clean photographic facial detail with natural pores and smooth tonal transitions, without enlarging the eyes, "
        "beautifying, airbrushing, sharpening halos, waxy skin, metallic hair or harsh sunlight. "
        f"Replace all background, including gaps around hair, with solid RGB {tuple(background)}. "
        "Keep the entire head and both hair accessories visible with headroom, shoulders and upper chest. "
        "No hands, extra objects, collage or invented ornaments. Output a single photograph."
    )
