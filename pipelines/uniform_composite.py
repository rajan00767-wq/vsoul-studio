"""Rough garment placement for a Qwen refinement experiment, not final output."""
import numpy as np
import cv2
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
            "Keep image 2's black/dark-brown hair, hairstyle, strands, length and parting. Preserve only accessories that are visibly present in image 2. Remove false colored glare; "
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


def measure_source_person_colors(person, labels):
    """Return robust source skin and hair RGB samples from parsed regions."""
    rgb = np.asarray(person.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Source parsing dimensions do not match")

    def stable_median(mask, *, darker_half=False):
        pixels = rgb[mask].astype(np.float32)
        if len(pixels) < 32:
            return None
        luma = pixels @ np.asarray((.2126, .7152, .0722), dtype=np.float32)
        if darker_half:
            pixels = pixels[luma <= np.percentile(luma, 55)]
        else:
            low, high = np.percentile(luma, (20, 80))
            stable = pixels[(luma >= low) & (luma <= high)]
            if len(stable) >= 32:
                pixels = stable
        value = np.median(pixels, axis=0).round().clip(0, 255).astype(np.uint8)
        return tuple(int(channel) for channel in value)

    return {
        "skin_rgb": stable_median(labels == 13),
        # Shaded strands reveal base pigment more reliably than sunlit crowns.
        "hair_rgb": stable_median(labels == 2, darker_half=True),
    }


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


def opaque_garment_cutout(garment):
    """Return a garment cutout whose interior cannot leak backdrop color."""
    rgba = ensure_garment_cutout(garment)
    pixels = np.asarray(rgba)
    alpha = pixels[:, :, 3]
    mask = (alpha > 16).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("Cannot locate garment silhouette")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)

    inverse = (mask == 0).astype(np.uint8)
    _, regions = cv2.connectedComponents(inverse, connectivity=8)
    border_regions = np.unique(np.concatenate((
        regions[0], regions[-1], regions[:, 0], regions[:, -1],
    )))
    internal_holes = (inverse > 0) & ~np.isin(regions, border_regions)
    mask[internal_holes] = 1

    edge_alpha = cv2.GaussianBlur(mask * 255, (3, 3), 0.45)
    opaque = pixels.copy()
    opaque[:, :, 3] = edge_alpha
    return Image.fromarray(opaque, "RGBA")


def garment_on_selected_background(garment, background):
    """Place an opaque garment silhouette on the exact generation backdrop.

    The soft alpha used by the rough placement guide is unsuitable for a
    colored reference canvas: blue can show through white fabric highlights.
    Fill internal alpha holes and keep softness only at the outer silhouette.
    """
    cutout = opaque_garment_cutout(garment)
    canvas = Image.new("RGBA", cutout.size, (*tuple(int(v) for v in background), 255))
    canvas.alpha_composite(cutout)
    return canvas.convert("RGB")


def passport_garment_conditioning(garment, background, size):
    """Create a passport-sized garment-only reference for Qwen.

    Full-body/template canvases give the editor room to invent a scene or a
    different pose. Crop the transparent garment to its visible bounds and
    place only the upper-chest portion on a fixed passport canvas.
    """
    width, height = (int(size[0]), int(size[1]))
    cutout = opaque_garment_cutout(garment)
    bounds = cutout.getchannel("A").getbbox()
    if not bounds:
        return garment_on_selected_background(garment, background).resize((width, height), Image.Resampling.LANCZOS)
    cutout = cutout.crop(bounds)
    target_width = max(1, int(width * .76))
    target_height = max(1, int(cutout.height * target_width / max(1, cutout.width)))
    cutout.thumbnail((target_width, max(1, int(height * .62))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (*tuple(int(v) for v in background), 255))
    left = (width - cutout.width) // 2
    top = int(height * .38)
    canvas.alpha_composite(cutout, (left, min(top, height - cutout.height)))
    return canvas.convert("RGB")


def build_rough_composite(person, garment, labels, face_box, background):
    person = person.convert("RGB")
    if labels.shape != (person.height, person.width):
        raise ValueError("Source parsing dimensions do not match")
    rgba = opaque_garment_cutout(garment)
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
    measured_width = float(np.percentile(cols, 98) - np.percentile(cols, 2))
    # A small garment thumbnail encourages Qwen to keep the source clothing.
    # Fill the portrait width like an actual upper-chest school-ID uniform.
    width = int(np.clip(
        max(measured_width, (x2 - x1) * 2.25, person.width * .92),
        person.width * .82,
        person.width * 1.08,
    ))
    size = (max(1, width), max(1, round(rgba.height * width / rgba.width)))
    rgba = rgba.resize(size, Image.Resampling.LANCZOS)
    left = round((x1 + x2 - width) / 2)
    top = round(y2 - (y2 - y1) * .03)
    base = np.array(person)
    base[labels == 0] = background
    base[clothing] = background
    # The Qwen edit canvas is a passport upper-chest guide. Do not expose
    # source forearms/hands or carried objects to the model; they encourage
    # full-body generations that later fail the passport framing check.
    arms = np.isin(labels, (14, 15))
    base[arms] = background
    canvas = Image.fromarray(base).convert("RGBA")
    canvas.alpha_composite(rgba, (left, top))
    # Restore only parsed person regions over the uniform guide. Restoring the
    # whole top half also restored the outdoor scene, causing Qwen to ignore
    # the requested studio background.
    protect = np.isin(labels, (1, 2, 3, 4, 8, 9, 13, 16, 17, 18, 19)).astype(np.uint8) * 255
    canvas = Image.composite(person, canvas.convert("RGB"), Image.fromarray(protect))
    return canvas, {"garment_box": [left, top, left + size[0], top + size[1]],
                    "mode": "rough_placement_for_qwen_refinement", "not_final": True}


def build_person_backdrop_guide(person, labels, background, remove_clothing=False):
    """Keep the uploaded person while removing all original scenery.

    Qwen receives this as the editable source portrait. The uniform is supplied
    separately, so it must synthesize a natural garment instead of tracing a
    pasted rough composite.
    """
    rgb = np.asarray(person.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Source parsing dimensions do not match")
    backdrop = np.empty_like(rgb)
    backdrop[:] = tuple(int(value) for value in background)
    subject = labels != 0
    if remove_clothing:
        # Keep the real face, neck, hair and accessories, but clear the source
        # garment so Qwen has an unambiguous region to fill from the supplied
        # uniform reference. This is an edit canvas, not a garment composite.
        subject &= ~np.isin(labels, (5, 6, 7, 9, 10, 12))
    # A slightly feathered outer boundary gives Qwen strong, untouched face
    # and hairstyle pixels without teaching it to reproduce a hard pasted
    # silhouette. Interior subject pixels remain exactly the upload.
    alpha = cv2.GaussianBlur(subject.astype(np.float32), (0, 0), 1.15)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    guide = rgb.astype(np.float32) * alpha + backdrop.astype(np.float32) * (1.0 - alpha)
    return Image.fromarray(np.rint(guide).clip(0, 255).astype(np.uint8), "RGB")


def build_person_identity_guide(person, labels, face_box):
    """Return an untouched rectangular source crop for face and hair guidance.

    A semantic cutout creates hard face/hair boundaries that Qwen can reproduce
    as a pasted head. This crop keeps the uploaded pixels and their natural
    local context intact while excluding as much irrelevant lower torso as the
    source framing allows.
    """
    rgb = np.asarray(person.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Source parsing dimensions do not match")
    height, width = labels.shape
    x1, y1, x2, y2 = (float(value) for value in face_box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid source face box")
    face_w = x2 - x1
    face_h = y2 - y1

    head = np.isin(labels, (1, 2, 4, 13))
    rows, cols = np.where(head)
    if rows.size:
        content_left = min(float(np.percentile(cols, 1)), x1)
        content_right = max(float(np.percentile(cols, 99)), x2)
        content_top = min(float(np.percentile(rows, 1)), y1)
        content_bottom = max(float(np.percentile(rows, 99)), y2)
    else:
        content_left, content_top, content_right, content_bottom = x1, y1, x2, y2

    left = max(0, int(np.floor(content_left - face_w * .38)))
    right = min(width, int(np.ceil(content_right + face_w * .38)))
    top = max(0, int(np.floor(content_top - face_h * .30)))
    # Preserve long hair and shoulder context when present, without turning the
    # identity guide into a second garment reference.
    bottom = min(height, int(np.ceil(max(content_bottom, y2 + face_h * 1.05))))
    if right - left < 2 or bottom - top < 2:
        return person.convert("RGB").copy()
    return person.convert("RGB").crop((left, top, right, bottom))


def refinement_prompt(
    background, fabric_color_instruction=None, hair_color_instruction=None,
    lighting_instruction=None, skin_tone_instruction=None, identity_instruction=None,
    primary_is_identity: bool = False,
):
    fabric_color_instruction = fabric_color_instruction or (
        "Copy the outer fabric color directly from image 3 without increasing saturation."
    )
    hair_color_instruction = hair_color_instruction or (
        "Preserve the source person's natural hair color and correct only obvious lighting artifacts."
    )
    lighting_instruction = lighting_instruction or (
        "Use neutral studio light; no sunlight, hot spots, hard shadows or color cast; retain real texture."
    )
    skin_tone_instruction = skin_tone_instruction or (
        "Copy facial, ear and neck skin color and undertone directly from image 2; correct uneven light only and never "
        "recolor the complexion."
    )
    identity_instruction = identity_instruction or (
        "Image 2 contains the original person's face, hair and accessories and is the only identity authority."
    )
    selected_rgb = tuple(int(value) for value in background)
    selected_hex = "#" + "".join(f"{value:02X}" for value in selected_rgb)
    selected_rgb_text = f"RGB({selected_rgb[0]}, {selected_rgb[1]}, {selected_rgb[2]})"
    if primary_is_identity:
        fabric_color_instruction = fabric_color_instruction.replace("image 3", "image 2")
        hair_color_instruction = hair_color_instruction.replace("image 2", "image 1")
        lighting_instruction = lighting_instruction.replace("image 2", "image 1")
        skin_tone_instruction = skin_tone_instruction.replace("image 2", "image 1")
        identity_instruction = identity_instruction.replace("Image 2", "Image 1")
        return (
            "CREATE ONE NATURAL FINAL SCHOOL-ID PORTRAIT OF EXACTLY THE CHILD IN IMAGE 1. "
            "IMAGE 1 IS THE ORIGINAL IDENTITY, FACE, HAIR AND POSE AUTHORITY. "
            "IMAGE 2 IS THE GARMENT AUTHORITY: replace only the clothing with its exact uniform. "
            f"BACKGROUND MUST BE ONLY one featureless, perfectly flat field of {selected_hex}, {selected_rgb_text}; no gradient, texture or scenery; no shadow, room, wall, window or objects. "
            "Create a photorealistic passport photo; no room, equipment, scenery, props, hands or lower arms. "
            "Show the full head, shoulders and upper chest with headroom; stop at the upper chest. "
            "Use a standard school-ID passport pose: center the face on the vertical axis, face the camera directly, keep the head upright with no roll or tilt, "
            "keep both eyes level, and square both shoulders evenly to the camera. Preserve exact face, age, ethnicity, complexion, "
            "expression, earrings, geometry, hairline, hairstyle, hair length and parting from image 1. "
            "Keep only the head accessories visibly present in image 1; do not add, remove, move or redesign any accessory. "
            "Photographic restoration only: improve detail and neutral light without recreating the child, pasting a face, or changing hair. "
            "Remove all necklaces, chains, pendants, lockets and non-uniform neck accessories, even when visible in image 1. "
            "Keep the skin between the jawline and uniform collar clear; reproduce only the collar construction visibly present in image 2. "
            + lighting_instruction + " " + skin_tone_instruction + " " + identity_instruction + " "
            + hair_color_instruction + " "
            + "Copy image 2's color, pattern scale, weave, layers, seams and collar exactly, with natural folds. "
            + fabric_color_instruction + " "
            "Render image 2's complete uniform seamlessly across both shoulders and upper torso, preserving its colors, layers, collar, "
            "seams, weave and fabric texture exactly. Remove badges, logos and text. "
            f"FINAL BACKGROUND: {selected_hex}, {selected_rgb_text}."
        )
    return (
        "REFINE IMAGE 1'S ROUGH UNIFORM LAYOUT INTO ONE NATURAL FINAL SCHOOL PORTRAIT OF EXACTLY ONE CHILD. "
        "IMAGE 2 IS THE SOFT-EDGED ORIGINAL IDENTITY/HAIR GUIDE, NOT FINAL PASTED PIXELS. "
        "IMAGE 3 IS THE GARMENT AUTHORITY: use its exact uniform. "
        f"BACKGROUND MUST BE ONLY one featureless, perfectly flat field of {selected_hex}, {selected_rgb_text}; no gradient, texture or scenery; no shadow, room, wall, window or objects. "
        "Create a photorealistic portrait, not a pasted composite: passport photo with soft neutral light on the child only; no interior, equipment, scenery, or shop. Show full head, shoulders and upper chest "
        "with headroom; the application will crop to passport size. Stop at the upper chest; no lower torso. "
        "Keep a natural neck and continuous shoulders/garment; no hollow gap or floating head. Camera-facing: upright head, "
        "level eyes, square even shoulders. "
        "Keep hands, wrists, props and lower arms outside the portrait. "
        "IMAGE 2 IS THE ONLY IDENTITY SOURCE. Photographic restoration, not person recreation: do not regenerate or "
        "redesign the face, head or hair. Preserve exact face, ethnicity, skin, expression, earrings, geometry, hairline, hairstyle, "
        "hair length, parting and every visible clip or tie from image 2. Restore natural detail and light only. "
        "Never pull hair back, shorten it, smooth it into a new style, or replace its accessories. No beautification or face paste. "
        "Remove all necklaces, chains, pendants and lockets; retain only image 3 uniform neckwear. "
        + lighting_instruction + " " + skin_tone_instruction + " " + identity_instruction + " "
        + hair_color_instruction + " "
        + FABRIC_INSTRUCTION + fabric_color_instruction + " "
        "Render the complete uniform seamlessly across both shoulders and the torso; no rectangular patches, transparent "
        "sleeves, source-clothing remnants or collage edges. Remove badges, logos, text. "
        f"FINAL BACKGROUND: {selected_hex}, {selected_rgb_text}."
    )
