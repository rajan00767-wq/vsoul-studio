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
            "Keep image 1's black/dark-brown hair, hairstyle, strands, length and parting. Preserve only accessories that are visibly present on the person in image 1. Remove false colored glare; "
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
            "Use soft even frontal light; remove only harsh highlights and shadows while preserving natural detail."
        )

    # Region descriptions such as "reconstruct crown strands" made Qwen
    # reinterpret the child's face and hairstyle.  The VL result remains in
    # the audit, while generation receives a restrained global relighting
    # instruction that produced the last known-good portrait prompt.
    return (
        "Use neutral even frontal light. Remove sunlight, hot spots, hard shadows and color cast; retain natural detail."
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
    # Raised from 82->105 to include dark brown hair (common in women/girls)
    # that was incorrectly classified as 'not dark' and skipped correction
    return bool(shaded.size and np.median(shaded) < 105.0)


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
    # Remove only background-coloured pixels connected to the canvas border.
    # The former per-pixel colour key made every white plaid/check region
    # transparent, creating repeated X-shaped holes throughout the garment
    # reference. Internal white fabric is foreground even when it matches the
    # product-photo backdrop exactly.
    background_like = (distance <= 42.0).astype(np.uint8)
    count, components = cv2.connectedComponents(background_like, connectivity=8)
    border_components = np.unique(np.concatenate((
        components[0, :], components[-1, :],
        components[:, 0], components[:, -1],
    )))
    border_components = border_components[border_components != 0]
    connected_background = np.isin(components, border_components)
    foreground = (~connected_background).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    derived = _smooth_inner_alpha(foreground, radius=2.2)
    if np.count_nonzero(derived > 16) < derived.size * .02:
        raise ValueError("Cannot separate the garment from its background")
    result = Image.fromarray(np.dstack((rgb.astype(np.uint8), derived)), "RGBA")
    return result


def _smooth_inner_alpha(mask, radius=2.4):
    """Create an antialiased alpha ramp entirely inside a binary silhouette."""
    mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8)

    # Smooth raster stair steps in the contour itself before measuring the
    # inner distance. The threshold keeps the geometry centred rather than
    # growing it into product-background pixels.
    contour = cv2.GaussianBlur(
        mask.astype(np.float32), (0, 0), sigmaX=.62, sigmaY=.62,
        borderType=cv2.BORDER_REPLICATE,
    )
    smooth_mask = (contour >= .5).astype(np.uint8)
    inside = cv2.distanceTransform(smooth_mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - smooth_mask, cv2.DIST_L2, 5)
    signed_distance = cv2.GaussianBlur(
        inside - outside, (0, 0), sigmaX=.82, sigmaY=.82,
        borderType=cv2.BORDER_REPLICATE,
    )
    alpha = np.clip(
        (signed_distance - .10) / max(float(radius), .5), 0.0, 1.0,
    )
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    alpha[smooth_mask == 0] = 0.0
    return np.rint(alpha * 255.0).clip(0, 255).astype(np.uint8)


def opaque_garment_cutout(garment):
    """Return a garment cutout whose interior cannot leak backdrop color."""
    supplied_alpha = np.asarray(garment.convert("RGBA").getchannel("A"))
    has_supplied_transparency = int(supplied_alpha.min()) < 255
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

    # Some catalog PNGs store an opaque white keyline several pixels inside
    # the nominal alpha silhouette. Remove only backdrop-coloured pixels in a
    # narrow *outer* band; internal white checks/collars are far from that
    # boundary and remain untouched. The sampled colour is per upload, so this
    # also handles gray or other solid product-photo canvases.
    interior_depth = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 3,
    )
    background_distance = None
    transparent_pixels = pixels[alpha <= 4, :3]
    if transparent_pixels.shape[0] >= 32:
        product_background = np.median(
            transparent_pixels.astype(np.float32), axis=0,
        )
        background_distance = np.linalg.norm(
            pixels[:, :, :3].astype(np.float32)
            - product_background[None, None, :],
            axis=2,
        )
        opaque_keyline = (
            (mask > 0)
            & (alpha >= 224)
            & (interior_depth <= 3.0)
            & (background_distance <= 42.0)
        )
        mask[opaque_keyline] = 0

    # Feather inside the detected garment silhouette. A signed inner-distance
    # ramp gives diagonal shoulders and sleeve curves stable sub-pixel edges;
    # the old erode-plus-3x3-blur path left visible stair steps and could shrink
    # collar tips by several pixels.
    eroded = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        # Keep clean colour donors several pixels away from the catalogue
        # background. Legitimate semi-transparent antialiasing is restored
        # below after its contaminated RGB has been replaced.
        iterations=3 if not has_supplied_transparency else 2,
    )
    edge_alpha = _smooth_inner_alpha(mask, radius=2.35)
    edge_alpha[(mask != 0) & (interior_depth >= 4.0)] = 255
    semi_transparent_edge = (
        has_supplied_transparency
        & (mask != 0) & (interior_depth <= 2.0)
        & (alpha > 16) & (alpha < 224)
    )
    edge_alpha[semi_transparent_edge] = np.minimum(
        edge_alpha[semi_transparent_edge], alpha[semi_transparent_edge],
    )
    opaque = pixels.copy()
    # The original RGB at a catalog cutout edge is already blended with the
    # product-photo backdrop. Alpha compositing that RGB onto a saturated
    # passport backdrop creates a white/gray outline around sleeves and
    # shoulders. Propagate clean fabric colour outward from the eroded core;
    # the alpha below still supplies the one-pixel photographic softness.
    clean_rgb = opaque[:, :, :3].astype(np.float32)
    # High-confidence alpha is required as a colour donor. A geometrically
    # eroded but still semi-transparent product-edge pixel already contains
    # backdrop RGB and must itself be replaced.
    known = eroded.astype(bool) & (alpha >= 224)
    if background_distance is not None:
        known &= ~(
            (interior_depth <= 6.0) & (background_distance <= 42.0)
        )
    unresolved = mask.astype(bool) & ~known
    for _ in range(8):
        if not np.any(unresolved):
            break
        neighbor_count = cv2.boxFilter(
            known.astype(np.float32), -1, (3, 3), normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        fillable = unresolved & (neighbor_count > 0)
        if not np.any(fillable):
            break
        for channel in range(3):
            neighbor_sum = cv2.boxFilter(
                clean_rgb[:, :, channel] * known.astype(np.float32),
                -1, (3, 3), normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            clean_rgb[:, :, channel][fillable] = (
                neighbor_sum[fillable] / neighbor_count[fillable]
            )
        known[fillable] = True
        unresolved &= ~fillable
    opaque[:, :, :3] = np.rint(clean_rgb).clip(0, 255).astype(np.uint8)
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
    # `opaque_garment_cutout` already removes contaminated product-background
    # RGB and supplies a tight inner feather. Preserve that antialiasing here;
    # converting it back to a binary eroded mask caused jagged template edges.
    bounds = cutout.getchannel("A").getbbox()
    if not bounds:
        return garment_on_selected_background(garment, background).resize((width, height), Image.Resampling.LANCZOS)
    cutout = cutout.crop(bounds)
    # Qwen needs the portion that is visible in an ID photo, not the lower shirt
    # or the template's original canvas. Retain the collar, shoulder seams,
    # sleeves, placket and upper pocket/fabric, then enlarge that crop so pattern
    # weave and construction occupy useful conditioning pixels.
    upper_chest_bottom = max(1, min(cutout.height, int(round(cutout.height * .62))))
    cutout = cutout.crop((0, 0, cutout.width, upper_chest_bottom))
    target_width = max(1, int(width * .94))
    cutout.thumbnail((target_width, max(1, int(height * .62))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (*tuple(int(v) for v in background), 255))
    left = (width - cutout.width) // 2
    top = int(height * .34)
    canvas.alpha_composite(cutout, (left, min(top, height - cutout.height)))
    return canvas.convert("RGB")


def garment_design_conditioning(
    garment,
    *,
    exclude_region=None,
    exclude_region_size=None,
):
    """Return a tight upper-garment reference without a portrait backdrop.

    A passport-shaped reference leaves most pixels as the selected blue
    backdrop. Qwen can then mistake that large field for the shirt's base
    colour. Keep only the collar, shoulders, sleeves, placket and upper fabric
    on a narrow neutral product-photo margin so garment pixels dominate visual
    conditioning for every uploaded template.
    """
    conditioning_garment = garment
    badge_excluded = False
    if exclude_region and exclude_region_size:
        try:
            _, region_top, _, _ = (float(value) for value in exclude_region)
            reference_width, reference_height = (
                float(value) for value in exclude_region_size
            )
            mapped_top = int(round(
                region_top * garment.height / max(reference_height, 1.0)
            ))
            # A repaired badge rectangle is itself a visual instruction: Qwen
            # reproduces it as an extra stitched panel. When VL has measured a
            # badge, give Qwen the untouched garment only down to just above
            # that region. The collar, shoulders, plaid scale, trim and pocket
            # opening remain literal source pixels, while neither the badge nor
            # a synthetic replacement patch can enter conditioning.
            safety_gap = max(3, int(round(garment.height * .008)))
            crop_bottom = max(1, mapped_top - safety_gap)
            if crop_bottom >= int(round(garment.height * .28)):
                conditioning_garment = garment.crop(
                    (0, 0, garment.width, crop_bottom)
                )
                badge_excluded = True
        except (TypeError, ValueError):
            conditioning_garment = garment

    cutout = opaque_garment_cutout(conditioning_garment)
    # Preserve the cleaned soft edge. The RGB underneath partial alpha has
    # already been extended from the fabric core, so it cannot introduce a
    # white/gray keyline on this neutral reference canvas.
    bounds = cutout.getchannel("A").getbbox()
    if not bounds:
        raise ValueError("Empty garment cutout")
    cutout = cutout.crop(bounds)
    if not badge_excluded:
        upper_chest_bottom = max(
            1, min(cutout.height, int(round(cutout.height * .62)))
        )
        cutout = cutout.crop((0, 0, cutout.width, upper_chest_bottom))
    pad = max(4, int(round(min(cutout.size) * .025)))
    canvas = Image.new(
        "RGBA",
        (cutout.width + pad * 2, cutout.height + pad * 2),
        (245, 245, 245, 255),
    )
    canvas.alpha_composite(cutout, (pad, pad))
    return canvas.convert("RGB")


def build_rough_composite(
    person, garment, labels, face_box, background, *, expect_tied_hair=False,
    minimum_neck_gap_ratio=0.0,
):
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
    face_w = x2 - x1
    face_h = y2 - y1
    face_center_x = (x1 + x2) * .5
    clothing = np.isin(labels, (5, 6, 7, 10, 11, 12))
    arms = np.isin(labels, (14, 15))
    # Measure the person's actual shoulder/upper-chest span. Using every
    # clothing pixel also included lower sleeves and flared outerwear, which
    # inflated the passport body to four or more face widths. The template is
    # fitted only to this source-relative upper band and never supplies a body
    # silhouette of its own.
    shoulder_top = max(0, int(round(y2 - face_h * .05)))
    shoulder_bottom = min(labels.shape[0], int(round(y2 + face_h * .65)))
    shoulder_band = (clothing | arms)[shoulder_top:shoulder_bottom]
    shoulder_row_widths = []
    for row in shoulder_band:
        columns = np.flatnonzero(row)
        if columns.size >= max(8, int(face_w * .20)):
            shoulder_row_widths.append(float(columns[-1] - columns[0] + 1))
    if not shoulder_row_widths:
        for row in (clothing | arms):
            columns = np.flatnonzero(row)
            if columns.size >= max(8, int(face_w * .20)):
                shoulder_row_widths.append(float(columns[-1] - columns[0] + 1))
    if not shoulder_row_widths:
        raise ValueError("Cannot locate source clothing for placement")
    # Measure complete row spans rather than pooling all column coordinates.
    # Pooling biases the result toward the narrow neck/upper rows and made the
    # fitted torso too slim, so a correctly sized head appeared oversized.
    # The 85th percentile follows the uploaded upper chest without allowing a
    # single extended sleeve or hand to define the body width.
    measured_width = float(np.percentile(shoulder_row_widths, 85))
    # The uploaded person's visible upper torso is the sizing authority. The
    # garment template supplies appearance, not a new body width or silhouette.
    source_body_to_face = measured_width / max(face_w, 1.0)
    original_size = person.size
    central_clothing = clothing[:, max(0, int(face_center_x - face_w * .38)):
                                min(labels.shape[1], int(face_center_x + face_w * .38) + 1)]
    central_rows = np.flatnonzero(np.any(central_clothing, axis=1))
    central_rows = central_rows[central_rows >= int(y2 - face_h * .08)]
    source_collar_y = float(central_rows[0]) if central_rows.size else y2 + face_h * .22
    source_neck_gap_ratio = float(np.clip((source_collar_y - y2) / max(face_h, 1.0), .06, .32))
    effective_neck_gap_ratio = max(
        source_neck_gap_ratio,
        float(np.clip(minimum_neck_gap_ratio, 0.0, .32)),
    )
    base = np.array(person)
    # Keep the source background coherent around the complete head and hair.
    # Partially replacing only SCHP label 0 produced a blue cutout mixed with
    # green scenery fragments that the parser had mislabeled as hair/person;
    # Qwen then preserved those fragments as head detail. The single Qwen edit
    # already owns exact-backdrop generation, matching the Enhancement route.
    # Remove all parsed source clothing before placing the target garment.
    # Keeping the old collar/vest in a central neck corridor leaves dark blocks
    # beside the fitted collar, which Edit Plus faithfully reproduces. Genuine
    # neck skin and tied-hair pixels are restored later through their dedicated
    # colour/anatomy masks, so source garment pixels are never needed as a gap
    # filler here.
    clear_clothing = clothing | arms
    base[clear_clothing] = background
    # Upper arms are natural body anatomy and should remain intact rather
    # than being turned into blue cutout holes. Lower hands are framed out.
    # Fit the garment proportionally, then clip it to the uploaded person's
    # clothing silhouette. The former row-by-row stretch turned sleeves and
    # cuffs into broad horizontal bands; Qwen interpreted those bands as bent
    # arms and generated a seated pose. A proportional upper-shirt placement
    # retains the template construction while the source mask remains the body
    # width and shoulder authority.
    source_fit = cv2.morphologyEx(
        (clothing | arms).astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
    ).astype(bool)
    # Human parsing fragments sleeves, vest openings and crossed seams into
    # separate islands. Those gaps become visible backdrop holes in the rough
    # guide and Qwen then invents an undershirt or a detached garment. Build a
    # row-wise envelope from the uploaded person's own clothing/upper-arm
    # silhouette. This preserves their measured pose and width while making
    # the target shirt continuous across the upper torso.
    body_envelope = np.zeros_like(source_fit)
    supported_rows = np.flatnonzero(np.any(source_fit, axis=1))
    for row in supported_rows:
        columns = np.flatnonzero(source_fit[row])
        if columns.size >= 2:
            left = int(np.percentile(columns, 2))
            right = int(np.percentile(columns, 98))
            body_envelope[row, left:right + 1] = True
    body_envelope = cv2.morphologyEx(
        body_envelope.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13)), iterations=1,
    ).astype(bool)
    source_fit |= body_envelope
    # Keep the measured shoulder width centered beneath the uploaded face.
    # A source pose or asymmetric lower sleeve can otherwise shift the
    # template placket away from the neck even when the final crop is centered.
    desired_fit_width = max(3, int(round(measured_width)))
    fit_left_limit = int(round(face_center_x - desired_fit_width * .5))
    fit_left_limit = max(0, min(labels.shape[1] - desired_fit_width, fit_left_limit))
    fit_right_limit = min(labels.shape[1], fit_left_limit + desired_fit_width)
    source_fit[:, :fit_left_limit] = False
    source_fit[:, fit_right_limit:] = False
    fit_rows = np.flatnonzero(np.any(source_fit, axis=1))
    template = np.asarray(rgba.convert("RGBA"))
    template_alpha = template[:, :, 3] > 16
    template_rows = np.flatnonzero(np.any(template_alpha, axis=1))
    if not fit_rows.size or not template_rows.size:
        raise ValueError("Cannot map uniform to source body silhouette")

    fit_top, fit_bottom = int(fit_rows[0]), int(fit_rows[-1])
    # Keep the uploaded person's neckline/body geometry. Only pull a garment
    # upward when SCHP leaves an abnormally large gap below the detected face;
    # placing the template inside the jaw produces pasted collar wedges and an
    # artificial neck in the Qwen conditioning image.
    max_gap_top = int(round(y2 + face_h * .10))
    if fit_top > max_gap_top:
        fit_top = max_gap_top
    # A raised source collar can hide nearly the entire neck even when the
    # requested uniform has an ordinary folded collar. Move only the target
    # garment down by the missing face-relative distance so the visual guide
    # and prompt agree on the intended jaw-to-collar transition.
    collar_drop = int(round(
        max(0.0, effective_neck_gap_ratio - source_neck_gap_ratio) * face_h
    ))
    fit_top = min(fit_bottom - 1, fit_top + collar_drop)
    fit_columns = np.flatnonzero(np.any(source_fit, axis=0))
    fit_left, fit_right = int(fit_columns[0]), int(fit_columns[-1])
    fit_width = fit_right - fit_left + 1
    fit_height = fit_bottom - fit_top + 1
    scale = fit_width / max(1, template.shape[1])
    resized_height = max(1, int(round(template.shape[0] * scale)))
    # Resize RGBA in premultiplied space. Straight-alpha interpolation mixes
    # RGB from fully transparent catalog-background pixels into the garment,
    # recreating a white outline even when the source cutout was clean.
    template_alpha_float = template[:, :, 3].astype(np.float32) / 255.0
    template_premultiplied = (
        template[:, :, :3].astype(np.float32)
        * template_alpha_float[:, :, None]
    )

    def resize_premultiplied(target_width, target_height, interpolation):
        resized_alpha = cv2.resize(
            template_alpha_float,
            (target_width, target_height),
            interpolation=interpolation,
        )
        resized_premultiplied = cv2.resize(
            template_premultiplied,
            (target_width, target_height),
            interpolation=interpolation,
        )
        alpha_safe = np.maximum(resized_alpha[:, :, None], 1.0 / 255.0)
        resized_rgb = np.clip(
            resized_premultiplied / alpha_safe, 0.0, 255.0,
        )
        resized_rgb[resized_alpha <= 1.0 / 255.0] = 0.0
        return np.dstack((
            np.rint(resized_rgb).astype(np.uint8),
            np.rint(resized_alpha * 255.0).clip(0, 255).astype(np.uint8),
        ))

    resized = resize_premultiplied(
        fit_width, resized_height, cv2.INTER_AREA,
    )
    if resized_height < fit_height:
        # Resize the original premultiplied template directly to avoid a
        # second alpha/color interpolation pass.
        resized = resize_premultiplied(
            fit_width, fit_height, cv2.INTER_LINEAR,
        )
    else:
        # Passport framing needs only the collar, shoulders and upper chest.
        # Discard the template's lower torso rather than compressing it upward.
        resized = resized[:fit_height]

    warped = base.copy()
    placed_rgb = resized[:, :, :3]
    placed_alpha = resized[:, :, 3] > 16
    target_fit = source_fit[fit_top:fit_top + resized.shape[0], fit_left:fit_right + 1]
    # White collars and deep neck shadows are frequently classified as
    # background rather than clothing.  Fill only the anatomical collar
    # corridor in the rough guide so the garment reaches the neck; otherwise
    # shortening the protected source neck exposes a blue gap and teaches
    # Qwen to generate a floating head or an elongated neck.
    local_y, local_x = np.indices(target_fit.shape)
    global_y = local_y + fit_top
    global_x = local_x + fit_left
    # Admit the rear collar from the lifted garment top. The protected source
    # head is composited afterward, so this can sit naturally behind the neck
    # without covering the jaw or face.
    collar_top = fit_top
    collar_bottom = min(fit_bottom, int(round(y2 + face_h * .34)))
    collar_progress = np.clip(
        (global_y - collar_top) / max(1.0, collar_bottom - collar_top), 0.0, 1.0,
    )
    collar_half_width = face_w * (.30 + .30 * collar_progress)
    collar_corridor = (
        (global_y >= collar_top)
        & (global_y <= collar_bottom)
        & (np.abs(global_x - ((x1 + x2) * .5)) <= collar_half_width)
    )
    target_fit = target_fit | collar_corridor
    source_fit[fit_top:fit_top + resized.shape[0], fit_left:fit_right + 1] |= target_fit & placed_alpha
    active = target_fit & placed_alpha
    target = warped[fit_top:fit_top + resized.shape[0], fit_left:fit_right + 1]
    target[active] = placed_rgb[active]

    # Feather only the outer guide boundary. Qwen receives the untouched
    # template separately and regenerates the final fabric rather than copying
    # this deliberately rough mapping.
    fit_alpha = cv2.GaussianBlur(source_fit.astype(np.float32), (0, 0), .65)
    fit_alpha = np.clip(fit_alpha, 0.0, 1.0)[:, :, None]
    canvas_rgb = np.rint(
        warped.astype(np.float32) * fit_alpha + base.astype(np.float32) * (1.0 - fit_alpha)
    ).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(canvas_rgb, "RGB")
    # Restore only parsed person regions over the uniform guide. Restoring the
    # whole top half also restored the outdoor scene, causing Qwen to ignore
    # the requested studio background.
    # Do not trust every SCHP hair-class pixel below the jaw. Dark vests and
    # jackets are commonly mislabeled as hair and previously survived as a
    # rectangular block beside a real ponytail. Crown/ear-level hair is safe;
    # lower tied hair is admitted separately through the measured side path.
    # Start with head/identity classes only. Keeping every parsed body class
    # copied lower arms, shoes and occasional parser noise into a passport
    # guide as detached fragments. The fitted garment already owns the torso;
    # source arms are admitted separately only within the upper-chest frame.
    protect_subject = np.isin(labels, (1, 4, 13))
    # SCHP commonly classifies the exposed neck as source clothing. Recover
    # only skin-like pixels in the anatomical bridge under the detected face,
    # constrained to the central collar opening so that front collar points,
    # lapels and placket are never covered by skin.
    rgb = np.asarray(person)
    yy, xx = np.indices(labels.shape)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Clear lenses, thin frames and lens reflections are often parsed as face
    # holes or background. The eye band is entirely inside the anatomical face
    # area, so preserving its source pixels cannot reintroduce scenery.
    eye_band = (
        (xx >= int(x1))
        & (xx <= int(x2))
        & (yy >= int(y1 + face_h * .12))
        & (yy <= int(y1 + face_h * .56))
    )
    face_oval = np.zeros(labels.shape, dtype=np.uint8)
    cv2.ellipse(
        face_oval,
        (int((x1 + x2) * .5), int(y1 + face_h * .52)),
        (max(1, int(face_w * .57)), max(1, int(face_h * .58))),
        0, 0, 360, 1, -1,
    )
    # Recover parser misses around dark skin, temples and ears without copying
    # the complete geometric oval. The old unconditional oval also copied the
    # source wall beside both cheeks, creating the pale jaw halo that Qwen then
    # treated as part of the person. Require recovered face pixels to be both
    # near the parsed face and close to its measured LAB skin colour.
    face_oval_bool = face_oval.astype(bool)
    parsed_face = labels == 13
    face_kernel_w = max(3, int(round(face_w * .20)) | 1)
    face_kernel_h = max(3, int(round(face_h * .16)) | 1)
    near_parsed_face = cv2.dilate(
        parsed_face.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (face_kernel_w, face_kernel_h),
        ),
        iterations=1,
    ).astype(bool)
    face_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    parsed_face_pixels = face_lab[parsed_face & face_oval_bool]
    if len(parsed_face_pixels) >= 32:
        median_face_lab = np.median(parsed_face_pixels, axis=0)
        face_colour_distance = np.linalg.norm(
            face_lab - median_face_lab[None, None, :], axis=2,
        )
        recovered_face = (
            face_oval_bool & near_parsed_face & (face_colour_distance < 48.0)
        )
        protect_subject |= recovered_face
    # Lens glare can be much brighter than skin, so preserve parser holes only
    # inside the central eye band and next to the known face. This cannot pull
    # scenery from outside the detected face box.
    protect_subject |= eye_band & face_oval_bool & near_parsed_face

    # Keep only source upper-arm skin that can occur in an upper-chest crop.
    # Lower arms/hands otherwise survive at the bottom corners as disconnected
    # matte islands even though the requested passport portrait excludes them.
    upper_arm_bottom = int(round(y2 + face_h * .82))
    protect_subject |= arms & (yy <= upper_arm_bottom)

    # Recover fine hair and accessories missed by SCHP only when their colour
    # is connected to the parsed source hair. This closes small holes and
    # keeps wispy edges while rejecting unrelated dark scenery farther away.
    parsed_hair = labels == 2
    measured_hair_mask = parsed_hair.copy()
    protect_subject |= parsed_hair & (yy <= int(y2 + face_h * .10))
    hair_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if np.count_nonzero(parsed_hair) >= 128:
        shaded = parsed_hair & (hair_lab[:, :, 0] < np.percentile(hair_lab[:, :, 0][parsed_hair], 72))
        sample = hair_lab[shaded if np.count_nonzero(shaded) >= 64 else parsed_hair]
        hair_colour = np.median(sample, axis=0)
        hair_distance = np.linalg.norm(hair_lab - hair_colour[None, None, :], axis=2)
        # SCHP can retain only a narrow strip at the part/crown on slicked or
        # tightly tied hair. Recover the remaining colour-connected cap over a
        # wider face-relative radius; the anatomical head bounds below keep
        # dark scenery and clothing out.
        kernel_size = max(9, int(round(face_w * .65)) | 1)
        near_hair = cv2.dilate(
            parsed_hair.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        ).astype(bool)
        head_and_sides = (
            (xx >= int(x1 - face_w * 1.05))
            & (xx <= int(x2 + face_w * 1.05))
            & (yy >= max(0, int(y1 - face_h * .82)))
            & (yy <= min(labels.shape[0] - 1, int(y2 + face_h * 1.20)))
        )
        # General edge recovery is only for the crown and ear-level hair. Dark
        # source clothing below the jaw can be connected to a real ponytail and
        # has nearly identical colour; admitting it here created a rectangular
        # vest patch behind the generated head. Lower tied hair is handled by
        # the component-checked ponytail path below.
        recovered_hair = near_hair & head_and_sides & (yy <= int(y2 + face_h * .10)) & (
            (hair_distance < 62.0)
            | ((hsv[:, :, 2] < 112) & (hsv[:, :, 1] < 170))
        )
        # A nearby dark door edge, frame or foliage patch can have the same LAB
        # colour as hair. Keep recovered pixels only in connected components
        # that actually touch parsed source hair; proximity alone is not proof
        # of hairstyle geometry.
        hair_candidates = parsed_hair | recovered_hair
        component_count, component_labels, _, _ = cv2.connectedComponentsWithStats(
            hair_candidates.astype(np.uint8), connectivity=8,
        )
        connected_hair = np.zeros_like(parsed_hair)
        for component in range(1, component_count):
            component_mask = component_labels == component
            if np.count_nonzero(component_mask & parsed_hair):
                connected_hair |= component_mask
        protect_subject |= connected_hair & (yy <= int(y2 + face_h * .10))
        measured_hair_mask |= connected_hair
        # Hair parsers often stop at the ears and omit a side ponytail. Detect
        # a side only when the lower hair is genuinely connected to the parsed
        # crown. Never paint a synthetic connector: Qwen interprets a straight
        # guide stroke as a ribbon, strap, or gray artificial ponytail.
        lower_hair_band = (
            (yy >= int(y2 - face_h * .03))
            # Detect ponytail asymmetry cleanly above lower clothing.
            & (yy <= int(y2 + face_h * .46))
        )
        side_corridors = (
            ((xx >= int(x1 - face_w * .48)) & (xx <= int(x1 + face_w * .12)))
            | ((xx >= int(x2 - face_w * .12)) & (xx <= int(x2 + face_w * .48)))
        )
        tied_hair_distance = 58.0 if expect_tied_hair else 38.0
        tied_hair_value = 105 if expect_tied_hair else 72
        lower_hair = lower_hair_band & side_corridors & (
            (hair_distance < tied_hair_distance)
            | ((hsv[:, :, 2] < tied_hair_value) & (hsv[:, :, 1] < 175))
        )
        centre_x = (x1 + x2) * .5
        # Determine an occluded ponytail from its root immediately beside and
        # below an ear, before dark clothing enters the corridor. SCHP often
        # misses black hair entirely in outdoor portraits, while this narrow
        # anatomical band remains independent of the vest/garment below.
        root_band = (
            (yy >= int(y2 - face_h * .03))
            & (yy <= int(y2 + face_h * .12))
        )
        left_root_corridor = (
            (xx >= int(x1 - face_w * .35))
            & (xx <= int(x1 + face_w * .08))
        )
        right_root_corridor = (
            (xx >= int(x2 - face_w * .08))
            & (xx <= int(x2 + face_w * .35))
        )
        dark_root = root_band & (
            (hair_distance < tied_hair_distance)
            | ((hsv[:, :, 2] < tied_hair_value) & (hsv[:, :, 1] < 175))
        )
        left_root_pixels = int(np.count_nonzero(dark_root & left_root_corridor))
        right_root_pixels = int(np.count_nonzero(dark_root & right_root_corridor))
        left_tail_pixels = int(np.count_nonzero(lower_hair & (xx < centre_x)))
        right_tail_pixels = int(np.count_nonzero(lower_hair & (xx >= centre_x)))
        minimum_tail_area = max(20, int(round(face_w * face_h * .0012)))
        root_side = "unknown"
        if right_root_pixels >= minimum_tail_area and right_root_pixels > left_root_pixels * 2.0:
            root_side = "viewer-right"
        elif left_root_pixels >= minimum_tail_area and left_root_pixels > right_root_pixels * 2.0:
            root_side = "viewer-left"
        detected_ponytail_side = "unknown"
        bilateral_tail = bool(
            expect_tied_hair
            and left_root_pixels >= minimum_tail_area
            and right_root_pixels >= minimum_tail_area
            and left_tail_pixels >= minimum_tail_area
            and right_tail_pixels >= minimum_tail_area
            and max(left_tail_pixels, right_tail_pixels)
            <= max(1, min(left_tail_pixels, right_tail_pixels)) * 2.25
        )
        if bilateral_tail:
            detected_ponytail_side = "bilateral"
        elif right_tail_pixels >= minimum_tail_area and right_tail_pixels > left_tail_pixels * 1.45:
            detected_ponytail_side = "viewer-right"
        elif left_tail_pixels >= minimum_tail_area and left_tail_pixels > right_tail_pixels * 1.45:
            detected_ponytail_side = "viewer-left"
        if detected_ponytail_side != "unknown":
            side_mask = (
                side_corridors
                if detected_ponytail_side == "bilateral"
                else xx >= centre_x
                if detected_ponytail_side == "viewer-right"
                else xx < centre_x
            )
            # Retain the visible tail root and upper bundle. Below roughly one
            # quarter face height, dark source outerwear joins the tail in many
            # portraits and cannot be separated reliably by colour alone.
            # Qwen receives the measured side/extent and completes the lower
            # tail without inheriting those clothing pixels.
            full_tail_band = (yy >= int(y2 - face_h * .03)) & (yy <= int(y2 + face_h * .24))
            retained_lower_hair = full_tail_band & side_corridors & (
                (hair_distance < tied_hair_distance)
                | ((hsv[:, :, 2] < tied_hair_value) & (hsv[:, :, 1] < 175))
            ) & side_mask
            component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                retained_lower_hair.astype(np.uint8), connectivity=8,
            )
            filtered_tail = np.zeros_like(retained_lower_hair)
            for component in range(1, component_count):
                if component_stats[component, cv2.CC_STAT_AREA] >= minimum_tail_area:
                    filtered_tail |= component_labels == component
            connected_to_crown = cv2.dilate(
                parsed_hair.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(5, int(round(face_w * .34)) | 1),
                     # A tied ponytail can disappear behind the ear/neck and
                     # re-emerge near the collar. The old shallow vertical
                     # radius rejected that real occluded continuation. Keep
                     # the horizontal radius narrow, but bridge up to roughly
                     # one third of a face height vertically for connectivity
                     # testing only; no synthetic pixels are drawn.
                     max(5, int(round(face_h * .86)) | 1)),
                ),
                iterations=1,
            ).astype(bool)
            if np.count_nonzero(filtered_tail & connected_to_crown):
                protect_subject |= filtered_tail
                measured_hair_mask |= filtered_tail
            elif expect_tied_hair and (
                root_side == detected_ponytail_side
                or detected_ponytail_side == "bilateral"
            ):
                # Preserve only the small visible root. Qwen receives its side
                # and completes the tail; copying the larger colour-selected
                # region would also copy a dark vest into the uniform guide.
                root_corridor = (
                    left_root_corridor | right_root_corridor
                    if detected_ponytail_side == "bilateral"
                    else right_root_corridor
                    if root_side == "viewer-right"
                    else left_root_corridor
                )
                protect_subject |= dark_root & root_corridor
                measured_hair_mask |= dark_root & root_corridor
            else:
                detected_ponytail_side = "unknown"
    else:
        detected_ponytail_side = "unknown"
    neck_top = y2 - face_h * .14
    # Stop source-neck restoration at the uploaded garment's top edge.  The
    # previous fixed 0.32-face-height extension ran through high collars on
    # tightly framed portraits, producing a long rectangular neck that Qwen
    # then treated as intentional geometry.
    # Preserve only a very short, face-scaled bridge. Its lower edge is
    # independent of the lifted rear-collar row; coupling it to fit_top made
    # the last real skin row almost transparent on close portraits.
    # Preserve the real short neck transition instead of ending the source at
    # the jaw. A 4.5%-face-height strip left a bright garment gap directly
    # below the chin, forcing Qwen to reconstruct the entire face and neck.
    # Candidate pixels must remain close to the measured facial skin colour,
    # so red collars or dark source clothing cannot extend this bridge.
    face_skin = (labels == 13) & face_oval.astype(bool)
    face_skin_pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[face_skin]
    face_skin_lab = np.median(face_skin_pixels, axis=0) if len(face_skin_pixels) >= 32 else None
    # Retain enough real skin to connect the jaw to the collar, but do not
    # expose a tall source-neck column in the conditioning image. Qwen treats
    # that guide geometry literally and otherwise lengthens the neck again.
    # Match the uploaded portrait instead of imposing one fixed neck length on
    # every person. The measured ratio is bounded above when calculated, and
    # the colour/anatomy gates below still prevent source clothing from being
    # copied into the collar opening.
    collar_side_y = y2 + face_h * effective_neck_gap_ratio
    scan_bottom = y2 + face_h * .32
    neck_colour_candidate = (
        (np.abs(xx - ((x1 + x2) * .5)) <= face_w * .24)
        & (yy >= int(neck_top))
        & (yy <= int(scan_bottom))
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] >= 35)
        & (hsv[:, :, 2] <= 245)
        & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
    )
    if face_skin_lab is not None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Neck skin is commonly much darker than the face because it sits
        # under the jaw. The former tight LAB threshold rejected that real
        # shadowed skin and exposed the patterned rough garment through the
        # collar opening, forcing Qwen to rebuild the entire head/neck. The
        # anatomical corridor and HSV skin guard still exclude collar fabric.
        neck_colour_candidate &= (
            np.linalg.norm(lab - face_skin_lab[None, None, :], axis=2) < 65.0
        )
    # A folded collar begins beside the neck before reaching the lower centre
    # opening. Measure those independently: garment placement uses the side
    # height, while genuine source-coloured skin may continue farther down in
    # the V/opening. This prevents plaid pixels from appearing inside the neck
    # without turning the entire source collar into an elongated skin column.
    row_counts = np.count_nonzero(neck_colour_candidate, axis=1)
    valid_neck_rows = np.flatnonzero(
        (row_counts >= max(3, int(round(face_w * .10))))
        & (np.arange(labels.shape[0]) >= int(y2))
        & (np.arange(labels.shape[0]) <= int(scan_bottom))
    )
    detected_skin_bottom = (
        float(valid_neck_rows[-1] + 1) if valid_neck_rows.size else collar_side_y
    )
    # The skin-colour scan may continue through a shadowed shirt opening and
    # classify source-collar pixels as neck. Letting that observation extend
    # all the way to the scan limit doubled the jaw-to-collar gap in the fitted
    # seed (for example 0.145 -> 0.286 face heights). The source collar anchor
    # is the geometry authority; retain only a small face-relative allowance
    # below it for antialiasing and a natural collar overlap.
    maximum_neck_extension = face_h * .045
    neck_bottom = min(
        scan_bottom,
        max(collar_side_y, min(detected_skin_bottom, collar_side_y + maximum_neck_extension)),
    )
    neck_t = np.clip((yy - neck_top) / max(1.0, neck_bottom - neck_top), 0.0, 1.0)
    # Follow the jaw near the top, then taper inward through the collar opening.
    # CODEX NOTE: Using .28-.12 pinched the neck to 16% face width at the bottom,
    # leaving blue side gutters that bleed into the collar during blurring.
    neck_half_width = face_w * (.34 - .10 * neck_t)
    skin_neck = (
        neck_colour_candidate
        & (np.abs(xx - ((x1 + x2) * .5)) <= neck_half_width)
        & (yy <= int(neck_bottom))
    )
    # Parsing and deep collar shadows often remove a horizontal strip of the
    # real neck. Close only genuine skin-coloured source pixels. An earlier
    # unconditional rectangular bridge produced an unnaturally long, narrow
    # neck and taught Qwen to preserve that faulty geometry.
    kernel_w = max(3, int(round(face_w * .09)) | 1)
    kernel_h = max(3, int(round(face_h * .10)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    skin_neck = cv2.morphologyEx(
        skin_neck.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
    ).astype(bool)
    neck_alpha = cv2.GaussianBlur(skin_neck.astype(np.float32), (0, 0), 1.35)
    # Keep genuine central neck pixels opaque immediately below the jaw; the
    # blur alone makes a short bridge nearly transparent and exposes the rough
    # garment or backdrop through skin. Edge pixels remain softly feathered.
    neck_alpha = np.maximum(neck_alpha, skin_neck.astype(np.float32))
    # Let the target collar cover the final few neck rows. A hard opaque neck
    # bottom creates a rectangular cut line that Qwen preserves as an overly
    # long pasted neck.
    collar_overlap_fade = np.clip(
        (neck_bottom - yy) / max(1.0, face_h * .025), 0.0, 1.0,
    ).astype(np.float32)
    neck_alpha *= collar_overlap_fade
    neck_alpha = np.rint(np.clip(neck_alpha, 0.0, 1.0) * 255).astype(np.uint8)
    # Never restore pixels that are already the selected flat backdrop. SCHP
    # can mislabel a blue notch beside the neck as face, hair, ear or skin;
    # pasting that semantic false positive over the target garment cuts a hole
    # through the collar. This comparison is against the caller-selected color
    # for the current portrait, not a fixed backdrop assumption.
    selected_background = np.asarray(background, dtype=np.float32)
    backdrop_like = np.linalg.norm(
        rgb.astype(np.float32) - selected_background[None, None, :], axis=2,
    ) <= 26.0
    protect_subject &= ~backdrop_like
    neck_alpha[backdrop_like] = 0
    # Feather inward from a one-pixel erosion. Blurring the unmodified hard
    # mask spreads source edge colours outward and produces white/gray halos;
    # inward feathering keeps the selected backdrop outside the subject while
    # retaining a soft photographic transition inside hair, ears and arms.
    hard_protect = protect_subject.astype(np.uint8)
    matte_radius = max(1, min(3, int(round(face_w * .012))))
    matte_kernel_size = matte_radius * 2 + 1
    eroded_protect = cv2.erode(
        hard_protect,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (matte_kernel_size, matte_kernel_size),
        ),
        iterations=1,
    )
    soft_protect = cv2.GaussianBlur(
        eroded_protect.astype(np.float32), (0, 0), .55 + matte_radius * .12,
    )
    soft_protect[hard_protect == 0] = 0.0
    soft_protect[eroded_protect != 0] = 1.0
    subject_alpha = np.rint(
        np.clip(soft_protect, 0.0, 1.0) * 255,
    ).astype(np.uint8)
    protect = np.maximum(subject_alpha, neck_alpha)
    canvas = Image.composite(person, canvas, Image.fromarray(protect))
    hair_y, hair_x = np.where(measured_hair_mask)
    measured_hair_end_ratio = None
    measured_left_hair_end_ratio = None
    measured_right_hair_end_ratio = None
    measured_hair_extent = "unknown"
    if hair_y.size >= 96:
        # Use the 98th percentile rather than one lowest pixel so a parser
        # speck cannot turn short hair into a false long-hair instruction.
        measured_hair_end_ratio = float(
            (np.percentile(hair_y, 98) - y2) / max(face_h, 1.0)
        )
        left_y = hair_y[hair_x < face_center_x]
        right_y = hair_y[hair_x >= face_center_x]
        if left_y.size >= 32:
            measured_left_hair_end_ratio = float(
                (np.percentile(left_y, 98) - y2) / max(face_h, 1.0)
            )
        if right_y.size >= 32:
            measured_right_hair_end_ratio = float(
                (np.percentile(right_y, 98) - y2) / max(face_h, 1.0)
            )
        if measured_hair_end_ratio <= -0.08:
            measured_hair_extent = "ends above the jaw"
        elif measured_hair_end_ratio <= 0.16:
            measured_hair_extent = "ends around the jaw or upper neck"
        elif measured_hair_end_ratio <= 0.52:
            measured_hair_extent = "ends around the neck-to-shoulder line"
        else:
            measured_hair_extent = "extends visibly below the shoulder line"
    return canvas, {"garment_box": [int(fit_columns[0]), fit_top,
                                     int(fit_columns[-1] + 1), int(fit_bottom + 1)],
                    "source_canvas_size": list(original_size),
                    "expanded_canvas_size": list(person.size),
                    "source_body_width_px": round(measured_width, 2),
                    "source_face_width_px": round(face_w, 2),
                    "source_body_to_face_width": round(source_body_to_face, 3),
                    "garment_to_face_width": round(measured_width / max(face_w, 1.0), 3),
                    "source_neck_gap_ratio": round(effective_neck_gap_ratio, 3),
                    "measured_source_neck_gap_ratio": round(source_neck_gap_ratio, 3),
                    "detected_ponytail_side": detected_ponytail_side,
                    "detected_tied_section_count": (
                        2 if detected_ponytail_side == "bilateral"
                        else 1 if detected_ponytail_side in {"viewer-left", "viewer-right"}
                        else 0
                    ),
                    "detected_tied_section_sides": (
                        "viewer-left and viewer-right"
                        if detected_ponytail_side == "bilateral"
                        else detected_ponytail_side
                    ),
                    "measured_hair_extent": measured_hair_extent,
                    "measured_hair_end_ratio": (
                        round(measured_hair_end_ratio, 3)
                        if measured_hair_end_ratio is not None else None
                    ),
                    "measured_left_hair_end_ratio": (
                        round(measured_left_hair_end_ratio, 3)
                        if measured_left_hair_end_ratio is not None else None
                    ),
                    "measured_right_hair_end_ratio": (
                        round(measured_right_hair_end_ratio, 3)
                        if measured_right_hair_end_ratio is not None else None
                    ),
                    "mode": "rough_placement_for_qwen_refinement", "not_final": True}


def build_person_backdrop_guide(
    person, labels, background, remove_clothing=False, face_box=None,
):
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
    neck_bridge = np.zeros(labels.shape, dtype=bool)
    if face_box is not None:
        # SCHP sometimes marks the exposed neck above a standing collar as
        # background. Preserve only skin-coloured pixels in the anatomical
        # bridge below the detected face so the edit canvas never contains a
        # floating head or teaches Qwen to invent a new neck.
        x1, y1, x2, y2 = (float(value) for value in face_box)
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        yy, xx = np.indices(labels.shape)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        neck_bridge = (
            (xx >= int(x1 + face_w * .20))
            & (xx <= int(x2 - face_w * .20))
            & (yy >= int(y2 - face_h * .08))
            & (yy <= int(y2 + face_h * .42))
            & (hsv[:, :, 1] >= 22)
            & (hsv[:, :, 2] >= 35)
            & (hsv[:, :, 2] <= 245)
            & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
        )
        subject |= neck_bridge
    if remove_clothing:
        # This is a secondary identity reference, not a body/edit canvas. Keep
        # only the real head, hair, glasses/accessories and recovered neck.
        # Leaving source arms or any garment pixels here gives Qwen a second,
        # conflicting clothing authority and makes it restore the old uniform.
        head_labels = np.isin(labels, (1, 2, 3, 4, 8, 13, 16, 17, 18, 19))
        subject = head_labels | neck_bridge
    # A slightly feathered outer boundary gives Qwen strong, untouched face
    # and hairstyle pixels without teaching it to reproduce a hard pasted
    # silhouette. Interior subject pixels remain exactly the upload.
    alpha = cv2.GaussianBlur(subject.astype(np.float32), (0, 0), 1.15)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    guide = rgb.astype(np.float32) * alpha + backdrop.astype(np.float32) * (1.0 - alpha)
    return Image.fromarray(np.rint(guide).clip(0, 255).astype(np.uint8), "RGB")


def build_neutral_body_fit_guide(person, labels, face_box, background):
    """Preserve the uploaded body fit without showing either source garment.

    A template texture warped into the source silhouette teaches Qwen a
    distorted pattern. Conversely, deleting the clothing leaves a floating
    head and gives the model no shoulder/torso geometry. This guide keeps the
    uploaded person's complete silhouette and softly shaded body volume while
    replacing clothing detail with an achromatic placeholder. The separate
    garment reference remains the only source of uniform colour and pattern.
    """
    person = person.convert("RGB")
    rgb = np.asarray(person, dtype=np.uint8)
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Source parsing dimensions do not match")

    clothing = np.isin(labels, (5, 6, 7, 9, 10, 11, 12))
    rows, cols = np.where(clothing)
    if cols.size < labels.size * .03:
        raise ValueError("Cannot locate source clothing for body-fit guide")

    # Close small parser gaps so source logos/patterns cannot survive inside
    # the placeholder. Preserve the original outer silhouette and folds only
    # as broad luminance changes; all chroma and high-frequency fabric detail
    # are deliberately removed.
    clothing_fit = cv2.morphologyEx(
        clothing.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
    ).astype(bool)
    # A school-ID portrait requires square, level shoulders even when the
    # uploaded snapshot was taken with a slight body lean. Mirror-union the
    # neutral placeholder around the detected face centre. Qwen then receives
    # the uploaded body width but a symmetric shoulder envelope, while the
    # untouched template remains the sole fabric authority.
    face_center_x = (float(face_box[0]) + float(face_box[2])) * .5
    source_y, source_x = np.where(clothing_fit)
    mirrored_x = np.rint(face_center_x * 2.0 - source_x).astype(np.int32)
    valid = (mirrored_x >= 0) & (mirrored_x < clothing_fit.shape[1])
    clothing_fit[source_y[valid], mirrored_x[valid]] = True
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(7.0, min(rgb.shape[:2]) / 32.0)
    broad_light = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if np.any(clothing_fit):
        centre = float(np.median(broad_light[clothing_fit]))
    else:
        centre = 128.0
    neutral = np.clip(132.0 + (broad_light - centre) * .28, 102.0, 164.0)
    neutral_rgb = np.repeat(neutral[:, :, None], 3, axis=2)

    backdrop = np.empty_like(rgb)
    backdrop[:] = tuple(int(value) for value in background)
    subject = labels != 0
    subject_alpha = cv2.GaussianBlur(subject.astype(np.float32), (0, 0), 1.15)
    guide = (
        rgb.astype(np.float32) * subject_alpha[:, :, None]
        + backdrop.astype(np.float32) * (1.0 - subject_alpha[:, :, None])
    )
    clothing_alpha = cv2.GaussianBlur(clothing_fit.astype(np.float32), (0, 0), .8)
    guide = (
        neutral_rgb * clothing_alpha[:, :, None]
        + guide * (1.0 - clothing_alpha[:, :, None])
    )

    # Restore face, hair, skin, ears and visible accessories exactly. These
    # pixels are the identity authority; only the garment area is neutralized.
    protected = np.isin(labels, (1, 2, 3, 4, 8, 13, 14, 15, 16, 17, 18, 19))
    x1, y1, x2, y2 = (float(value) for value in face_box)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    yy, xx = np.indices(labels.shape)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Face parsing often stops exactly at the jaw. Recover only skin-coloured
    # source pixels in the anatomical neck corridor so the neutral guide does
    # not show a floating head or a white wedge above the garment placeholder.
    neck_top = y2 - face_height * .12
    neck_bottom = y2 + face_height * .34
    neck_t = np.clip((yy - neck_top) / max(1.0, neck_bottom - neck_top), 0.0, 1.0)
    neck_half_width = face_width * (.27 - .07 * neck_t)
    skin_neck = (
        (np.abs(xx - ((x1 + x2) * .5)) <= neck_half_width)
        & (yy >= int(neck_top))
        & (yy <= int(neck_bottom))
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] >= 35)
        & (hsv[:, :, 2] <= 245)
        & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
    )
    neck_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(face_width * .08)) | 1),
         max(3, int(round(face_height * .08)) | 1)),
    )
    skin_neck = cv2.morphologyEx(
        skin_neck.astype(np.uint8), cv2.MORPH_CLOSE, neck_kernel, iterations=2,
    ).astype(bool)
    protected |= skin_neck
    protected_alpha = cv2.GaussianBlur(protected.astype(np.float32), (0, 0), .65)
    guide = (
        rgb.astype(np.float32) * protected_alpha[:, :, None]
        + guide * (1.0 - protected_alpha[:, :, None])
    )

    measured_width = float(np.percentile(cols, 98) - np.percentile(cols, 2))
    clothing_rows = np.flatnonzero(np.any(clothing_fit, axis=1))
    clothing_cols = np.flatnonzero(np.any(clothing_fit, axis=0))
    result = Image.fromarray(np.rint(guide).clip(0, 255).astype(np.uint8), "RGB")
    return result, {
        "garment_box": [int(clothing_cols[0]), int(clothing_rows[0]),
                        int(clothing_cols[-1] + 1), int(clothing_rows[-1] + 1)],
        "source_canvas_size": list(person.size),
        "expanded_canvas_size": list(person.size),
        "source_body_width_px": round(measured_width, 2),
        "source_face_width_px": round(face_width, 2),
        "source_body_to_face_width": round(measured_width / face_width, 3),
        "garment_to_face_width": round(measured_width / face_width, 3),
        "mode": "neutral_source_body_fit_for_qwen", "not_final": True,
    }


def build_person_identity_guide(person, labels, face_box):
    """Return an untouched rectangular head/neck crop with no source clothing.

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
    # Stop before the first source-clothing row in the face corridor. A full
    # portrait reference makes Qwen restore the old uniform, while a semantic
    # head cutout creates blue holes and hard seams under the chin. This
    # untouched rectangular crop preserves natural hair/skin boundaries and
    # local context without exposing the source garment.
    clothing = np.isin(labels, (5, 6, 7, 9, 10, 11, 12))
    corridor_left = max(0, int(np.floor(x1 - face_w * .35)))
    corridor_right = min(width, int(np.ceil(x2 + face_w * .35)))
    clothing_rows = np.flatnonzero(np.any(
        clothing[:, corridor_left:corridor_right], axis=1,
    ))
    clothing_rows = clothing_rows[clothing_rows >= int(y2)]
    clothing_start = int(clothing_rows[0]) if clothing_rows.size else height

    hair_rows = np.where(labels == 2)[0]
    hair_bottom = int(hair_rows.max()) if hair_rows.size else 0

    desired_bottom = int(np.ceil(max(content_bottom, y2 + face_h * .42)))
    effective_limit = max(clothing_start, min(height, hair_bottom))
    bottom = min(height, desired_bottom, effective_limit)
    bottom = max(bottom, min(height, int(np.ceil(y2 + face_h * .10))))
    if right - left < 2 or bottom - top < 2:
        return person.convert("RGB").copy()

    crop_rgb = rgb[top:bottom, left:right].copy()
    if hair_bottom > clothing_start:
        crop_clothing = clothing[top:bottom, left:right]
        crop_hair_neck = np.isin(labels[top:bottom, left:right], (1, 2, 4, 13))
        mask_out = crop_clothing & ~crop_hair_neck
        crop_rgb[mask_out] = (232, 232, 232)

    return Image.fromarray(crop_rgb)


def hide_product_rear_collar(garment):
    """Hide the inside rear collar visible only on an unworn product photo.

    A folded shirt photographed without a wearer exposes a horizontal rear
    collar band. If that band is included in the fitted portrait, an image
    editor can reproduce it across the subject's throat as a choker. Detect a
    saturated, horizontally continuous component crossing the garment centre
    near its top and clear only those rows from the conditioning cutout.
    """
    rgba = np.asarray(opaque_garment_cutout(garment)).copy()
    alpha = rgba[:, :, 3]
    bounds = Image.fromarray(alpha).point(lambda value: 255 if value > 16 else 0).getbbox()
    if not bounds:
        return garment.convert("RGBA"), 0
    left, top, right, bottom = bounds
    width = max(1, right - left)
    height = max(1, bottom - top)
    centre_x = (left + right) * .5
    yy, xx = np.indices(alpha.shape)
    hsv = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2HSV)
    search = (
        (alpha > 32)
        & (yy >= top)
        & (yy <= top + int(round(height * .14)))
        & (xx >= centre_x - width * .34)
        & (xx <= centre_x + width * .34)
    )
    saturated = search & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 35)
    central = saturated & (
        (xx >= centre_x - width * .07)
        & (xx <= centre_x + width * .07)
    )
    row_counts = np.count_nonzero(saturated, axis=1)
    central_counts = np.count_nonzero(central, axis=1)
    qualifying = np.flatnonzero(
        (row_counts >= max(8, int(round(width * .18))))
        & (central_counts >= max(4, int(round(width * .07))))
    )
    if not qualifying.size:
        return Image.fromarray(rgba, "RGBA"), 0
    runs = np.split(qualifying, np.where(np.diff(qualifying) > 1)[0] + 1)
    runs = [run for run in runs if run.size]
    # The rear band is the first centre-crossing saturated component below the
    # top of a folded shirt.  The lower front collar leaves are usually larger,
    # so selecting by area removes the wrong part and leaves the throat band in
    # the rough fit.
    selected = min(runs, key=lambda run: int(run[0]))
    maximum_height = max(3, int(round(height * .10)))
    if selected.size > maximum_height:
        selected = selected[:maximum_height]
    remove = np.zeros_like(alpha, dtype=bool)
    for row in selected:
        columns = np.flatnonzero(saturated[row])
        if columns.size:
            remove[row, int(columns[0]):int(columns[-1]) + 1] = True
    remove = cv2.dilate(
        remove.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    ).astype(bool)
    rgba[remove, 3] = 0
    return Image.fromarray(rgba, "RGBA"), int(np.count_nonzero(remove))


def refinement_prompt(
    background, fabric_color_instruction=None, hair_color_instruction=None,
    lighting_instruction=None, skin_tone_instruction=None, identity_instruction=None,
    primary_is_identity: bool = False, single_composite: bool = False,
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
    if single_composite:
        fabric_color_instruction = fabric_color_instruction.replace("image 3", "image 1")
        hair_color_instruction = hair_color_instruction.replace("image 2", "image 1")
        lighting_instruction = lighting_instruction.replace("image 2", "image 1")
        skin_tone_instruction = skin_tone_instruction.replace("image 2", "image 1")
        identity_instruction = identity_instruction.replace("Image 2", "Image 1")
        return (
            "REFINE IMAGE 1 IN PLACE AS A FINISHED PHOTOREALISTIC SCHOOL-ID PASSPORT PORTRAIT. "
            "IMAGE 1 IS THE COMPLETE CANVAS AND ALREADY CONTAINS THE SOURCE IDENTITY, HAIR, POSE AND REQUIRED UNIFORM. "
            "PRESERVE ITS CURRENT CENTERED SINGLE-SUBJECT COMPOSITION AND SILHOUETTE. "
            "Perform photographic restoration in place while retaining the existing garment region. Blend only the narrow neck-to-collar "
            "junction and shoulder boundary into one continuous natural photograph. The head must sit anatomically on the neck, "
            "and the neck must enter the uniform collar naturally with no seam, gap, halo, floating head or pasted-head edge. "
            "Do not redesign "
            "the face, hair, collar, pattern, colors or garment construction already visible in image 1. "
            "Keep image 1's corrected head size and center position. Keep natural child/adult proportions from image 1, with "
            "shoulders broad enough to support the head; do not enlarge the head, shrink the torso, lengthen or stretch the neck. "
            "Keep the full head with headroom and show shoulders and upper chest in a centered camera-facing pose; the frame ends at the upper chest. "
            f"The complete background must remain one flat featureless field of {selected_hex}, {selected_rgb_text}. "
            + lighting_instruction + " " + skin_tone_instruction + " " + identity_instruction + " "
            + hair_color_instruction + " "
            + FABRIC_INSTRUCTION.replace("image 3", "image 1") + fabric_color_instruction + " "
            "Keep every visible plaid line, panel, button, sleeve, collar edge and fabric colour already present in image 1; perform no garment redesign. "
            "Remove badges, emblems, logos, text, necklaces, chains and pendants. "
            "Return a coherent generated photograph, not a collage or pasted cutout."
        )
    if primary_is_identity:
        fabric_color_instruction = fabric_color_instruction.replace("image 3", "image 2")
        hair_color_instruction = hair_color_instruction.replace("image 2", "image 1")
        lighting_instruction = lighting_instruction.replace("image 2", "image 1")
        skin_tone_instruction = skin_tone_instruction.replace("image 2", "image 1")
        return (
            f"HIGHEST PRIORITY BACKGROUND EDIT: generate every pixel outside the person as one edge-to-edge, perfectly flat solid field of exactly {selected_hex}, {selected_rgb_text}. "
            "Copy the solid background already visible in images 1 and 2. Do not copy the original photo background. "
            "Do not generate a room, outdoor scene, wall, bokeh, foliage, building, window, prop, gradient, texture, shadow, vignette or light pattern. "
            "Create exactly one finished school-ID passport photograph of the person in image 1. "
            "Image 1 is the only authority for identity, face, skin, hair, accessories and pose. Image 2 is the only authority for the replacement uniform. "
            "Replace the clothing with image 2's exact uniform while preserving the same person. "
            "Show only the full head with headroom, level camera-facing face, square shoulders and upper chest; no hands or lower body. "
            "Preserve exact face, expression, complexion, hairline, hairstyle, hair length, parting, earrings and visible hair accessories from image 1. "
            "Improve detail and neutralize harsh light without recreating or beautifying the face or hair. "
            "Keep the selected background color strictly behind the person; no blue/cyan spill, shadow, rim, reflection or color cast on skin, hair or uniform. "
            "Remove necklaces, chains, pendants, lockets, badges, logos and text. Keep the neck-to-collar connection natural. "
            + lighting_instruction + " " + skin_tone_instruction + " " + identity_instruction + " "
            + hair_color_instruction + " "
            + "Copy image 2's exact color, pattern scale, weave, layers, seams and collar with natural folds. "
            + fabric_color_instruction + " "
            f"FINAL CHECK: the background must contain only the same featureless solid {selected_hex}, {selected_rgb_text}, with no other background content or color variation."
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
