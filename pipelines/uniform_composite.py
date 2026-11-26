"""Rough garment placement for a Qwen refinement experiment, not final output."""
import numpy as np
import cv2
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

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

    def stable_median(mask, *, darker_half=False, lit_skin=False):
        pixels = rgb[mask].astype(np.float32)
        if len(pixels) < 32:
            return None
        luma = pixels @ np.asarray((.2126, .7152, .0722), dtype=np.float32)
        if darker_half:
            pixels = pixels[luma <= np.percentile(luma, 55)]
        elif lit_skin:
            # SCHP's face label is a head region, not a skin mask: it also
            # carries eyebrows, eyelashes, pupils, nostrils, the lip line and
            # the shadowed under-jaw. A median over all of it reported this
            # boy's complexion as RGB(155,91,65) when his actual cheek is
            # RGB(208,143,107) - a deltaE of 51, dark enough that correcting a
            # portrait toward it looks heavily processed. The lit upper band
            # is overwhelmingly cheek, forehead and nose-bridge skin, and its
            # median lands within a few units of a hand-sampled cheek.
            lit = pixels[luma >= np.percentile(luma, 70)]
            if len(lit) >= 32:
                pixels = lit
        else:
            low, high = np.percentile(luma, (20, 80))
            stable = pixels[(luma >= low) & (luma <= high)]
            if len(stable) >= 32:
                pixels = stable
        value = np.median(pixels, axis=0).round().clip(0, 255).astype(np.uint8)
        return tuple(int(channel) for channel in value)

    return {
        "skin_rgb": stable_median(labels == 13, lit_skin=True),
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
    corner_distances = np.linalg.norm(corners - background[None, :], axis=1)
    bg_tol = max(10.0, min(36.0, float(np.percentile(corner_distances, 98)) + 6.0))
    # Remove only background-coloured pixels connected to the canvas border.
    # The former per-pixel colour key made every white plaid/check region
    # transparent, creating repeated X-shaped holes throughout the garment
    # reference. Internal white fabric is foreground even when it matches the
    # product-photo backdrop exactly.
    background_like = (distance <= bg_tol).astype(np.uint8)
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
    exclude_region_order="xy",
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
            if str(exclude_region_order).lower() == "yx":
                region_top = float(exclude_region[0])
            else:
                region_top = float(exclude_region[1])
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


def _fill_collar_opening_with_skin(
    canvas,
    source_rgb,
    garment_mask,
    neck_skin_mask,
    face_center_x,
    face_w,
    face_h,
    chin_y,
    backdrop,
    neck_half_width=None,
    neck_seat_y=None,
):
    """Paint measured neck skin into the target collar's empty opening.

    The uniform template reserves a neck hole for the catalogue model. When the
    uploaded person's neck is narrower than that hole, the fitted guide keeps
    the selected flat backdrop in the gap, so the rough fit shows a white wedge
    between the neck and the collar. Qwen then treats that wedge as real
    geometry and either rebuilds the neck or leaves a lit gap. Only pixels that
    are already the selected backdrop, that sit between the garment's own left
    and right edges on the same row, and that lie below the jaw are replaced.
    The collar's opening width is read from this job's own fitted garment, so a
    different uniform or a different person produces a different corridor.
    """
    rgb = np.asarray(canvas.convert("RGB"))
    height, width = garment_mask.shape
    if rgb.shape[:2] != (height, width):
        return canvas
    yy, xx = np.indices((height, width))
    samples = source_rgb[neck_skin_mask]
    if samples.size < 144:
        fallback_band = (
            (yy >= int(chin_y - face_h * .10))
            & (yy <= int(chin_y + face_h * .06))
            & (np.abs(xx - face_center_x) <= face_w * .18)
        )
        samples = source_rgb[fallback_band]
    if samples.size < 144:
        return canvas
    skin_colour = np.median(samples.reshape(-1, 3), axis=0).astype(np.float32)
    backdrop_like = np.linalg.norm(
        rgb.astype(np.float32) - backdrop[None, None, :], axis=2,
    ) <= 26.0
    max_seat_y = int(round(neck_seat_y)) if neck_seat_y is not None else int(round(chin_y + face_h * .22))
    # Cap at neck_seat_y so we fill up to (and including) the collar seat.
    # The old .18 cap was often shorter than the actual collar position, leaving
    # a backdrop-coloured wedge between jaw and collar that Qwen treated as a gap.
    max_fill_y = max_seat_y
    candidate = (
        (yy >= int(chin_y - face_h * .03))
        & (yy <= max_fill_y)
        & (np.abs(xx - face_center_x) <= face_w * .48)
        & backdrop_like
        & ~garment_mask
    )
    if not np.any(candidate):
        return canvas
    garment_rows = np.flatnonzero(np.any(garment_mask, axis=1))
    if garment_rows.size == 0:
        return canvas
    first_row = int(garment_rows[0])
    first_columns = np.flatnonzero(garment_mask[first_row])
    if first_columns.size < 2:
        return canvas
    # Above the collar's own first row there is no garment to bound the row, so
    # inherit the fitted collar's opening width centered on face_center_x.
    centre = int(round(face_center_x))
    if neck_half_width:
        corridor_half = float(np.clip(neck_half_width, face_w * .18, face_w * .32))
    else:
        corridor_half = float(face_w * .30)
    corridor_left = int(round(face_center_x - corridor_half))
    corridor_right = int(round(face_center_x + corridor_half))

    upper_left, upper_right = corridor_left, corridor_right
    for r in garment_rows[:25]:
        cols = np.flatnonzero(garment_mask[r])
        if cols.size >= 4 and cols[0] <= centre <= cols[-1]:
            upper_left = int(cols[0])
            upper_right = int(cols[-1])
            first_row = int(r)
            break
    upper_limit = int(round(chin_y - face_h * .03))
    # The added skin must follow the person's real neck silhouette instead of a
    # rectangle: a flat corridor wider than the neck produced a slab with
    # straight vertical edges, which Qwen faithfully preserved. Widen only the
    # measured skin region itself by a small margin.
    margin = max(3, int(round(face_w * .02)))
    neck_shape = cv2.dilate(
        neck_skin_mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
        ),
        iterations=1,
    ).astype(bool)
    opening = np.zeros_like(candidate)
    for row in range(upper_limit, min(height, max_fill_y + 1)):
        columns = np.flatnonzero(garment_mask[row])
        row_skin = np.flatnonzero(neck_shape[row])
        if columns.size >= 2:
            left_g = columns[columns < centre]
            right_g = columns[columns > centre]
            if left_g.size and right_g.size:
                # Inside the collar opening: reach all the way to the garment's inner
                # edges so no backdrop wedge is left between neck skin and collar.
                left = max(int(left_g[-1]), corridor_left)
                right = min(int(right_g[0]), corridor_right)
            else:
                left = corridor_left
                right = corridor_right
        elif row_skin.size >= 2:
            left = max(int(row_skin[0]), corridor_left)
            right = min(int(row_skin[-1]), corridor_right)
        elif row < first_row:
            left = min(upper_left, corridor_left)
            right = max(upper_right, corridor_right)
        else:
            left = corridor_left
            right = corridor_right
        if right <= left:
            continue
        opening[row, left:right + 1] = candidate[row, left:right + 1]
    if not np.any(opening):
        return canvas
    # Continue the neck with its own photographed pixels: each row's real skin
    # is tiled outward across the gap, so shading, pores and the under-chin
    # falloff stay continuous instead of turning into one flat colour card.
    fill = np.empty_like(rgb, dtype=np.float32)
    fill[:] = skin_colour[None, None, :]
    last_pattern = None
    for row in range(int(opening.shape[0])):
        target = np.flatnonzero(opening[row])
        if target.size == 0:
            continue
        source_columns = np.flatnonzero(neck_skin_mask[row])
        if source_columns.size >= 6:
            pattern = source_rgb[row, source_columns].astype(np.float32)
            last_pattern = pattern
        elif last_pattern is not None:
            pattern = last_pattern
        else:
            pattern = skin_colour[None, :]
        index = np.arange(target.size) % pattern.shape[0]
        fill[row, target] = pattern[index]
    fill = cv2.GaussianBlur(fill, (0, 0), .9)
    alpha = cv2.GaussianBlur(opening.astype(np.float32), (0, 0), 1.6)
    alpha = np.clip(alpha * 1.7, 0.0, 1.0)[:, :, None]
    filled = rgb.astype(np.float32) * (1.0 - alpha) + fill * alpha
    return Image.fromarray(np.rint(filled).clip(0, 255).astype(np.uint8), "RGB")


def _measure_garment_neck_opening(
    garment_mask,
    fit_top,
    face_center_x,
    face_w,
    probe_rows=8,
):
    """Measure the fitted garment's own neck opening width, in pixels.

    Every uniform is fitted separately to the current person, and the collar
    construction differs between templates (shirt with a raised band, V-neck
    sweater, blazer lapel, tunic). Nothing here may assume one template's
    geometry, so the opening is read from the garment that was just placed:
    for each of the first rows of the placement, take the contiguous garment
    run that crosses the face centre and record its width. A template whose
    centre is hollow (a V opening) yields no central run and is reported as
    unmeasured, so the caller falls back to the person's own neck width.
    """
    widths = []
    rows = garment_mask.shape[0]
    for row in range(int(fit_top), min(rows, int(fit_top) + probe_rows)):
        columns = np.flatnonzero(garment_mask[row])
        if columns.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(columns) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [columns.size - 1]))
        centre = int(round(face_center_x))
        for start, end in zip(starts, ends):
            left, right = int(columns[start]), int(columns[end])
            if left <= centre <= right:
                widths.append(float(right - left + 1))
                break
    if not widths:
        return None
    return float(np.median(widths)) * .5


def build_rough_composite(
    person, garment, labels, face_box, background, *, expect_tied_hair=False,
    minimum_neck_gap_ratio=0.0, analysed_collar: str = "",
    target_neck_gap_ratio=None, target_shoulder_width=None,
    target_shoulder_ratio=None,
):
    person = person.convert("RGB")
    if labels.shape != (person.height, person.width):
        raise ValueError("Source parsing dimensions do not match")
    # The caller supplies a source portrait whose scenery has already been
    # replaced by the selected backdrop, so every colour/anatomy measurement
    # below reads real person pixels and no paste can transfer the upload's
    # wall into the guide.
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
    # Locate visible clothing for shoulder placement: look in the upper chest
    # band below the jawline. Rows far down the torso (past ~0.42 face heights)
    # capture underarms, wide bicep sleeves and lower chest rather than the
    # true biacromial shoulder line, leading to massive, oversized boxy shoulders.
    shoulder_top = max(0, int(round(y2 + face_h * .08)))
    shoulder_bottom = min(labels.shape[0], int(round(y2 + face_h * .42)))
    shoulder_band = (clothing | arms)[shoulder_top:shoulder_bottom]
    canvas_width = labels.shape[1]

    def _row_spans(rows):
        """Row spans that end inside the canvas, plus the clipped-row count.

        A row whose clothing reaches either image edge is a lower bound on the
        body, not a measurement of it: the torso continues outside the frame.
        Counting those rows as real widths pins the measurement at the canvas
        width, which then scales the template edge to edge and erases the
        shoulder slope entirely.
        """
        spans, clipped, clipped_floor = [], 0, 0.0
        for row in rows:
            columns = np.flatnonzero(row)
            if columns.size < max(8, int(face_w * .20)):
                continue
            width = float(columns[-1] - columns[0] + 1)
            if columns[0] <= 1 or columns[-1] >= canvas_width - 2:
                # The torso continues outside the frame, so this row is a lower
                # bound on the body, not a measurement of it. It must not be
                # averaged in as if it were the real width - but discarding it
                # outright is just as wrong: on an upload cropped at the
                # shoulders these clipped rows are the widest real shoulders,
                # and dropping them collapsed the fitted shirt to the narrow
                # neck/upper-chest rows (measured 838px against a true 943px
                # body, a shirt ~180px too narrow at the shoulder line).
                clipped += 1
                clipped_floor = max(clipped_floor, width)
                continue
            spans.append(width)
        return spans, clipped, clipped_floor

    shoulder_row_widths, clipped_rows, clipped_width_floor = _row_spans(shoulder_band)
    if not shoulder_row_widths:
        shoulder_row_widths, clipped_rows, clipped_width_floor = _row_spans(
            clothing | arms,
        )
    if not shoulder_row_widths:
        # Every candidate row runs off the frame. The upload is cropped tighter
        # than the body, so fall back to the widest in-frame evidence there is
        # rather than declaring the canvas itself to be the person's width.
        fallback = []
        for row in shoulder_band:
            columns = np.flatnonzero(row)
            if columns.size >= max(8, int(face_w * .20)):
                fallback.append(float(columns.size))
        if not fallback:
            raise ValueError("Cannot locate source clothing for placement")
        shoulder_row_widths = fallback
    # Measure complete row spans rather than pooling all column coordinates.
    # Pooling biases the result toward the narrow neck/upper rows and made the
    # fitted torso too slim, so a correctly sized head appeared oversized.
    # The 85th percentile follows the uploaded upper chest without allowing a
    # single extended sleeve or hand to define the body width.
    if target_shoulder_width is not None and float(target_shoulder_width) > 0:
        measured_width = float(np.clip(
            float(target_shoulder_width),
            face_w * 1.45,
            min(float(canvas_width - 2), face_w * 2.15),
        ))
        logger.info(
            "[UNIFORM_FIT] Using direct target shoulder width %.1fpx (%.2fx face width)",
            measured_width, measured_width / max(face_w, 1.0),
        )
    elif target_shoulder_ratio is not None and float(target_shoulder_ratio) > 0:
        measured_width = float(np.clip(
            float(target_shoulder_ratio) * face_w,
            face_w * 1.45,
            min(float(canvas_width - 2), face_w * 2.15),
        ))
        logger.info(
            "[UNIFORM_FIT] Using direct target shoulder ratio %.2f -> %.1fpx",
            float(target_shoulder_ratio), measured_width,
        )
    else:
        measured_width = float(np.percentile(shoulder_row_widths, 80))
        if clipped_width_floor >= canvas_width - 2:
            clipped_width_floor = 0.0
        if clipped_width_floor > measured_width:
            measured_width = min(clipped_width_floor, float(canvas_width - 2), face_w * 2.15)
            logger.info(
                "[UNIFORM_FIT] %d edge-clipped row(s) raise the measured body width "
                "from %.0fpx to at least %.0fpx",
                clipped_rows, float(np.percentile(shoulder_row_widths, 80)), measured_width,
            )
        elif clipped_rows:
            logger.info(
                "[UNIFORM_FIT] Ignored %d edge-clipped row(s) narrower than the "
                "measured body width", clipped_rows,
            )
        source_body_to_face = measured_width / max(face_w, 1.0)
        if source_body_to_face < 1.35:
            in_frame_spans = []
            for row in shoulder_band:
                columns = np.flatnonzero(row)
                if columns.size >= max(8, int(face_w * .20)):
                    in_frame_spans.append(float(columns[-1] - columns[0] + 1))
            if not in_frame_spans:
                for row in (clothing | arms):
                    columns = np.flatnonzero(row)
                    if columns.size >= max(8, int(face_w * .20)):
                        in_frame_spans.append(float(columns[-1] - columns[0] + 1))
            if in_frame_spans:
                in_frame_p85 = float(np.percentile(in_frame_spans, 80))
                measured_width = float(np.clip(
                    max(measured_width, in_frame_p85),
                    min(float(canvas_width), face_w * 1.5),
                    min(float(canvas_width), face_w * 2.15),
                ))
            else:
                measured_width = min(float(canvas_width), face_w * 1.85)
            source_body_to_face = measured_width / max(face_w, 1.0)

        measured_width = float(np.clip(
            measured_width,
            face_w * 1.45,
            min(float(canvas_width - 2), face_w * 2.15),
        ))
    source_body_to_face = measured_width / max(face_w, 1.0)
    original_size = person.size
    neck_half = max(2, int(round(face_w * .16)))
    central_left = max(0, int(round(face_center_x - neck_half)))
    central_right = min(labels.shape[1], int(round(face_center_x + neck_half)))
    central_clothing = clothing[:, central_left:central_right + 1]
    central_rows = np.flatnonzero(np.any(central_clothing, axis=1))
    central_rows = central_rows[central_rows >= int(y2 - face_h * .08)]
    # Check clothing across the collar span (within face_w * .38 of center).
    # When a source shirt has an open collar or V-neck, central_rows only detects the
    # deep tie knot or chest, while the actual collar wings start much higher at the sides.
    collar_left = max(0, int(round(face_center_x - face_w * .38)))
    collar_right = min(labels.shape[1], int(round(face_center_x + face_w * .38)))
    collar_clothing = clothing[:, collar_left:collar_right + 1]
    collar_rows = np.flatnonzero(np.any(collar_clothing, axis=1))
    collar_rows = collar_rows[collar_rows >= int(y2 - face_h * .08)]

    # The central throat corridor determines where clothing opens on the neck.
    # Outer collar wingtips (at face_w * .38) frequently rise near the jaw/ears,
    # but the neck opening sits centrally between those points.
    if central_rows.size:
        source_collar_y = float(central_rows[0])
    elif collar_rows.size:
        source_collar_y = float(collar_rows[0])
    else:
        source_collar_y = y2 + face_h * .18

    # Scan the central neck corridor for real skin down to where central clothing begins,
    # ensuring at least a natural anatomical search range (up to 0.32 face heights below the chin).
    # The floor here was .22 while this comment said .32, so an open-collar
    # upload whose neck really runs to .31 face-heights could never measure
    # past ~.23: job 5a1bbeaa reported .204 for a neck measured at .320 and
    # the prompt then told Qwen a gap a third shorter than image 1 shows.
    placement_hsv = cv2.cvtColor(np.asarray(person), cv2.COLOR_RGB2HSV)
    visible_neck_limit = min(
        labels.shape[0],
        int(round(max(source_collar_y, y2 + face_h * .32))) + 1,
    )
    visible_neck_rows = []
    for row in range(int(y2), visible_neck_limit):
        band = placement_hsv[row, central_left:central_right]
        skin_fraction = np.count_nonzero(
            (band[:, 1] >= 20)
            & (band[:, 2] >= 35)
            & (band[:, 2] <= 245)
            & ((band[:, 0] <= 25) | (band[:, 0] >= 170))
        ) / max(1, band.shape[0])
        if skin_fraction >= .45:
            visible_neck_rows.append(row)
    measured_neck_bottom = (
        float(visible_neck_rows[-1]) if visible_neck_rows
        else min(source_collar_y, y2 + face_h * .15)
    )
    # Seat the collar at the measured neck base. The anatomical minimum floor
    # is 0.06 face-heights so the collar never lands directly on the jaw.
    requested_gap = float(np.clip(minimum_neck_gap_ratio, 0.0, .26))
    # The .01 face-height bias seats the collar just below the last skin row
    # rather than on it. The detected clothing is then a hard ceiling: the
    # scan range is an upper bound on anatomy, not a licence to run into the
    # garment. Where source clothing is near skin colour the corridor kept
    # accumulating rows past the collar, so widening the range would have
    # invented neck instead of revealing it.
    # Whether the skin scan overran the detected clothing decides which
    # measurement to trust. If it did, the corridor was reading garment pixels
    # as skin (source clothing close to skin colour), so the collar row is the
    # only reliable neck base and the bias would push the seat into fabric.
    # If it stopped at or above the collar the skin reading is real, and the
    # .01 face-height bias seats the collar just below the last skin row
    # instead of on it.
    if measured_neck_bottom > source_collar_y:
        neck_base = float(source_collar_y)
    else:
        neck_base = measured_neck_bottom + face_h * .01
    measured_gap = (neck_base - y2) / max(face_h, 1.0)
    # The cap must stay above the anatomical range the scan can now reach, or
    # it silently re-imposes the old ceiling: with the cap at .20 an upload
    # measured at .32 was clipped straight back to .20 (job 5a1bbeaa), and the
    # NECK/COLLAR LOCK prompt then under-reported image 1's own geometry.
    # .32 matches the scan limit; a genuinely longer neck is implausible and
    # still clamps.
    neck_cap = max(0.32, requested_gap)
    # The caller's request is a floor, not a target, and taking the larger of
    # the two let it overrule the anatomy: job 1461a896 measured this boy's
    # neck base at 0.166 face-heights and seated the collar at 0.23, a 38%
    # over-long neck that reads as a pasted-on head. Honour the measurement
    # whenever it is anatomically plausible and let the request raise it only
    # when the measurement is implausibly short (a jaw shadow or a high
    # polo/turtleneck can truncate the visible skin column).
    # When the caller has measured exposed neck directly from the untouched
    # upload, that per-person value is the geometry authority.  Rough fitting
    # previously ran first and seated the garment using its less reliable
    # clothing/skin scan; replacing only the metadata afterwards changed the
    # prompt but left the visual collar cue at the wrong height.  Honour the
    # direct dynamic measurement here so cue, prompt and delivery QA agree.
    try:
        direct_target_gap = float(target_neck_gap_ratio)
    except (TypeError, ValueError):
        direct_target_gap = float("nan")
    if np.isfinite(direct_target_gap) and .04 <= direct_target_gap <= .42:
        neck_seat_ratio = float(np.clip(direct_target_gap, .04, .42))
    elif measured_gap >= .12:
        neck_seat_ratio = float(np.clip(measured_gap, .06, neck_cap))
    else:
        neck_seat_ratio = float(np.clip(
            max(measured_gap, requested_gap), .06, neck_cap,
        ))
    neck_seat_y = int(round(y2 + neck_seat_ratio * face_h))
    # Reporting keeps the historical clothing-anchored measurement so existing
    # diagnostics and fit metadata stay comparable.
    source_neck_gap_ratio = float(np.clip((source_collar_y - y2) / max(face_h, 1.0), .06, .32))
    effective_neck_gap_ratio = neck_seat_ratio
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
    base[labels == 0] = background
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

    # In a passport portrait, the torso continues to the bottom edge of the frame.
    # When a source portrait is closely cropped and ends above the chest/sternum,
    # the uniform must not stop mid-air leaving empty background below; extend
    # source_fit downward to the canvas bottom across the measured torso width.
    canvas_bottom = labels.shape[0] - 1
    if fit_rows[-1] < canvas_bottom:
        lower_rows = fit_rows[fit_rows >= fit_rows[-1] - max(5, int(face_h * 0.05))]
        lower_cols = np.flatnonzero(np.any(source_fit[lower_rows], axis=0))
        if lower_cols.size >= 2:
            ext_left = max(fit_left_limit, int(lower_cols[0]))
            ext_right = min(fit_right_limit, int(lower_cols[-1] + 1))
        else:
            ext_left = fit_left_limit
            ext_right = fit_right_limit
        # The extension must be continuous with the photographed torso. A
        # closely cropped upload ends partway down the canvas and everything
        # below is padding, so the parsed silhouette stops there even though
        # this fill continues to the frame edge. Any row between the last
        # parsed row and the fill start is left with no subject at all, and a
        # fully blank row inside the garment is what Qwen renders as a straight
        # tear across the shirt - measured on job 9edf1076 as 2 rows at 100%
        # backdrop (y=877-878), which split the uniform into two pieces.
        #
        # Fill from the last row that actually has subject, with no gap. The
        # region below the photograph is then a continuous garment block that
        # the model extends downward, rather than two blocks with a hole.
        subject_rows = np.flatnonzero(np.any(source_fit, axis=1))
        fill_from = int(subject_rows[-1]) if subject_rows.size else int(fit_rows[-1])
        source_fit[fill_from:canvas_bottom + 1, ext_left:ext_right] = True
        # Close any one-off blank rows the parse left inside the torso, for the
        # same reason: a hole in the middle of cloth becomes a visible tear.
        torso_band = source_fit[fit_rows[0]:canvas_bottom + 1]
        if torso_band.size:
            row_has_subject = np.any(torso_band, axis=1)
            blank = np.flatnonzero(~row_has_subject)
            for blank_row in blank:
                absolute = int(fit_rows[0] + blank_row)
                above = source_fit[:absolute][::-1]
                below = source_fit[absolute + 1:]
                prev_rows = np.flatnonzero(np.any(above, axis=1))
                next_rows = np.flatnonzero(np.any(below, axis=1))
                if not prev_rows.size or not next_rows.size:
                    continue
                prev_row = absolute - 1 - int(prev_rows[0])
                next_row = absolute + 1 + int(next_rows[0])
                # Bridge with the narrower of its two neighbours so the fill
                # can never be wider than the real garment around it.
                bridge = source_fit[prev_row] & source_fit[next_row]
                if bridge.any():
                    source_fit[absolute] = bridge
        fit_rows = np.flatnonzero(np.any(source_fit, axis=1))

    fit_top, fit_bottom = int(fit_rows[0]), int(fit_rows[-1])
    # The rear collar band belongs at the measured neck base (neck_seat_y).
    # Stray shoulder/edge pixels in source_fit must not pull the collar into
    # the throat or jaw. The rear collar stand rises at most ~0.05 face-heights
    # above the front collar seat.
    collar_stand = int(round(face_h * .04))
    min_collar_top = int(round(y2 + face_h * .08))
    target_collar_top = max(min_collar_top, neck_seat_y - collar_stand)
    fit_top = target_collar_top
    collar_corridor_w = int(round(desired_fit_width * 0.5))
    c_left = max(0, int(round(face_center_x - collar_corridor_w)))
    c_right = min(labels.shape[1], int(round(face_center_x + collar_corridor_w)))
    source_fit[fit_top:canvas_bottom + 1, c_left:c_right] = True
    fit_columns = np.flatnonzero(np.any(source_fit, axis=0))
    fit_left = int(fit_columns[0]) if fit_columns.size else max(0, int(round(face_center_x - desired_fit_width * .5)))
    fit_right = int(fit_columns[-1]) if fit_columns.size else min(labels.shape[1] - 1, fit_left + desired_fit_width - 1)
    fit_width = fit_right - fit_left + 1
    fit_height = fit_bottom - fit_top + 1
    template_row_widths = np.count_nonzero(template_alpha, axis=1)
    garment_rows_present = np.flatnonzero(template_row_widths > 0)
    provisional_scale = fit_width / max(1, template.shape[1])
    if garment_rows_present.size >= 2:
        garment_top = int(garment_rows_present[0])
        # Trim transparent top margin so the collar starts at fit_top rather than drifting downward
        if garment_top > 0:
            template = template[garment_top:]
            template_alpha = template_alpha[garment_top:]
            template_row_widths = template_row_widths[garment_top:]
            garment_top = 0
        # Measure widest template row within the visible height of the garment
        visible_template_rows = max(
            1, min(template.shape[0] - garment_top, int(round(fit_height / max(provisional_scale, 1e-6))))
        )
        widest_visible = float(template_row_widths[garment_top:garment_top + visible_template_rows].max())
        if widest_visible >= 8.0:
            scale = fit_width / widest_visible
        else:
            scale = provisional_scale
        # Cap scale so fitted template stays naturally tailored and never blows up collar/shoulders
        scale = min(scale, (fit_width * 1.08) / max(1, template.shape[1]))
    else:
        scale = provisional_scale

    # Measure the front collar notch (throat seat) in the template dynamically:
    # Completely color-agnostic; works dynamically for any garment template and collar style.
    cx_t = template.shape[1] // 2
    half_w_t = max(4, int(round(template.shape[1] * 0.06)))
    notch_row = None

    # Strategy 1: Center aperture transparency (for templates with cutout transparent neck hole)
    for r in range(int(template.shape[0] * 0.06), int(template.shape[0] * 0.45)):
        if template_alpha[r, cx_t] < 32 and r + 1 < template.shape[0] and template_alpha[r + 1, cx_t] >= 128:
            notch_row = r + 1
            break

    # Strategy 2: Collar style anatomical ratio
    if notch_row is None:
        collar_style = (analysed_collar or "").lower()
        if any(w in collar_style for w in ("mandarin", "band", "stand", "choker")):
            notch_ratio = 0.07
        elif any(w in collar_style for w in ("crew", "round", "t-shirt")):
            notch_ratio = 0.11
        elif any(w in collar_style for w in ("pointed", "folded", "lapel", "polo", "shirt", "blazer")):
            notch_ratio = 0.22
        else:
            notch_ratio = 0.18
        notch_row = int(round(template.shape[0] * notch_ratio))

    collar_notch_drop = int(round(notch_row * scale))
    # Align the template so its front collar notch lands exactly at neck_seat_y (the anatomical throat base)
    aligned_fit_top = int(round(neck_seat_y - collar_notch_drop))
    # Ensure aligned_fit_top does not rise above eye level or drop below the collar seat
    fit_top = max(int(round(y1)), min(int(round(neck_seat_y - face_h * 0.06)), aligned_fit_top))
    fit_height = fit_bottom - fit_top + 1
    source_fit[:fit_top, :] = False
    source_fit[fit_top:canvas_bottom + 1, c_left:c_right] = True
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

    # ``scale`` may now exceed the plain width fit so the shoulder line lands
    # inside the frame; the garment is then wider than the measured body and
    # the surplus is sleeve that genuinely continues past the crop. Keep the
    # real scaled width here and trim it symmetrically below, instead of
    # squeezing the whole shirt back into fit_width and losing the shoulders.
    resized_width = max(1, int(round(template.shape[1] * scale)))
    resized = resize_premultiplied(
        resized_width, resized_height, cv2.INTER_AREA,
    )
    if resized_height < fit_height:
        # Resize the original premultiplied template directly to avoid a
        # second alpha/color interpolation pass.
        resized = resize_premultiplied(
            resized_width, fit_height, cv2.INTER_LINEAR,
        )
    else:
        # Passport framing needs only the collar, shoulders and upper chest.
        # Discard the template's lower torso rather than compressing it upward.
        resized = resized[:fit_height]
    if resized_width > fit_width:
        # Centre the garment on the measured body and let equal amounts of
        # sleeve fall outside each edge, as a real upper-chest crop would.
        overhang = (resized_width - fit_width) // 2
        resized = resized[:, overhang:overhang + fit_width]
    elif resized_width < fit_width:
        padded = np.zeros(
            (resized.shape[0], fit_width, 4), dtype=resized.dtype,
        )
        offset = (fit_width - resized_width) // 2
        padded[:, offset:offset + resized_width] = resized
        resized = padded

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
    # Restrict label 13 (face/neck) to stop at the chin (y2) so the face is protected,
    # but neck skin below the chin is not pasted over the collar wings.
    yy, xx = np.indices(labels.shape)
    protect_subject = np.isin(labels, (1, 4)) | ((labels == 13) & (yy <= int(y2)))
    # SCHP commonly classifies the exposed neck as source clothing. Recover
    # only skin-like pixels in the anatomical bridge under the detected face,
    # constrained to the central collar opening so that front collar points,
    # lapels and placket are never covered by skin.
    rgb = np.asarray(person)
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
    face_kernel_w = max(5, int(round(face_w * .35)) | 1)
    face_kernel_h = max(5, int(round(face_h * .25)) | 1)
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
        face_box_bool = (
            (xx >= int(x1 - face_w * .04))
            & (xx <= int(x2 + face_w * .04))
            & (yy >= int(y1))
            & (yy <= int(y2))
        )
        recovered_face = (
            (face_oval_bool | face_box_bool)
            & near_parsed_face
            & (face_colour_distance < 65.0)
            & (yy <= int(y2))
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
        tied_hair_value = 68 if expect_tied_hair else 55
        lower_hair = lower_hair_band & side_corridors & (
            (hair_distance < tied_hair_distance)
            & ((hsv[:, :, 2] < tied_hair_value) | (hair_distance < 32.0))
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
            & ((hsv[:, :, 2] < tied_hair_value) | (hair_distance < 32.0))
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
            (left_tail_pixels >= minimum_tail_area and right_tail_pixels >= minimum_tail_area)
            or (left_root_pixels >= minimum_tail_area and right_root_pixels >= minimum_tail_area)
        )
        if bilateral_tail:
            detected_ponytail_side = "bilateral"
        elif right_tail_pixels >= minimum_tail_area and left_tail_pixels < minimum_tail_area:
            detected_ponytail_side = "viewer-right"
        elif left_tail_pixels >= minimum_tail_area and right_tail_pixels < minimum_tail_area:
            detected_ponytail_side = "viewer-left"
        elif right_tail_pixels >= minimum_tail_area and right_tail_pixels > left_tail_pixels * 3.5:
            detected_ponytail_side = "viewer-right"
        elif left_tail_pixels >= minimum_tail_area and left_tail_pixels > right_tail_pixels * 3.5:
            detected_ponytail_side = "viewer-left"
        elif right_tail_pixels >= minimum_tail_area or left_tail_pixels >= minimum_tail_area:
            detected_ponytail_side = "bilateral"
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
                & ((hsv[:, :, 2] < tied_hair_value) | (hair_distance < 32.0))
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
    collar_side_y = max(float(neck_seat_y), float(fit_top))
    # This colour scan is the only thing that recovers the neck, because the
    # LIP-20 parser has no neck class at all: label 13 (Face) stops at the jaw
    # and the skin below it comes back as label 0 (Background), which this
    # routine paints with the flat backdrop. The scan must therefore reach the
    # collar the garment was actually seated at, not a fixed .24 face-heights.
    # Measured on a cut-out upload whose collar sat at .31: the scan stopped 33
    # rows short and left a white wedge from jaw to collar, so the conditioning
    # showed Qwen a floating head and it rebuilt the neck and neckline.
    scan_bottom = max(
        float(collar_side_y) + face_h * .03,
        y2 + face_h * .03,
    )
    neck_colour_candidate = (
        # The corridor must clear the collar opening it has to fill. Widen to
        # 0.42 of face width to match natural jaw width and prevent white side gaps.
        (np.abs(xx - ((x1 + x2) * .5)) <= face_w * .42)
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
    # Use the upload's own neck skin down to where that skin really ends. The
    # collar seat only guarantees a short bridge; capping the neck at a couple
    # of pixels below it made the region under the jaw a flat synthetic patch
    # with straight vertical edges. Real photographed skin keeps the neck's own
    # shading and width, and the collar still covers its lower edge.
    # Extend the neck skin restoration down to where the collar actually sits.
    # The old cap of collar_side_y + face_h * .03 was only a few pixels below
    # the collar seat, leaving a bare backdrop strip between jaw and collar —
    # the visible neck hole/gap. Now we allow skin to reach all the way to the
    # collar seat (collar_side_y), bounded only by scan_bottom and detected skin.
    neck_bottom = min(
        scan_bottom,
        max(collar_side_y, min(detected_skin_bottom, collar_side_y + face_h * .03)),
    )
    neck_t = np.clip((yy - neck_top) / max(1.0, neck_bottom - neck_top), 0.0, 1.0)
    # Follow the jaw near the top, then taper inward through the collar opening.
    # CODEX NOTE: Using .28-.12 pinched the neck to 16% face width at the bottom,
    # leaving blue side gutters that bleed into the collar during blurring.
    neck_half_width = face_w * (.34 - .06 * neck_t)
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
    # Ensure collar leaves of the uniform fold in front of the neck below the chin
    garment_canvas_mask = np.zeros(labels.shape, dtype=bool)
    garment_canvas_mask[fit_top:fit_top + resized.shape[0], fit_left:fit_right + 1] = active
    # Neck width is measured twice, on this upload and on the garment that was
    # just fitted to it, because both the person and the uniform change per job.
    measured_neck_half = None
    neck_rows = np.flatnonzero(
        np.any(skin_neck[int(y2):int(neck_bottom) + 1], axis=0)
    )
    if neck_rows.size >= 8:
        measured_neck_half = (float(neck_rows[-1] - neck_rows[0]) + 1.0) * .5
    opening_half = _measure_garment_neck_opening(
        garment_canvas_mask, fit_top, (x1 + x2) * .5, face_w,
    )
    person_neck_half = measured_neck_half or face_w * .32
    neck_core_half = float(np.clip(person_neck_half, face_w * .18, face_w * .34))
    if opening_half is not None:
        # The collar must never be painted over the neck. Its own opening is
        # measured on the fitted garment this job just placed, so the protected
        # column tracks whatever collar construction that uniform has. The
        # measured opening may exceed the generic neck bound: leaving its last
        # few pixels unfilled left white specks along the collar edge, which
        # Qwen preserved as gaps between neck and collar.
        neck_core_half = float(np.clip(
            max(neck_core_half, min(opening_half, face_w * .34)),
            face_w * .18, face_w * .34,
        ))
    # The upload's own neck skin stays in front of the garment. A shirt's rear
    # collar band stands up BEHIND the throat, so the photographed neck must
    # cover it: clearing neck alpha under the fabric instead exposed the
    # product photo's unworn rear band and Qwen regenerated it as a collar ring
    # or a navy neck band around the throat. The neck alpha is already gated to
    # measured skin colour and cleared on backdrop pixels, so restoring it here
    # cannot carry source clothing into the collar opening.
    protect = np.maximum(subject_alpha, neck_alpha)
    canvas = Image.composite(person, canvas, Image.fromarray(protect))
    # The template's collar leaves surround an opening sized for the catalogue
    # model's neck. On this person's narrower neck the leftover ring between
    # neck and collar stayed at the selected flat backdrop, which read as white
    # wedges beside the neck and made Qwen rebuild the neckline. Fill only
    # backdrop-coloured pixels that sit inside the garment's own row span,
    # below the jaw, with this person's measured neck skin.
    canvas = _fill_collar_opening_with_skin(
        canvas,
        rgb,
        garment_canvas_mask,
        skin_neck,
        float((x1 + x2) * .5),
        float(face_w),
        float(face_h),
        float(y2),
        np.asarray(background, dtype=np.float32),
        neck_half_width=neck_core_half,
        neck_seat_y=float(neck_seat_y),
    )
    # Measure hair extent on a wider, measurement-only mask.
    #
    # measured_hair_mask is a *protection* mask: the tied-hair band above is
    # deliberately capped at y2 + face_h * .24 because below that, dark source
    # outerwear merges with a tail and copying it drags a vest into the guide.
    # Reporting inherited that cap, so a braid could never measure past ~0.24.
    # Job 30399633's braids run to at least 0.73 face-heights below the chin
    # and were reported as 0.212, "ends around the neck-to-shoulder line";
    # Qwen then rendered a jaw-length bob and the real plaits were lost.
    #
    # Extending the protection mask is not safe, but measuring further down is:
    # these pixels are only counted, never copied. Require hair colour inside
    # the side corridors and connectivity to the parsed crown, so a dark collar
    # cannot be mistaken for a plait.
    extent_hair_mask = measured_hair_mask
    try:
        if np.count_nonzero(parsed_hair) >= 128:
            extent_band = (
                (yy >= int(y2 - face_h * .03))
                & (yy <= int(y2 + face_h * 1.60))
                & (np.abs(xx - face_center_x) <= face_w * 1.15)
            )
            extent_candidates = extent_band & (
                (hair_distance < 58.0)
                & ((hsv[:, :, 2] < 118) | (hair_distance < 30.0))
            )
            extent_candidates |= measured_hair_mask
            count, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
                extent_candidates.astype(np.uint8), connectivity=8,
            )
            minimum_extent_area = max(24, int(round(face_w * face_h * .0012)))
            # Require connection to the crown, but allow the braid to be
            # *occluded* on the way down. On job 30399633's passport crop SCHP
            # labels hair only to 0.24 face-heights ABOVE the chin, while the
            # plaits resume ~100 rows lower behind the ears and jaw; a strict
            # touch test linked zero components and both plaits (20k and 15k
            # pixels, reaching 0.71 face-heights) were discarded. Dilate the
            # parsed crown for the connectivity test only - no pixel is drawn
            # or copied, the same approach the tied-hair path already takes.
            crown_reach = cv2.dilate(
                parsed_hair.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(5, int(round(face_w * .40)) | 1),
                     max(5, int(round(face_h * .95)) | 1)),
                ),
                iterations=1,
            ).astype(bool)
            linked = np.zeros_like(extent_candidates)
            for component in range(1, count):
                if comp_stats[component, cv2.CC_STAT_AREA] < minimum_extent_area:
                    continue
                component_mask = comp_labels == component
                if np.count_nonzero(component_mask & crown_reach):
                    linked |= component_mask
            if np.count_nonzero(linked) >= np.count_nonzero(measured_hair_mask):
                extent_hair_mask = linked
    except Exception:
        # Measurement must never break the composite; fall back to the
        # protection mask's (conservative) extent.
        extent_hair_mask = measured_hair_mask
    hair_y, hair_x = np.where(extent_hair_mask)
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
    # How much torso the upload actually contains, measured from the chin in
    # face-heights. A 35:45 passport frame needs roughly 1.15 face-heights of
    # chest below the chin; an upload that stops at 0.66 (a hoodie photographed
    # at the jawline, job 58682ecb) is asking the model to invent 43% of the
    # body from no shoulder evidence at all, which is what produces a fitted
    # garment on a torso that does not match the person. Report it so the
    # caller can warn rather than silently deliver an invented body.
    source_chin_to_frame = (original_size[1] - y2) / max(face_h, 1.0)
    torso_evidence_sufficient = bool(source_chin_to_frame >= .90)

    return canvas, {"garment_box": [fit_left, fit_top,
                                     fit_right + 1, int(fit_bottom + 1)],
                    "source_chin_to_frame_ratio": round(float(source_chin_to_frame), 3),
                    "torso_evidence_sufficient": torso_evidence_sufficient,
                    "source_canvas_size": list(original_size),
                    "expanded_canvas_size": list(person.size),
                    "source_body_width_px": round(measured_width, 2),
                    "source_face_width_px": round(face_w, 2),
                    "source_body_to_face_width": round(source_body_to_face, 3),
                    "garment_to_face_width": round(measured_width / max(face_w, 1.0), 3),
                    "source_neck_gap_ratio": round(effective_neck_gap_ratio, 3),
                    "measured_source_neck_gap_ratio": round(source_neck_gap_ratio, 3),
                    # Per-job neck geometry: the collar seat, the neck column
                    # width and the garment opening it was fitted into all come
                    # from measurements on this upload and this uniform.
                    "neck_seat_ratio": round(neck_seat_ratio, 3),
                    "neck_column_half_px": round(float(neck_core_half), 1),
                    "garment_neck_opening_half_px": (
                        round(float(opening_half), 1)
                        if opening_half is not None else None
                    ),
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

    # An unworn rear collar exists ONLY inside an open central neckline opening
    # bounded by outer garment shoulders/leaves to the left and right of the saturated band.
    # If the candidate band spans the outer edges of the garment (no outer structure to its left or right),
    # it is the garment's own outer collar and must NEVER be removed.
    min_outer_pixels = max(4, int(round(width * .03)))
    has_outer_left = True
    has_outer_right = True
    for r in selected:
        sat_cols = np.flatnonzero(saturated[r])
        if sat_cols.size:
            left_outer = np.count_nonzero(alpha[r, :sat_cols[0]] > 32)
            right_outer = np.count_nonzero(alpha[r, sat_cols[-1] + 1:] > 32)
            if left_outer < min_outer_pixels:
                has_outer_left = False
            if right_outer < min_outer_pixels:
                has_outer_right = False
    if not (has_outer_left and has_outer_right):
        return Image.fromarray(rgba, "RGBA"), 0

    remove = np.zeros_like(alpha, dtype=bool)
    band_left, band_right = None, None
    for row in selected:
        columns = np.flatnonzero(saturated[row])
        if columns.size:
            start, end = int(columns[0]), int(columns[-1])
            band_left = start if band_left is None else min(band_left, start)
            band_right = end if band_right is None else max(band_right, end)
    if band_left is not None:
        band_bottom = int(selected[-1]) + 1
        band_top = int(selected[0])
        strip = np.zeros_like(alpha, dtype=bool)
        # Clear strictly within the central neck corridor between band_left and band_right
        strip[max(0, band_top):band_bottom + 1, max(0, band_left):min(alpha.shape[1], band_right + 1)] = True
        remove |= strip & (alpha > 0)
    remove = cv2.dilate(
        remove.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    ).astype(bool)
    if band_left is not None:
        remove[:, :max(0, band_left)] = False
        remove[:, min(alpha.shape[1], band_right + 1):] = False
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
