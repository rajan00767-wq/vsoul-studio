"""Conservative removal of template-matched vivid badges on dark fabric."""
import cv2
import numpy as np
from PIL import Image


def _patches(
    rgb,
    hue=None,
    allow_bottom_edge=False,
    max_aspect=1.5,
    rectangular=False,
):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    selected = (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 125)
    selected[:int(h * .45)] = False
    if hue is not None:
        delta = np.abs(hsv[:, :, 0].astype(float) - hue)
        selected &= np.minimum(delta, 180 - delta) < 10
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
    patches = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not (
            .0008 < area / (h * w) < .025
            and .35 < bw / max(bh, 1) < float(max_aspect)
        ):
            continue
        if x < 5 or y < 5 or x + bw >= w - 5 or (not allow_bottom_edge and y + bh >= h - 5):
            continue
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
        # A real round badge with pale lettering occupies roughly 40-48% of
        # its bounding rectangle after the lettering splits the saturated
        # field.  The old 50% floor therefore rejected exactly those badges.
        # Compactness, chest position and size are still enforced below.
        if mask.sum() / (bw * bh) < .35:
            continue
        pad = max(3, round(min(bw, bh) * .12))
        if rectangular:
            # Include antialiased/recoloured badge borders that fall below the
            # saturated-core threshold. These are the thin blue lines left
            # behind after Qwen turns a round crest into a cropped rectangle.
            pad = max(pad, 8, round(min(bw, bh) * .35))
            expanded = np.zeros_like(mask, dtype=np.uint8)
            cv2.rectangle(
                expanded,
                (max(0, x - pad), max(0, y - pad)),
                (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)),
                1,
                -1,
            )
        else:
            expanded = cv2.dilate(
                mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8),
            )
        # Badges can sit on white/red plaid rather than dark fabric. Requiring
        # a dark surrounding ring rejected the real compact badge; size,
        # aspect, saturation and location already exclude collars and buttons.
        patches.append((expanded, float(np.median(hsv[:, :, 0][labels == i]))))
    return patches


def _brown_patches(rgb, hue=None, allow_bottom_edge=False):
    """Find compact dark-red/brown badge panels split by pale logo pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    selected = (
        ((hsv[:, :, 0] <= 22) | (hsv[:, :, 0] >= 172))
        & (hsv[:, :, 1] >= 50)
        & (hsv[:, :, 2] >= 40)
        & (hsv[:, :, 2] <= 200)
    )
    selected[:int(h * .40)] = False
    if hue is not None:
        delta = np.abs(hsv[:, :, 0].astype(float) - hue)
        selected &= np.minimum(delta, 180 - delta) < 14
    # Remove narrow plaid stripes before joining the logo-split badge panel.
    # Without the opening, a red check line connects the badge to the entire
    # shirt and prevents it from being recognized as a compact rectangle.
    raw_selected = selected.astype(np.uint8)
    selected = cv2.morphologyEx(
        raw_selected, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),
    )
    selected = cv2.morphologyEx(
        selected, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected, 8)
    patches = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        area_fraction = area / float(h * w)
        if not (.0007 < area_fraction < .03 and .55 < bw / max(bh, 1) < 1.8):
            continue
        if x < 5 or y < 5 or x + bw >= w - 5 or (not allow_bottom_edge and y + bh >= h - 5):
            continue
        component = labels == i
        # A badge panel is materially darker than the bright red/brown trim
        # and plaid intersections that can form similar compact components.
        # This separates the real dark logo panel from sleeve/collar fabric
        # without hard-coding a garment colour or image position.
        if float(np.median(hsv[:, :, 2][component])) > 145.0:
            continue
        pad = max(3, round(min(bw, bh) * .12))
        # Replace the complete rectangular badge panel, including pale logo
        # marks that split the dark component. A dilated component retained a
        # jagged plaid-shaped boundary and produced block seams in the guide.
        expanded = np.zeros_like(selected, dtype=np.uint8)
        cv2.rectangle(
            expanded,
            (max(0, x - pad), max(0, y - pad)),
            (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)),
            1, -1,
        )
        patches.append((expanded, float(np.median(hsv[:, :, 0][component]))))
    # A generated badge may be cropped by the lower passport edge. Retain
    # compact lower-chest brown panels before morphology, which otherwise can
    # merge them into a nearby red plaid stripe.
    raw_count, raw_labels, raw_stats, _ = cv2.connectedComponentsWithStats(raw_selected, 8)
    for i in range(1, raw_count):
        x, y, bw, bh, area = raw_stats[i]
        area_fraction = area / float(h * w)
        if not (
            y >= int(h * .38)
            and .0015 < area_fraction < .03
            and .50 < bw / max(bh, 1) < 2.2
        ):
            continue
        component = raw_labels == i
        if float(np.median(hsv[:, :, 2][component])) > 145.0:
            continue
        pad = max(3, round(min(bw, bh) * .12))
        expanded = np.zeros_like(raw_selected, dtype=np.uint8)
        cv2.rectangle(
            expanded,
            (max(0, x - pad), max(0, y - pad)),
            (min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)),
            1, -1,
        )
        overlapping = next(
            (index for index, (existing, _) in enumerate(patches)
             if np.any((existing > 0) & (expanded > 0))),
            None,
        )
        if overlapping is not None:
            existing, existing_hue = patches[overlapping]
            patches[overlapping] = (
                np.maximum(existing, expanded), existing_hue,
            )
        else:
            patches.append((expanded, float(np.median(hsv[:, :, 0][component]))))
    return patches


def _normalized_center(mask):
    y, x = np.where(mask > 0)
    if not len(x):
        return 0.5, 0.5
    h, w = mask.shape
    return float(np.mean(x) / w), float(np.mean(y) / h)


def _repair_with_shifted_texture(rgb, mask):
    """Fill a compact badge from a phase-matched clean region on the same row."""
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return rgb.copy(), False
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    patch_width = x2 - x1
    ring = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    ring &= mask == 0
    ring_y, ring_x = np.where(ring)
    if ring_x.size < 24:
        return rgb.copy(), False

    # Never use the product-canvas/background as replacement fabric.  This was
    # the source of the large white rectangle seen on light-background garment
    # templates: a horizontal candidate could match the *ring* well while the
    # actual badge-sized source patch landed outside the shirt.  Estimate the
    # canvas colour from a narrow border and reject candidates dominated by it.
    border_pixels = np.concatenate((
        rgb[:max(1, height // 40)].reshape(-1, 3),
        rgb[-max(1, height // 40):].reshape(-1, 3),
        rgb[:, :max(1, width // 40)].reshape(-1, 3),
        rgb[:, -max(1, width // 40):].reshape(-1, 3),
    ), axis=0).astype(np.float32)
    border_colour = np.median(border_pixels, axis=0)

    def source_is_fabric(source_y, source_x):
        pixels = rgb[source_y, source_x].astype(np.float32)
        if not pixels.size:
            return False
        border_fraction = float(np.mean(
            np.linalg.norm(pixels - border_colour, axis=1) <= 22.0
        ))
        return border_fraction < .18

    # Prefer a vertical translation on the same chest column. This preserves
    # the pocket's horizontal perspective and avoids sampling the red placket
    # or the opposite side's differently curved fabric.
    vertical_shift = None
    vertical_score = float("inf")
    patch_height = int(ys.max()) - int(ys.min()) + 1
    minimum_shift_y = max(8, int(patch_height * 1.10))
    for shift in range(-height // 3, height // 3 + 1):
        if abs(shift) < minimum_shift_y:
            continue
        source_y = ring_y + shift
        badge_source_y = ys + shift
        if source_y.min() < 0 or source_y.max() >= height or badge_source_y.min() < 0 or badge_source_y.max() >= height:
            continue
        if np.any(mask[badge_source_y, xs] > 0):
            continue
        if not source_is_fabric(badge_source_y, xs):
            continue
        target_ring = rgb[ring_y, ring_x].astype(np.float32)
        source_ring = rgb[source_y, ring_x].astype(np.float32)
        score = float(np.mean(np.abs(target_ring - source_ring)))
        if score < vertical_score:
            vertical_score = score
            vertical_shift = shift

    # Horizontal transfer is a fallback for a badge whose vertical column is
    # obstructed by a pocket seam or image boundary. For plaid/check garments,
    # prefer a clean sample on the same chest row: it retains perspective and
    # avoids copying collar/placket pixels from another vertical garment band.
    minimum_shift_x = max(8, int(patch_width * 1.15))
    horizontal_shift = None
    horizontal_score = float("inf")
    for shift in range(-width // 2, width // 2 + 1):
        if abs(shift) < minimum_shift_x:
            continue
        source_x = ring_x + shift
        badge_source_x = xs + shift
        if source_x.min() < 0 or source_x.max() >= width or badge_source_x.min() < 0 or badge_source_x.max() >= width:
            continue
        if np.any(mask[ys, badge_source_x] > 0):
            continue
        if not source_is_fabric(ys, badge_source_x):
            continue
        target_ring = rgb[ring_y, ring_x].astype(np.float32)
        source_ring = rgb[ring_y, source_x].astype(np.float32)
        score = float(np.mean(np.abs(target_ring - source_ring)))
        if score < horizontal_score:
            horizontal_score = score
            horizontal_shift = shift

    if vertical_shift is None and horizontal_shift is None:
        return rgb.copy(), False
    # A badge printed inside a chest pocket should normally borrow fabric from
    # the same pocket column. Horizontal copying can cross the pocket seam and
    # create the pale flap/overlay seen in the final portrait. Prefer the
    # vertical continuation whenever its measured boundary error is reasonably
    # close; use horizontal only when vertical evidence is clearly worse.
    if vertical_shift is not None and (
        horizontal_shift is None or vertical_score <= horizontal_score + 12.0
    ):
        best_axis, best_shift = "y", vertical_shift
    else:
        best_axis, best_shift = "x", horizontal_shift
    cleaned = rgb.copy()
    if best_axis == "y":
        cleaned[ys, xs] = rgb[ys + best_shift, xs]
    else:
        cleaned[ys, xs] = rgb[ys, xs + best_shift]
    # Feather only the patch boundary. The badge core remains fully replaced,
    # while plaid lines meet without rectangular cut-and-paste edges.
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.6)
    # Gaussian blur lowers alpha at the inside edge too. That left a thin blue
    # remnant along cropped badges even though the detected rectangle was
    # correct. Keep every verified badge pixel fully replaced and feather only
    # the transition outside its mask.
    alpha[mask > 0] = 1.0
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    blended = rgb.astype(np.float32) * (1.0 - alpha) + cleaned.astype(np.float32) * alpha
    return np.rint(blended).clip(0, 255).astype(np.uint8), True


def _repair_with_periodic_texture(rgb, mask):
    """Continue a locally repeating fabric vertically through a badge.

    School plaids/checks repeat over a short vertical period. Copying a full
    rectangle from another chest location changes pocket perspective and looks
    stitched on; generic inpainting collapses intersecting lines into a blur.
    Estimate the repeat immediately above the verified badge and extend each
    original image column with the same phase, so vertical threads, pocket
    perspective and illumination remain local.
    """
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return rgb.copy(), False
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    patch_height = y2 - y1
    # A usable repeat needs two clean periods above the badge. Restrict the
    # search to fabric-scale intervals; tiny offsets reproduce noise and very
    # large offsets can cross the collar or pocket boundary.
    maximum_period = min(120, patch_height, y1 // 2)
    if maximum_period < 10:
        return rgb.copy(), False
    search_x1 = max(0, x1 - 4)
    search_x2 = min(width, x2 + 4)
    best_period = None
    best_score = float("inf")
    for period in range(10, maximum_period + 1):
        first_top = y1 - 2 * period
        second_top = y1 - period
        if first_top < 0:
            continue
        first = rgb[first_top:second_top, search_x1:search_x2].astype(np.float32)
        second = rgb[second_top:y1, search_x1:search_x2].astype(np.float32)
        if first.shape != second.shape or not first.size:
            continue
        # Median error is robust to a narrow pocket seam or a specular thread.
        score = float(np.median(np.abs(first - second)))
        if score < best_score:
            best_score = score
            best_period = period
    # Non-periodic/illustrated fabrics fall back to the existing guarded
    # texture repair instead of being forced into a repeated pattern.
    if best_period is None or best_score > 24.0:
        return rgb.copy(), False

    donor = rgb[y1 - best_period:y1, x1:x2]
    if donor.shape[0] != best_period or donor.shape[1] != x2 - x1:
        return rgb.copy(), False
    repaired = rgb.copy()
    for target_y in range(y1, y2):
        row_mask = mask[target_y, x1:x2] > 0
        if np.any(row_mask):
            source_row = donor[(target_y - y1) % best_period]
            target_row = repaired[target_y, x1:x2].copy()
            target_row[row_mask] = source_row[row_mask]
            repaired[target_y, x1:x2] = target_row

    # Blend only a two-pixel inner boundary. The core stays exact repeated
    # fabric; the outside pocket seam and surrounding plaid remain untouched.
    inner_edge = (mask > 0) & (
        cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2) == 0
    )
    if np.any(inner_edge):
        softened = cv2.GaussianBlur(repaired, (3, 3), .55)
        repaired[inner_edge] = softened[inner_edge]
    return repaired, True


def remove_template_badges(
    image: Image.Image,
    template: Image.Image,
    *,
    repair_mode: str = "auto",
    reference_bbox=None,
    reference_bbox_size=None,
):
    """Return corrected RGB and count; leave unsupported badge types untouched.

    Color alone is insufficient: each matched patch must be compact, interior
    to the lower image, and surrounded by dark cloth. No face/hair edits.
    """
    reference = np.array(template.convert("RGB"))
    rgb = np.array(image.convert("RGB"))
    image_is_reference = rgb.shape == reference.shape and np.array_equal(rgb, reference)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    count = 0
    # When VL supplied the visible emblem rectangle, use that measured region
    # for the person-free garment authority. Colour-only discovery can confuse
    # red cuffs or plaid intersections with a second badge and can expand over
    # the whole pocket. The bbox is produced on the analyzer thumbnail, so map
    # it back to this exact template size before repairing only those pixels.
    if image_is_reference and reference_bbox and reference_bbox_size:
        try:
            bx1, by1, bx2, by2 = (float(value) for value in reference_bbox)
            ref_width, ref_height = (float(value) for value in reference_bbox_size)
            x1 = int(round(bx1 * rgb.shape[1] / max(ref_width, 1.0)))
            y1 = int(round(by1 * rgb.shape[0] / max(ref_height, 1.0)))
            x2 = int(round(bx2 * rgb.shape[1] / max(ref_width, 1.0)))
            y2 = int(round(by2 * rgb.shape[0] / max(ref_height, 1.0)))
            # VL already returns the complete rectangular badge panel for the
            # garment analyzer. The former 30% expansion reached the pocket
            # seam and adjacent plaid rows, so the texture repair produced a
            # pale rectangular overlay that Qwen reproduced in the portrait.
            # Keep only a narrow antialias/border allowance around the measured
            # panel; the pocket construction must remain untouched.
            badge_width = max(1, x2 - x1)
            badge_height = max(1, y2 - y1)
            # Analyzer boxes tend to follow the pale printed crest and omit
            # the dark rectangular backing at its left/right edges. Expand
            # horizontally to include that backing, but keep vertical growth
            # narrow so the pocket opening and lower seam are never replaced.
            pad_x = max(3, int(round(badge_width * .30)))
            pad_y = max(3, int(round(min(badge_width, badge_height) * .08)))
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(rgb.shape[1], x2 + pad_x), min(rgb.shape[0], y2 + pad_y)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
                count = 1
        except (TypeError, ValueError):
            mask.fill(0)
            count = 0

    candidates = [] if count else (
        _patches(reference, allow_bottom_edge=True)
        + _brown_patches(reference, allow_bottom_edge=True)
    )
    for reference_patch, hue in candidates:
        reference_x, reference_y = _normalized_center(reference_patch)
        # Badges sit on the torso/chest, including the lower outer corner of a
        # full product cutout. The former x<=.80/y<=.94 limits discarded real
        # lower-right school crests and passed them directly to Qwen. Keep only
        # the extreme sleeve/canvas edges excluded; compactness, aspect, colour
        # and the template-to-output side match still guard cuffs and trim.
        if not (.12 <= reference_x <= .94 and .36 <= reference_y <= .99):
            continue
        generated_patches = (
            [(reference_patch, hue)]
            if image_is_reference else
            # Qwen can recolor a cyan circular crest into a royal-blue wide
            # block and crop its lower half at the passport edge. Match the
            # generated candidate by the verified template side/position, not
            # by an exact hue or near-square aspect that generation may alter.
            _patches(
                rgb,
                hue,
                allow_bottom_edge=True,
                max_aspect=4.0,
                rectangular=True,
            ) + _brown_patches(
                rgb, hue, allow_bottom_edge=True,
            )
        )
        for patch, _ in generated_patches:
            generated_x, generated_y = _normalized_center(patch)
            same_side = (reference_x >= .5) == (generated_x >= .5)
            # Badges belong to a side-chest region. Saturated lips, nostrils
            # and shaded skin can otherwise match a red/brown badge hue and
            # were being replaced with mirrored pixels after identity review.
            outside_face_column = abs(generated_x - .5) >= .12
            if (
                not same_side
                or not outside_face_column
                # In an upper-chest crop a real school badge is below the
                # collar. The reference-relative floor below is the primary
                # guard against red collar/placket false positives.
                # Full-shirt product references can place a chest badge near
                # the vertical middle before the portrait crop. Match it to
                # the reference position instead of assuming it is already in
                # the lower 40 percent of a passport frame.
                or generated_y < .38
                or generated_y < reference_y - .05
                or abs(generated_x - reference_x) > .22
                or abs(generated_y - reference_y) > .25
            ):
                continue
            if not np.any(mask[patch > 0]):
                count += 1
            mask[patch > 0] = 255
    if not count:
        return image.convert("RGB"), 0
    # Mirror-copy is the stable choice for a symmetric shirt reference: it
    # preserves the plaid scale and thread colours. Telea can radiate bright
    # checks into a white starburst, so rough-fit guides explicitly request
    # mirror mode and let Qwen blend the small boundary in its single pass.
    cleaned = rgb.copy()
    mirrored_fabric = rgb[:, ::-1]
    cleaned[mask > 0] = mirrored_fabric[mask > 0]
    # Check if the mirrored result looks plausible by comparing with the
    # local neighbourhood -- if it deviates heavily, use inpainting instead
    mirror_region = cleaned[mask > 0].astype(np.float32)
    surround_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8)) - mask
    surround_pixels = rgb[surround_mask > 0].astype(np.float32)
    if repair_mode == "periodic":
        periodic_cleaned, repaired = _repair_with_periodic_texture(rgb, mask)
        if repaired:
            return Image.fromarray(periodic_cleaned), count
        texture_cleaned, repaired = _repair_with_shifted_texture(rgb, mask)
        if repaired:
            return Image.fromarray(texture_cleaned), count
    if repair_mode == "texture":
        # Copy a phase-matched clean patch from the same garment row/column.
        # This makes the hidden badge inherit the uploaded uniform's actual
        # local colour, weave and pattern instead of a fixed fill colour.
        texture_cleaned, repaired = _repair_with_shifted_texture(rgb, mask)
        if repaired:
            return Image.fromarray(texture_cleaned), count
    if repair_mode not in {"auto", "mirror", "texture", "periodic"}:
        raise ValueError(f"Unsupported badge repair mode: {repair_mode}")
    if repair_mode == "auto" and surround_pixels.size > 0 and mirror_region.size > 0:
        mirror_mean = mirror_region.mean(axis=0)
        surround_mean = surround_pixels.mean(axis=0)
        color_diff = float(np.linalg.norm(mirror_mean - surround_mean))
        if color_diff > 35.0:
            # Mirror result looks implausible -- use inpainting instead
            cleaned = cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned), count


def remove_generated_backdrop_badge(
    image: Image.Image,
    cleaned_template: Image.Image,
    background_rgb,
    clothing_mask: np.ndarray,
):
    """Remove a compact backdrop-coloured badge copied into the uniform.

    Qwen can join a cyan/blue source badge to an identically coloured selected
    backdrop at the bottom of an upper-chest crop. It is then not a standalone
    colour component, so the ordinary badge detector cannot isolate it. This
    guard operates only inside SCHP clothing, only when the cleaned template
    does not contain that colour, and repairs from local garment texture.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if clothing_mask.shape != (height, width):
        clothing_mask = cv2.resize(
            clothing_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        clothing_mask = clothing_mask.astype(bool)
    if np.count_nonzero(clothing_mask) < 128:
        return image.convert("RGB"), 0

    template = np.asarray(cleaned_template.convert("RGBA"), dtype=np.uint8)
    template_rgb = template[:, :, :3]
    template_alpha = template[:, :, 3] > 32
    target = np.asarray(background_rgb, dtype=np.float32)
    template_distance = np.linalg.norm(
        template_rgb.astype(np.float32) - target, axis=2,
    )
    # The selected colour may legitimately be part of another uploaded
    # uniform. In that case color alone must never erase it.
    template_match_fraction = float(np.mean(
        template_distance[template_alpha] <= 38.0,
    )) if np.any(template_alpha) else 0.0
    if template_match_fraction >= .012:
        return image.convert("RGB"), 0

    distance = np.linalg.norm(rgb.astype(np.float32) - target, axis=2)
    candidate = (
        clothing_mask
        & (distance <= 38.0)
        & (np.indices((height, width))[0] >= int(height * .48))
    ).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    repair = np.zeros((height, width), dtype=np.uint8)
    found = 0
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        area_fraction = area / float(height * width)
        aspect = box_width / max(1, box_height)
        center_x = (x + box_width * .5) / width
        center_y = (y + box_height * .5) / height
        if not (
            .00025 <= area_fraction <= .035
            and .35 <= aspect <= 4.0
            and .10 <= center_x <= .94
            and .50 <= center_y <= 1.0
        ):
            continue
        component = labels == label
        ring = cv2.dilate(component.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        ring &= ~component
        ring &= clothing_mask
        if np.count_nonzero(ring) < 24:
            continue
        ring_colour = np.median(rgb[ring], axis=0)
        # A real template-coloured trim region has a similar local surround;
        # a copied badge/background block sharply disagrees with the fabric.
        if float(np.linalg.norm(ring_colour - target)) < 42.0:
            continue
        pad = max(3, int(round(min(box_width, box_height) * .14)))
        expanded = cv2.dilate(
            component.astype(np.uint8),
            np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8),
        )
        expanded &= clothing_mask.astype(np.uint8)
        repair[expanded > 0] = 255
        found += 1
    if not found:
        return image.convert("RGB"), 0
    repaired, success = _repair_with_shifted_texture(rgb, repair)
    if not success:
        # Local Telea repair is a last resort, still confined to the verified
        # clothing-only badge mask.
        repaired = cv2.inpaint(rgb, repair, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(repaired), found
