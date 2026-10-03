"""Conservative semantic tone correction after Qwen uniform refinement."""
import cv2
import numpy as np
from PIL import Image


def restore_uniform_hair_accessories(image, aligned_source, labels, aligned_source_labels):
    """Blend aligned original hair/accessories without touching face or clothing."""
    rgb = np.array(image.convert("RGB"))
    source_rgb = np.array(aligned_source.convert("RGB"))
    expected = rgb.shape[:2]
    if source_rgb.shape[:2] != expected or labels.shape != expected or aligned_source_labels.shape != expected:
        raise ValueError("Head restoration inputs must have matching dimensions")

    generated_hair = (labels == 2).astype(np.uint8)
    source_hair = (aligned_source_labels == 2).astype(np.uint8)
    # Qwen now receives the original portrait as its primary image, so it
    # already carries the intended hair silhouette. A full source-hair paste
    # can import matte errors, outdoor shadow, or background pixels around
    # long strands. Restore only the common, confident hair interior and let
    # Qwen keep its own clean outer edge.
    generated_hair_core = cv2.erode(
        generated_hair,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    generated_hair_safe = cv2.dilate(
        generated_hair_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    restore_mask = source_hair.astype(bool) & generated_hair_safe.astype(bool)
    face_guard = cv2.dilate(
        (labels == 13).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1,
    ).astype(bool)
    face_y, face_x = np.where(labels == 13)
    if face_x.size:
        x1, x2 = int(face_x.min()), int(face_x.max())
        y1, y2 = int(face_y.min()), int(face_y.max())
        face_w, face_h = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        # Parsing often labels a sunlit forehead fringe as hair. Protect a
        # geometric facial oval derived from the actual parsed face extent.
        cv2.ellipse(
            face_guard.view(np.uint8),
            ((x1 + x2) // 2, (y1 + y2) // 2),
            (max(1, int(face_w * .57)), max(1, int(face_h * .64))),
            0, 0, 360, 1, -1,
        )
        forehead_top = 0
        forehead_bottom = min(labels.shape[0], y1 + int(face_h * .30))
        forehead_left = max(0, x1 - int(face_w * .06))
        forehead_right = min(labels.shape[1], x2 + int(face_w * .06) + 1)
        face_guard[forehead_top:forehead_bottom, forehead_left:forehead_right] = True
    restore_mask &= ~face_guard
    restore_mask[:2, :] = False
    restore_mask[-2:, :] = False
    restore_mask[:, :2] = False
    restore_mask[:, -2:] = False
    metadata = {"hair_restore_pixels": int(np.count_nonzero(restore_mask)), "applied": False}
    if np.count_nonzero(restore_mask) < labels.size * .015:
        return image.convert("RGB"), metadata

    target_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_core = restore_mask & generated_hair.astype(bool)
    if np.count_nonzero(target_core) < 128:
        target_core = restore_mask
    # Match the uploaded hair to the generated studio exposure before
    # compositing. Bounded shifts retain the original accessory colours.
    limits = (24.0, 10.0, 10.0)
    for channel, limit in enumerate(limits):
        source_median = float(np.median(source_lab[:, :, channel][restore_mask]))
        target_median = float(np.median(target_lab[:, :, channel][target_core]))
        source_lab[:, :, channel] += np.clip(target_median - source_median, -limit, limit)
    matched_source = cv2.cvtColor(
        source_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
    ).astype(np.float32)

    source_luma = source_lab[:, :, 0]
    # Clips, ribbons and flowers need stronger recovery than ordinary strands.
    hair_median = np.median(source_lab[restore_mask], axis=0)
    color_distance = np.linalg.norm(source_lab - hair_median[None, None, :], axis=2)
    accessory_candidates = restore_mask & (source_luma > 95) & (color_distance > 24)
    accessory_candidates = cv2.morphologyEx(
        accessory_candidates.astype(np.uint8), cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        accessory_candidates, connectivity=8,
    )
    accessory = np.zeros_like(restore_mask)
    max_accessory_area = max(32, int(labels.size * .003))
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if 8 <= area <= max_accessory_area:
            accessory |= component_labels == component
    base_alpha = cv2.GaussianBlur(
        restore_mask.astype(np.uint8) * 255, (7, 7), 0,
    ).astype(np.float32) / 255.0 * .38
    accessory_alpha = cv2.GaussianBlur(
        accessory.astype(np.uint8) * 255, (5, 5), 0,
    ).astype(np.float32) / 255.0 * .72
    alpha = np.clip(base_alpha + accessory_alpha, 0.0, .82)
    result = rgb.astype(np.float32) * (1.0 - alpha[:, :, None]) + matched_source * alpha[:, :, None]
    metadata["accessory_restore_pixels"] = int(np.count_nonzero(accessory))
    metadata["applied"] = True
    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8)), metadata


def restore_uniform_face_detail(image, labels):
    """Restore restrained local facial detail without regenerating the face."""
    rgb = np.array(image.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Face restoration mask dimensions do not match the portrait")

    face = (labels == 13).astype(np.uint8)
    metadata = {"face_detail_pixels": 0, "face_detail_applied": False}
    if np.count_nonzero(face) < labels.size * .015:
        return image.convert("RGB"), metadata

    # Stay inside the parsed face so hair, jewellery, uniform and backdrop are
    # byte-identical. Feathering avoids a visible oval restoration boundary.
    core = cv2.erode(
        face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    if np.count_nonzero(core) < 128:
        core = face
    alpha = cv2.GaussianBlur(core * 255, (9, 9), 0).astype(np.float32) / 255.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    original_luma = lab[:, :, 0].astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8))
    local_luma = clahe.apply(lab[:, :, 0]).astype(np.float32)

    # Keep the original facial exposure while recovering feature contrast.
    core_mask = core.astype(bool)
    local_luma += float(np.median(original_luma[core_mask]) - np.median(local_luma[core_mask]))
    contrast_luma = original_luma * .64 + local_luma * .36
    fine_blur = cv2.GaussianBlur(contrast_luma, (0, 0), .72)
    broad_blur = cv2.GaussianBlur(contrast_luma, (0, 0), 1.45)
    detail = (contrast_luma - fine_blur) * .82 + (contrast_luma - broad_blur) * .18
    restored_luma = contrast_luma + np.clip(detail, -12.0, 12.0) * .90

    restored_lab = lab.copy()
    restored_lab[:, :, 0] = np.rint(restored_luma).clip(0, 255).astype(np.uint8)
    restored = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    result = rgb.astype(np.float32) * (1.0 - alpha[:, :, None]) + restored * alpha[:, :, None]
    metadata["face_detail_pixels"] = int(np.count_nonzero(core))
    metadata["face_detail_applied"] = True
    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8)), metadata


def finish_uniform_tones(
    image, labels, make_outer_black=False, correct_dark_hair=False, restore_head_detail=False,
    outer_target_rgb=None, face_target_luma=None,
):
    rgb = np.array(image.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Finishing mask dimensions do not match the portrait")
    result = rgb.astype(np.float32)
    metadata = {
        "outer_black_pixels": 0, "outer_color_pixels": 0,
        "hair_glare_pixels": 0, "hair_glare_skipped": False, "head_detail_pixels": 0,
        "face_tone_pixels": 0, "face_luma_shift": 0.0,
    }

    if make_outer_black:
        outer = labels == 7  # LIP coat / outer garment
        if np.count_nonzero(outer) >= labels.size * .02:
            # Neutral luma retains photographed weave, seams and folds while
            # removing the blue chroma Qwen tends to add to near-black cloth.
            luma = (result[:, :, 0] * .2126 + result[:, :, 1] * .7152
                    + result[:, :, 2] * .0722)
            neutral = np.stack((luma, luma, luma), axis=2)
            result[outer] = neutral[outer]
            metadata["outer_black_pixels"] = int(np.count_nonzero(outer))

    if outer_target_rgb is not None:
        outer = labels == 7
        # A real passport-crop vest occupies a substantial lower region. SCHP
        # occasionally returns a small isolated coat patch; recolouring that
        # patch creates a visible painted mark, so require plausible coverage.
        if np.count_nonzero(outer) >= labels.size * .08:
            lab = cv2.cvtColor(result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            current = np.median(lab[outer], axis=0)
            target_pixel = np.asarray([[outer_target_rgb]], dtype=np.uint8)
            target = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
            # Keep folds and highlights while matching the template's median
            # luminance and chroma. Limit L movement so cloth stays photographic.
            shift = target - current
            shift[0] = np.clip(shift[0], -18.0, 18.0)
            adjusted = lab.copy()
            adjusted[outer] += shift
            adjusted = cv2.cvtColor(adjusted.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            result[outer] = adjusted[outer]
            metadata["outer_color_pixels"] = int(np.count_nonzero(outer))

    if correct_dark_hair:
        hair = (labels == 2).astype(np.uint8)
        # SCHP can leak the hair label onto forehead skin and the outer matte
        # boundary. Work only on the mask interior and protect a generous
        # band around pixels confidently parsed as face.
        hair = cv2.erode(
            hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        ).astype(bool)
        face_y, face_x = np.where(labels == 13)
        upper_hairline = np.zeros_like(hair)
        if face_x.size:
            x1, x2 = int(face_x.min()), int(face_x.max())
            y1 = int(face_y.min())
            face_w = max(1, x2 - x1 + 1)
            left = max(0, x1 - int(face_w * .12))
            right = min(labels.shape[1], x2 + int(face_w * .12) + 1)
            # Qwen glare at the parting is sometimes parsed as face. Pixels
            # above the actual face start and inside the head span still need
            # hair correction, but background label 0 must remain untouched.
            upper_hairline[:y1, left:right] = labels[:y1, left:right] != 0
            hair |= upper_hairline
        hair[:2, :] = False
        hair[-2:, :] = False
        hair[:, :2] = False
        hair[:, -2:] = False
        face_guard = cv2.dilate(
            (labels == 13).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1,
        ).astype(bool)
        hair &= ~(face_guard & ~upper_hairline)
        count = np.count_nonzero(hair)
        if count >= labels.size * .02:
            lab = cv2.cvtColor(
                result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB,
            ).astype(np.float32)
            luma = lab[:, :, 0]

            # Filled pale accessories form solid bright regions, while glare
            # on strands is narrow texture. Protect the former before tone
            # compression so flowers, clips and bows retain their real colour.
            bright = (hair & (luma > 145.0)).astype(np.uint8)
            bright = cv2.morphologyEx(
                bright, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
                bright, connectivity=8,
            )
            accessory_core = np.zeros_like(bright)
            max_accessory_area = max(32, int(labels.size * .003))
            for component in range(1, component_count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if 8 <= area <= max_accessory_area:
                    accessory_core[component_labels == component] = 1
            accessory_guard = cv2.dilate(
                accessory_core,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            ).astype(bool)
            correction_mask = hair & ~accessory_guard

            # Compress the bright end smoothly across dark hair. This retains
            # strand contrast and avoids isolated dark marks at the parting.
            strength = np.clip((luma - 72.0) / 108.0, 0.0, 1.0)
            strength *= correction_mask.astype(np.float32)
            alpha = cv2.GaussianBlur(strength, (7, 7), 1.4)
            adjustment = alpha * 38.0
            lab[:, :, 0] -= adjustment
            dark_reference = correction_mask & (luma < 82.0)
            if np.count_nonzero(dark_reference) >= 64:
                target_a = float(np.median(lab[:, :, 1][dark_reference]))
                target_b = float(np.median(lab[:, :, 2][dark_reference]))
                chroma_alpha = alpha * 0.55
                lab[:, :, 1] += (target_a - lab[:, :, 1]) * chroma_alpha
                lab[:, :, 2] += (target_b - lab[:, :, 2]) * chroma_alpha
            corrected = cv2.cvtColor(
                lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
            ).astype(np.float32)
            boundary = cv2.GaussianBlur(
                correction_mask.astype(np.uint8) * 255, (7, 7), 1.4,
            ).astype(np.float32) / 255.0
            result = result * (1.0 - boundary[:, :, None]) + corrected * boundary[:, :, None]
            metadata["hair_glare_pixels"] = int(np.count_nonzero(adjustment > 1.0))

    if face_target_luma is not None:
        face = labels == 13
        if np.count_nonzero(face) >= labels.size * .015:
            lab = cv2.cvtColor(
                result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB,
            ).astype(np.float32)
            values = lab[:, :, 0][face]
            low, high = np.percentile(values, (15, 85))
            stable = values[(values >= low) & (values <= high)]
            current = float(np.median(stable if stable.size else values))
            # Only correct a meaningful exposure mismatch and keep the move
            # conservative. The reference comes from this uploaded portrait,
            # so no fixed skin colour or skin-tone preset is introduced.
            shift = float(np.clip(float(face_target_luma) - current, -18.0, 6.0))
            if abs(shift) >= 2.0:
                alpha = cv2.GaussianBlur(
                    face.astype(np.uint8) * 255, (11, 11), 0,
                ).astype(np.float32) / 255.0
                lab[:, :, 0] += alpha * shift
                corrected = cv2.cvtColor(
                    lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB,
                ).astype(np.float32)
                # LAB round-tripping can move even untouched RGB values by a
                # few levels. Composite only within the face feather so the
                # selected backdrop and garment pixels remain byte-exact.
                result = result * (1.0 - alpha[:, :, None]) + corrected * alpha[:, :, None]
                metadata["face_tone_pixels"] = int(np.count_nonzero(face))
                metadata["face_luma_shift"] = round(shift, 2)

    if restore_head_detail:
        head = np.isin(labels, (2, 13))
        if np.count_nonzero(head) >= labels.size * .03:
            # Recover restrained micro-contrast from Qwen's own pixels. No
            # source pixels are copied, so outdoor lighting cannot return.
            luma = (result[:, :, 0] * .2126 + result[:, :, 1] * .7152
                    + result[:, :, 2] * .0722)
            blur = cv2.GaussianBlur(luma, (0, 0), 0.85)
            detail = np.clip(luma - blur, -6.0, 6.0) * .62
            alpha = cv2.GaussianBlur(
                head.astype(np.uint8) * 255, (5, 5), 0,
            ).astype(np.float32) / 255.0
            result += detail[:, :, None] * alpha[:, :, None]
            metadata["head_detail_pixels"] = int(np.count_nonzero(head))

    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8)), metadata
