"""Conservative semantic tone correction after Qwen uniform refinement."""
import cv2
import numpy as np
from PIL import Image


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
        hair[:2, :] = False
        hair[-2:, :] = False
        hair[:, :2] = False
        hair[:, -2:] = False
        face_guard = cv2.dilate(
            (labels == 13).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1,
        ).astype(bool)
        hair &= ~face_guard
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
            accessory_core = cv2.morphologyEx(
                bright, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
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
