"""Conservative semantic tone correction after Qwen uniform refinement."""
import numpy as np
from PIL import Image


def finish_uniform_tones(image, labels, make_outer_black=False, correct_dark_hair=False):
    rgb = np.array(image.convert("RGB"))
    if labels.shape != rgb.shape[:2]:
        raise ValueError("Finishing mask dimensions do not match the portrait")
    result = rgb.astype(np.float32)
    metadata = {"outer_black_pixels": 0, "hair_glare_pixels": 0}

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

    if correct_dark_hair:
        hair = labels == 2
        count = np.count_nonzero(hair)
        if count >= labels.size * .02:
            luma = (result[:, :, 0] * .2126 + result[:, :, 1] * .7152
                    + result[:, :, 2] * .0722)
            # Dark hair should not become a broad silver/cyan surface. Compress
            # mid-bright glare while leaving near-white clips/accessories alone.
            glare = hair & (luma > 60.0) & (luma < 185.0)
            # A gamma curve retains strand-to-strand brightness differences,
            # unlike a flat cap, while pulling broad silver glare back to black.
            target = 255.0 * np.power(np.clip(luma / 255.0, 0, 1), 1.65)
            scale = target / np.maximum(luma, 1)
            scaled = result * scale[:, :, None]
            scaled[:, :, 2] *= .92
            result[glare] = scaled[glare]
            metadata["hair_glare_pixels"] = int(np.count_nonzero(glare))

    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8)), metadata
