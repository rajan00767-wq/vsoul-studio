"""Create inspectable checkpoints for the portrait finishing stages."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.birefnet_service import background_removal
from pipelines.photo_restoration import PhotoRestorationService


OUTPUTS = ROOT / "outputs"
SOURCE = ROOT / "uploads" / "qwen_enh_454b9cdd_WhatsApp Image 2026-08-11 at 3.58.18 PM.jpeg"
QWEN_RAW = OUTPUTS / "gusuq_edit_1788534405_qwen_enhanced.png"
PREFIX = OUTPUTS / "stage_test_"


def save(stage: str, image: Image.Image) -> Path:
    path = Path(f"{PREFIX}{stage}.png")
    image.convert("RGB").save(path, format="PNG")
    print(f"{stage}: {path}")
    return path


def main() -> None:
    source = Image.open(SOURCE).convert("RGB")
    raw = Image.open(QWEN_RAW).convert("RGB")
    save("01_qwen_raw", raw)

    hair_fixed = PhotoRestorationService.normalize_hair_tone_from_source(source, raw)
    save("02_hair_tone", hair_fixed)

    source_hair_locked = PhotoRestorationService.lock_source_hair(source, raw, strength=0.72)
    save("02b_source_hair_lock", source_hair_locked)

    with io.BytesIO() as buffer:
        hair_fixed.save(buffer, format="PNG")
        rgba = Image.open(io.BytesIO(background_removal.remove_background(
            buffer.getvalue(), job_id="stage_test", source_image=None, use_schp=False,
        ))).convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    backdrop.paste(rgba, (0, 0), rgba)
    matted = backdrop.convert("RGB")
    save("03_solid_background", matted)

    final_bgr = PhotoRestorationService.natural_portrait_camera_look(
        cv2.cvtColor(np.array(matted), cv2.COLOR_RGB2BGR)
    )
    save("04_lighting_restoration", Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)))


if __name__ == "__main__":
    main()
