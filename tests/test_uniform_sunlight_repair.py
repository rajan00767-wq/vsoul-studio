"""CPU regression test for the uniform crown-glare repair.

Mocks SCHP and InsightFace so no GPU/model weights are needed. Builds a
synthetic dark-hair head with a clipped (bright) crown and a colourful hair
accessory, then asserts the repair neutralizes the crown while leaving the
accessory, face and backdrop untouched and satisfying the residual-light
hair gate.
"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from pipelines.qwen_edit_pipeline import QwenEditPipeline

H, W = 260, 200
FACE_BBOX = [60, 90, 140, 190]


def _synthetic():
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :] = (4, 126, 246)  # flat backdrop
    labels = np.zeros((H, W), dtype=np.int32)

    # Hair: crown + sides + lower hair, excluding the face box.
    hair = np.zeros((H, W), dtype=bool)
    hair[12:228, 40:160] = True
    hair[90:190, 60:140] = False
    labels[hair] = 2
    rgb[hair] = (30, 25, 22)  # dark hair

    # Clipped near-neutral crown glare.
    rgb[15:60, 60:140] = (230, 225, 215)
    # Colourful accessory (pink flower) inside the hair window.
    rgb[40:70, 145:175] = (230, 120, 180)
    labels[40:70, 145:175] = 2

    # Face skin (non-hair).
    labels[90:190, 60:140] = 13
    rgb[90:190, 60:140] = (200, 160, 130)
    return rgb, labels


def _hair_metrics(rgb):
    x1, y1, x2, y2 = FACE_BBOX
    fw, fh = max(24, x2 - x1), max(24, y2 - y1)
    yy, xx = np.indices((H, W))
    window = (
        (xx >= max(0, x1 - int(fw * .85))) & (xx <= min(W - 1, x2 + int(fw * .85)))
        & (yy >= max(0, y1 - int(fh * .78))) & (yy <= min(H - 1, y2 + int(fh * .38)))
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    low = cv2.GaussianBlur(lab[:, :, 0], (0, 0), sigmaX=max(5.0, fw * .065))
    return low, window


class UniformSunlightRepairTests(unittest.TestCase):
    def _run(self, rgb, labels):
        detector = Mock()
        detector.get.return_value = [SimpleNamespace(bbox=FACE_BBOX)]
        restoration = SimpleNamespace(_get_insight_app=lambda: detector)
        schp = SimpleNamespace(parse=lambda img: {"labels": labels})
        with patch.dict(sys.modules, {
            "pipelines.photo_restoration": restoration,
            "pipelines.schp_service": schp,
        }):
            return QwenEditPipeline._reduce_uniform_hair_sun_glare(
                Image.fromarray(rgb, "RGB")
            )

    def test_crown_neutralized_accessory_and_face_preserved(self):
        rgb, labels = _synthetic()
        out = np.asarray(self._run(rgb, labels).convert("RGB"))

        # Crown low-frequency luminance must drop toward the shaded target.
        low_before, window = _hair_metrics(rgb)
        low_after, _ = _hair_metrics(out)
        crown = np.zeros((H, W), dtype=bool)
        crown[20:55, 70:130] = True
        self.assertGreater(
            float(low_before[crown].mean()) - float(low_after[crown].mean()), 20.0
        )

        # Residual-light hair gate must now pass (bright fraction below .06).
        x1, y1, x2, y2 = FACE_BBOX
        fw = max(24, x2 - x1)
        hair = (labels == 2) & window
        hv = low_after[hair]
        p25, p90 = np.percentile(hv, (25, 90))
        floor = max(145.0, p25 + 62.0)
        bright_fraction = float(np.mean(hv >= floor))
        hair_hard = bool(p90 >= 142.0 and (p90 - p25) >= 62.0 and bright_fraction >= .06)
        self.assertFalse(hair_hard)

        # Isolate the repair from the lossy whole-image LAB roundtrip: where the
        # feathered alpha is zero the output must equal a no-op roundtrip.
        control = cv2.cvtColor(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), cv2.COLOR_LAB2RGB
        )
        # Face skin and backdrop are non-hair and must be unchanged.
        self.assertTrue(np.array_equal(out[110:180, 70:130], control[110:180, 70:130]))
        self.assertTrue(np.array_equal(out[0:10, 0:20], control[0:10, 0:20]))
        # The colourful accessory must not be neutralized like the glare: its
        # luminance drop is only feather-level, far below the crown's drop.
        lab_b = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        lab_a = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        acc_drop = float(lab_b[44:52, 155:172].mean() - lab_a[44:52, 155:172].mean())
        self.assertLess(acc_drop, 15.0)

    def test_no_hair_returns_unchanged(self):
        rgb, labels = _synthetic()
        labels[:] = 0  # no hair parsed
        out = np.asarray(self._run(rgb, labels).convert("RGB"))
        self.assertTrue(np.array_equal(out, rgb))


if __name__ == "__main__":
    unittest.main()