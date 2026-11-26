"""Tests for seating the conditioning neck at the measured collar line.

Qwen obeys conditioning pixels, not the stated gap. Measured on job 5a1bbeaa:
the prompt said 0.204 face-heights and the decode rendered 0.589; feeding the
whole rough composite as image 1 pulled it to 0.529 but copied the composite's
pasted garment boundary as hard notches beside the jaw.

seat_conditioning_neck gives the geometry without the fabric: the neck column
fades to the backdrop below the measured seat, leaving no garment edge to copy.
These tests pin that it fades where it should, leaves everything else alone,
and never produces the hard step that made the rough composite unusable.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")


def _load_helper():
    """Exec just the helper, so no model or heavy import is needed."""
    tree = ast.parse(SOURCE)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "seat_conditioning_neck")
    ns = {"np": np, "Image": Image}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "seat", "exec"), ns)
    return ns["seat_conditioning_neck"]


WHITE = (255, 255, 255)


class SeatConditioningNeckTests(unittest.TestCase):
    def setUp(self):
        self.seat = _load_helper()
        # Skin-coloured person on a white field, neck column centred at x=50.
        self.person = Image.new("RGB", (100, 200), WHITE)
        px = np.array(self.person)
        px[:, 30:70] = (180, 130, 90)
        self.person = Image.fromarray(px)
        self.seat_y = 100
        self.cx = 50.0
        self.half = 20.0

    def _run(self, **kw):
        return np.array(self.seat(
            self.person, self.seat_y, self.cx, self.half, WHITE, **kw
        ).convert("RGB")).astype(int)

    def test_pixels_above_the_seat_are_untouched(self):
        out = self._run()
        before = np.array(self.person).astype(int)
        self.assertEqual(
            np.abs(out[: self.seat_y] - before[: self.seat_y]).max(), 0,
            "the jaw and everything above the collar line must not move",
        )

    def test_neck_column_fades_to_backdrop_below_the_seat(self):
        out = self._run()
        deep = out[self.seat_y + 40, int(self.cx)]
        self.assertTrue(
            (np.abs(deep - np.array(WHITE)) <= 2).all(),
            f"neck column should reach the backdrop, got {deep}",
        )

    def test_shoulders_outside_the_column_are_preserved(self):
        out = self._run()
        before = np.array(self.person).astype(int)
        # The column is cx +/- half, i.e. x in [30, 70]. x=31 lies just inside
        # the body but outside the faded column once the ramp is accounted
        # for; anything at or beyond the column edge must be untouched.
        for x in (30, 29, 20):
            self.assertEqual(
                list(out[self.seat_y + 40, x]), list(before[self.seat_y + 40, x]),
                f"x={x} is outside the neck column and must not fade",
            )

    def test_fade_is_gradual_not_a_hard_step(self):
        # The rough composite failed because its boundary was a hard edge the
        # model reproduced. No single row may jump the full way to backdrop.
        out = self._run()
        column = out[:, int(self.cx)].mean(1)
        steps = np.abs(np.diff(column[self.seat_y - 2: self.seat_y + 40]))
        self.assertLess(
            steps.max(), 60,
            f"fade must be a ramp, largest single-row jump was {steps.max():.0f}",
        )

    def test_fade_has_soft_horizontal_shoulders(self):
        # A rectangular cut is itself an edge. Fading toward a white backdrop
        # raises brightness, so the centre must be the *brightest* point and
        # the column edge must sit between skin and backdrop.
        out = self._run().mean(2)
        row = out[self.seat_y + 30]
        centre = row[int(self.cx)]
        edge = row[int(self.cx - self.half) + 3]
        skin = row[30]  # column edge; body spans x=30..70
        self.assertGreater(centre, edge, "centre should fade most")
        self.assertGreater(edge, skin, "column edge should fade partially")

    def test_seat_at_or_past_the_frame_is_a_no_op(self):
        for bad in (0, -5, 199, 500):
            out = np.array(self.seat(
                self.person, bad, self.cx, self.half, WHITE
            ).convert("RGB")).astype(int)
            if bad >= 199 or bad < 1:
                self.assertEqual(
                    np.abs(out - np.array(self.person).astype(int)).max(), 0,
                    f"seat {bad} is out of range and must change nothing",
                )

    def test_backdrop_colour_is_honoured(self):
        blue = (4, 126, 246)
        out = np.array(self.seat(
            self.person, self.seat_y, self.cx, self.half, blue
        ).convert("RGB")).astype(int)
        deep = out[self.seat_y + 40, int(self.cx)]
        self.assertTrue(
            (np.abs(deep - np.array(blue)) <= 2).all(),
            f"must fade to the selected backdrop, got {deep}",
        )


class WiringTests(unittest.TestCase):
    def test_seating_is_off_by_default(self):
        # Measured negative result: seating made the neck slightly longer
        # (0.280 -> 0.292 exposed skin). It stays available behind the flag
        # but must not run in normal jobs.
        self.assertIn('os.getenv("VSOUL_UNIFORM_SEAT_NECK", "0")', SOURCE)

    def test_rough_primary_experiment_does_not_also_seat(self):
        # Seating a composite that already has a pasted collar would stack two
        # geometry cues; the branches must stay exclusive.
        idx = SOURCE.index('VSOUL_UNIFORM_ROUGH_PRIMARY')
        tail = SOURCE[idx: idx + 1200]
        self.assertIn("elif", tail)

    def test_rough_composite_bridges_jaw_and_neck_to_collar_opening(self):
        """No white backdrop column or gap may exist between jaw/neck and collar."""
        from pipelines.uniform_composite import build_rough_composite
        # Synthetic portrait with skin face and neck, and undersegmented jaw
        person = Image.new("RGB", (120, 160), WHITE)
        p_arr = np.array(person)
        skin = (175, 115, 80)
        p_arr[20:70, 35:85] = skin
        p_arr[70:100, 42:78] = skin  # neck
        # Garment with collar opening
        garment = Image.new("RGBA", (120, 100), (10, 20, 40, 0))
        g_arr = np.array(garment)
        g_arr[15:100, 10:110] = (20, 30, 60, 255)
        # Collar V opening in the garment: rows 15..45, center cols 50..70 transparent
        for r in range(15, 45):
            hw = int(12 * (1.0 - (r - 15) / 30.0))
            g_arr[r, 60 - hw : 60 + hw] = (0, 0, 0, 0)
        garment = Image.fromarray(g_arr, "RGBA")
        labels = np.zeros((160, 120), dtype=np.uint8)
        # SCHP undersegments: face label 13 only in central part, stopping at chin
        labels[20:70, 42:78] = 13
        labels[90:, 10:110] = 5  # clothing below

        rough, meta = build_rough_composite(
            Image.fromarray(p_arr), garment, labels,
            (35, 20, 85, 70), WHITE,
        )
        r_arr = np.array(rough)
        # In the neck transition rows (68 to 85), across the central neck corridor (45..75),
        # there must be NO backdrop pixels
        for y in range(68, 85):
            for x in range(46, 75):
                pixel = r_arr[y, x]
                is_bg = np.all(pixel >= [245, 245, 245])
                self.assertFalse(
                    is_bg,
                    f"Backdrop hole found at y={y}, x={x}: RGB={pixel}",
                )


if __name__ == "__main__":
    unittest.main()
