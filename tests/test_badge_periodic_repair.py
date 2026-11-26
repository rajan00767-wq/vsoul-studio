import unittest

import numpy as np

from pipelines.uniform_badge_cleanup import _repair_with_periodic_texture

PERIOD = 40
CREST = (105, 55, 45)
# Unequal band widths, so the check has no sub-period that could tile by luck.
BANDS = (((41, 51, 91), 18), ((248, 249, 251), 10), ((240, 35, 39), 4), ((179, 183, 200), 8))


def _stripe(length):
    out = np.zeros((length, 3), np.uint8)
    index = 0
    while index < length:
        for colour, band_width in BANDS:
            out[index:index + band_width] = colour
            index += band_width
            if index >= length:
                break
    return out


def _plaid(width=400, height=500):
    """Woven check with a vertical illumination falloff, like a real garment.

    The falloff matters: it makes fabric copied from a distant row measurably
    wrong, which is the failure 'texture' mode produced on job 16682cec.
    """
    columns = np.broadcast_to(_stripe(width), (height, width, 3)).astype(int)
    rows = np.broadcast_to(_stripe(height)[:, None, :], (height, width, 3)).astype(int)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    weave = (
        12 * np.sin(2 * np.pi * (grid_y % PERIOD) / PERIOD * 3)
        + 12 * np.sin(2 * np.pi * (grid_x % PERIOD) / PERIOD * 3)
    )[..., None]
    shading = (grid_y / height * 40.0 - 20.0)[..., None]
    return np.clip((columns + rows) // 2 + weave + shading, 0, 255).astype(np.uint8)


class BadgePeriodicRepairTests(unittest.TestCase):
    """job 16682cec: the template's pocket crest was filled by 'texture' mode,
    which copies a badge-sized block from elsewhere on the chest. On a check
    that block lands out of phase, and Qwen reproduced the resulting
    rectangle in the delivered portrait. The pipeline now asks for 'periodic'
    repair, which must continue the measured vertical repeat in place."""

    def setUp(self):
        self.clean = _plaid()
        # 58px tall: not a whole multiple of the 40px repeat, so nothing lines
        # up by accident.
        self.box = (302, 360, 180, 280)
        y1, y2, x1, x2 = self.box
        self.rgb = self.clean.copy()
        self.rgb[y1:y2, x1:x2] = CREST
        self.mask = np.zeros(self.rgb.shape[:2], np.uint8)
        self.mask[y1:y2, x1:x2] = 255

    def _patch(self, image):
        y1, y2, x1, x2 = self.box
        return image[y1:y2, x1:x2].astype(int)

    def test_periodic_repair_restores_the_check_in_phase(self):
        repaired, ok = _repair_with_periodic_texture(self.rgb, self.mask)
        self.assertTrue(ok)
        self.assertLess(
            np.abs(self._patch(repaired) - self._patch(self.clean)).mean(), 8.0,
        )

    def test_no_crest_colour_survives(self):
        repaired, ok = _repair_with_periodic_texture(self.rgb, self.mask)
        self.assertTrue(ok)
        distance = np.abs(self._patch(repaired) - np.array(CREST)).sum(axis=2)
        self.assertGreater(distance.min(), 30)

    def test_plain_fabric_repairs_to_the_surrounding_colour(self):
        """Flat cloth has a trivial repeat; the fill must still be the fabric,
        never the crest."""
        flat = np.full((500, 400, 3), (120, 130, 150), np.uint8)
        y1, y2, x1, x2 = self.box
        flat[y1:y2, x1:x2] = CREST
        repaired, ok = _repair_with_periodic_texture(flat, self.mask)
        self.assertTrue(ok)
        self.assertLess(
            np.abs(self._patch(repaired) - np.array([120, 130, 150])).mean(), 4.0,
        )


if __name__ == "__main__":
    unittest.main()
