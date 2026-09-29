import unittest
import numpy as np
from PIL import Image
from pipelines.uniform_finishing import finish_uniform_tones


class UniformFinishingTests(unittest.TestCase):
    def test_neutralizes_only_outer_garment(self):
        array = np.full((20, 20, 3), (30, 50, 90), dtype=np.uint8)
        labels = np.full((20, 20), 5, dtype=np.uint8)
        labels[10:, :] = 7
        result, meta = finish_uniform_tones(Image.fromarray(array), labels, make_outer_black=True)
        output = np.array(result)
        np.testing.assert_array_equal(output[2, 2], array[2, 2])
        self.assertEqual(len(set(output[15, 2].tolist())), 1)
        self.assertEqual(meta["outer_black_pixels"], 200)

    def test_preserves_outer_fabric_luminance_variation(self):
        array = np.zeros((20, 20, 3), dtype=np.uint8)
        array[:10] = (12, 25, 48)
        array[10:] = (35, 48, 72)
        labels = np.full((20, 20), 7, dtype=np.uint8)
        result, _ = finish_uniform_tones(Image.fromarray(array), labels, make_outer_black=True)
        output = np.array(result)
        self.assertLess(output[2, 2, 0], output[15, 2, 0])
        self.assertEqual(len(set(output[2, 2].tolist())), 1)
        self.assertEqual(len(set(output[15, 2].tolist())), 1)

    def test_matches_measured_outer_color_and_preserves_folds(self):
        array = np.zeros((24, 24, 3), dtype=np.uint8)
        array[:12] = (12, 12, 12)
        array[12:] = (42, 42, 42)
        labels = np.full((24, 24), 7, dtype=np.uint8)
        result, meta = finish_uniform_tones(
            Image.fromarray(array), labels, outer_target_rgb=(28, 38, 54),
        )
        output = np.array(result)
        self.assertGreater(int(output[:, :, 2].mean()), int(output[:, :, 0].mean()))
        self.assertLess(output[3, 3].mean(), output[20, 3].mean())
        self.assertEqual(meta["outer_color_pixels"], labels.size)

    def test_matches_neutral_ash_target_without_blue_cast(self):
        array = np.full((24, 24, 3), (20, 34, 58), dtype=np.uint8)
        labels = np.full((24, 24), 7, dtype=np.uint8)
        result, _ = finish_uniform_tones(
            Image.fromarray(array), labels, outer_target_rgb=(37, 37, 37),
        )
        median = np.median(np.asarray(result), axis=(0, 1))
        self.assertLessEqual(int(median.max() - median.min()), 2)

    def test_skips_incomplete_outer_garment_patch(self):
        array = np.full((40, 40, 3), (90, 90, 90), dtype=np.uint8)
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[25:30, 15:25] = 7
        result, meta = finish_uniform_tones(
            Image.fromarray(array), labels, outer_target_rgb=(30, 40, 60),
        )
        np.testing.assert_array_equal(np.asarray(result), array)
        self.assertEqual(meta["outer_color_pixels"], 0)

    def test_compresses_only_bright_dark_hair(self):
        array = np.full((40, 40, 3), 20, dtype=np.uint8)
        labels = np.full((40, 40), 2, dtype=np.uint8)
        array[8:13] = (145, 148, 152)
        result, meta = finish_uniform_tones(Image.fromarray(array), labels, correct_dark_hair=True)
        output = np.array(result)
        np.testing.assert_array_equal(output[30, 30], array[30, 30])
        self.assertLess(output[12, 12].mean(), array[12, 12].mean())
        self.assertGreater(meta["hair_glare_pixels"], 0)

    def test_smoothly_compresses_broad_hair_glare(self):
        array = np.full((40, 40, 3), 20, dtype=np.uint8)
        labels = np.full((40, 40), 2, dtype=np.uint8)
        array[5:25:2] = (145, 148, 152)
        result, meta = finish_uniform_tones(Image.fromarray(array), labels, correct_dark_hair=True)
        output = np.asarray(result)
        self.assertLess(output[13, 20].mean(), array[13, 20].mean())
        self.assertGreater(meta["hair_glare_pixels"], 0)
        self.assertFalse(meta["hair_glare_skipped"])

    def test_dark_hair_correction_protects_face_and_mask_edge(self):
        array = np.full((48, 48, 3), (145, 148, 152), dtype=np.uint8)
        labels = np.full((48, 48), 2, dtype=np.uint8)
        labels[16:32, 16:32] = 13
        result, _ = finish_uniform_tones(
            Image.fromarray(array), labels, correct_dark_hair=True,
        )
        output = np.array(result)
        np.testing.assert_array_equal(output[24, 24], array[24, 24])
        np.testing.assert_array_equal(output[0, 0], array[0, 0])

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            finish_uniform_tones(Image.new("RGB", (20, 20)), np.zeros((10, 10)))

    def test_restores_head_detail_without_changing_flat_background(self):
        array = np.full((24, 24, 3), 100, dtype=np.uint8)
        array[8:16, 8:16] = 120
        array[10:14, 10:14] = 132
        labels = np.zeros((24, 24), dtype=np.uint8)
        labels[8:16, 8:16] = 13
        result, meta = finish_uniform_tones(
            Image.fromarray(array), labels, restore_head_detail=True,
        )
        output = np.array(result)
        np.testing.assert_array_equal(output[0, 0], array[0, 0])
        self.assertGreater(meta["head_detail_pixels"], 0)
        self.assertFalse(np.array_equal(output[10:14, 10:14], array[10:14, 10:14]))

    def test_reduces_face_exposure_without_changing_background(self):
        array = np.full((40, 40, 3), (4, 126, 246), dtype=np.uint8)
        array[10:32, 10:30] = (225, 190, 170)
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[10:32, 10:30] = 13
        result, meta = finish_uniform_tones(
            Image.fromarray(array), labels, face_target_luma=150,
        )
        output = np.asarray(result)
        np.testing.assert_array_equal(output[0, 0], array[0, 0])
        self.assertLess(output[20, 20].mean(), array[20, 20].mean())
        self.assertGreater(meta["face_tone_pixels"], 0)
        self.assertLess(meta["face_luma_shift"], 0)


if __name__ == "__main__":
    unittest.main()
