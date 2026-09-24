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

    def test_compresses_only_bright_dark_hair(self):
        array = np.full((20, 20, 3), 20, dtype=np.uint8)
        labels = np.full((20, 20), 2, dtype=np.uint8)
        array[:5] = (120, 140, 160)
        result, meta = finish_uniform_tones(Image.fromarray(array), labels, correct_dark_hair=True)
        output = np.array(result)
        np.testing.assert_array_equal(output[10, 10], array[10, 10])
        self.assertLess(output[2, 2].mean(), array[2, 2].mean())
        self.assertGreater(meta["hair_glare_pixels"], 0)

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            finish_uniform_tones(Image.new("RGB", (20, 20)), np.zeros((10, 10)))


if __name__ == "__main__":
    unittest.main()
