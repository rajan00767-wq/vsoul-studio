import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipelines.qwen_edit_pipeline import QwenEditPipeline


class UniformGarmentLockTests(unittest.TestCase):
    def test_restores_exact_rough_fabric_without_touching_face(self):
        height, width = 80, 60
        rough = np.full((height, width, 3), 255, dtype=np.uint8)
        generated = np.full((height, width, 3), 255, dtype=np.uint8)
        rough[10:40, 18:42] = (95, 70, 55)
        generated[10:40, 18:42] = (105, 80, 62)

        # The fitted guide has a red/blue check shirt. Qwen incorrectly
        # produced a plain beige sweater in the same garment region.
        rough[42:78, 8:52] = (220, 25, 35)
        rough[42:78:4, 8:52] = (30, 55, 110)
        rough[42:78, 8:52:5] = (245, 245, 245)
        generated[40:78, 5:55] = (165, 150, 130)

        rough_labels = np.zeros((height, width), dtype=np.uint8)
        rough_labels[40:78, 5:55] = 5
        with patch("pipelines.schp_service.parse", return_value={"labels": rough_labels}):
            result = QwenEditPipeline._lock_uniform_garment_from_rough(
                Image.fromarray(generated), Image.fromarray(rough),
            )

        output = np.asarray(result)
        np.testing.assert_array_equal(output[50:70, 15:45], rough[50:70, 15:45])
        np.testing.assert_array_equal(output[15:25, 20:40], generated[15:25, 20:40])

    def test_converts_rough_canvas_to_selected_backdrop_during_lock(self):
        height, width = 80, 60
        rough = np.full((height, width, 3), 255, dtype=np.uint8)
        generated = np.full((height, width, 3), (20, 120, 230), dtype=np.uint8)
        rough[42:78, 8:52] = (210, 25, 35)
        rough_labels = np.zeros((height, width), dtype=np.uint8)
        rough_labels[42:78, 8:52] = 5

        with patch("pipelines.schp_service.parse", return_value={"labels": rough_labels}):
            result = QwenEditPipeline._lock_uniform_garment_from_rough(
                Image.fromarray(generated),
                Image.fromarray(rough),
                background_rgb=(20, 120, 230),
            )

        output = np.asarray(result)
        np.testing.assert_allclose(output[65, 2], (20, 120, 230), atol=1)
        np.testing.assert_allclose(output[55:70, 15:45], rough[55:70, 15:45], atol=1)

    def test_arm_label_does_not_protect_a_patterned_sleeve(self):
        height, width = 100, 80
        rough = np.full((height, width, 3), 255, dtype=np.uint8)
        generated = np.full((height, width, 3), 255, dtype=np.uint8)
        rough[45:95, 8:72] = (35, 55, 100)
        rough[45:95:5, 8:72] = (235, 235, 240)
        generated[45:95, 8:72] = (145, 120, 95)

        rough_labels = np.zeros((height, width), dtype=np.uint8)
        rough_labels[45:95, 8:72] = 5
        output_labels = rough_labels.copy()
        output_labels[55:85, 8:22] = 14
        output_labels[55:85, 58:72] = 15

        with patch(
            "pipelines.schp_service.parse",
            side_effect=({"labels": output_labels}, {"labels": rough_labels}),
        ):
            result = QwenEditPipeline._lock_uniform_garment_from_rough(
                Image.fromarray(generated), Image.fromarray(rough),
            )

        output = np.asarray(result)
        np.testing.assert_allclose(output[62:76, 10:20], rough[62:76, 10:20], atol=1)
        np.testing.assert_allclose(output[62:76, 60:70], rough[62:76, 60:70], atol=1)

    def test_measured_neck_skin_is_not_covered_by_locked_fabric(self):
        height, width = 120, 90
        rough = np.full((height, width, 3), 255, dtype=np.uint8)
        generated = np.full((height, width, 3), 255, dtype=np.uint8)
        skin = (120, 78, 62)
        generated[18:55, 30:60] = skin
        generated[50:75, 36:54] = skin
        rough[48:115, 8:82] = (28, 48, 92)
        generated[48:115, 8:82] = (145, 125, 100)
        generated[50:75, 36:54] = skin

        rough_labels = np.zeros((height, width), dtype=np.uint8)
        rough_labels[48:115, 8:82] = 5
        output_labels = rough_labels.copy()
        output_labels[18:55, 30:60] = 13
        # Reproduce the parser failure from the real portrait: the lower neck
        # is labelled as clothing despite matching the generated face skin.
        output_labels[55:75, 36:54] = 5

        with patch(
            "pipelines.schp_service.parse",
            side_effect=({"labels": output_labels}, {"labels": rough_labels}),
        ):
            result = QwenEditPipeline._lock_uniform_garment_from_rough(
                Image.fromarray(generated), Image.fromarray(rough),
            )

        output = np.asarray(result)
        np.testing.assert_array_equal(output[56:64, 39:51], generated[56:64, 39:51])


if __name__ == "__main__":
    unittest.main()
