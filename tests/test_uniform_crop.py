import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from pipelines.qwen_edit_pipeline import QwenEditPipeline


class UniformCropTests(unittest.TestCase):
    def test_tight_upload_is_padded_to_passport_face_scale_before_generation(self):
        background = (4, 126, 246)
        image = Image.new("RGB", (390, 508), (34, 72, 30))
        draw = ImageDraw.Draw(image)
        draw.ellipse((112, 109, 294, 340), fill=(48, 34, 28))
        draw.rectangle((38, 320, 352, 508), fill=(72, 78, 86))

        result = QwenEditPipeline._source_identity_reference(
            image,
            (112, 109, 294, 340),
            944,
            1216,
            background_color=background,
            target_face_fraction=.35,
        )

        rgb = np.asarray(result)
        face = (
            (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
            & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
            & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
        )
        rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
        self.assertGreater(float(rows[-1] - rows[0] + 1) / result.height, .31)
        self.assertLess(float(rows[-1] - rows[0] + 1) / result.height, .37)
        self.assertTrue(np.all(rgb[0, 0] == background))

    def test_flat_background_detector_accepts_uniform_passport_field(self):
        image = Image.new("RGB", (560, 720), (4, 126, 246))
        draw = ImageDraw.Draw(image)
        draw.ellipse((150, 55, 410, 355), fill=(42, 28, 24))
        draw.rectangle((55, 330, 505, 720), fill=(220, 30, 45))

        self.assertTrue(QwenEditPipeline._has_flat_border_background(image))

    def test_flat_background_detector_rejects_blurred_scenery(self):
        height, width = 720, 560
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        # Smooth bokeh-like green scene: each tiny corner is locally smooth,
        # but the wider top strip changes substantially across the frame.
        base = np.asarray((18.0, 55.0, 12.0), dtype=np.float32)[None, None, :]
        variation = x * np.asarray((75.0, 120.0, 30.0))[None, None, :]
        vertical = y * np.asarray((15.0, 20.0, 8.0))[None, None, :]
        scene = np.broadcast_to(base + variation + vertical, (height, width, 3))
        image = Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8), "RGB")

        self.assertFalse(QwenEditPipeline._has_flat_border_background(image))

    def test_supplied_identity_face_box_bypasses_inconsistent_haar_crop(self):
        image = Image.new("RGB", (560, 720), (4, 126, 246))
        draw = ImageDraw.Draw(image)
        draw.ellipse((185, 75, 375, 305), fill=(48, 34, 28))
        draw.rectangle((55, 285, 505, 720), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                raise AssertionError("Haar must not run when the verified face box is supplied")

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(
                image,
                560,
                720,
                tight_school_id=True,
                delivery_upper_chest=True,
                face_bbox=(185, 75, 375, 305),
            )

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        face = (
            (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
            & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
            & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
        )
        rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
        # The verified detector box targets roughly 37% of frame height. The
        # synthetic ellipse occupies slightly less than that box, so assert a
        # conventional upper-chest scale without forcing an oversized head.
        self.assertGreater(float(rows[-1] - rows[0] + 1) / result.height, .30)
        self.assertLess(float(rows[-1] - rows[0] + 1) / result.height, .40)

    def test_preserved_frame_adds_headroom_when_crown_touches_top(self):
        image = Image.new("RGB", (560, 720), (4, 126, 246))
        draw = ImageDraw.Draw(image)
        draw.ellipse((175, 0, 385, 245), fill=(48, 34, 28))
        draw.rectangle((55, 235, 505, 720), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[195, 45, 170, 200]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(
                image, 560, 720, preserve_existing_frame=True,
            )

        rgb = np.asarray(result)
        crown_rows = np.flatnonzero(np.any(rgb != (4, 126, 246), axis=2).sum(axis=1) > 5)
        self.assertGreaterEqual(int(crown_rows[0]), 20)
        self.assertTrue(np.any(rgb[-1] == (72, 78, 86)))

    def test_broad_shoulders_do_not_force_a_half_body_crop(self):
        def portrait(body_left, body_right):
            image = Image.new("RGB", (496, 640), (128, 128, 128))
            draw = ImageDraw.Draw(image)
            draw.ellipse((174, 82, 322, 230), fill=(48, 34, 28))
            draw.rectangle((body_left, 215, body_right, 540), fill=(72, 78, 86))
            return image

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            narrow = QwenEditPipeline._crop_school_passport_portrait(
                portrait(150, 346), 560, 720,
            )
            broad = QwenEditPipeline._crop_school_passport_portrait(
                portrait(40, 456), 560, 720,
            )

        def face_height(result):
            rgb = np.asarray(result)
            mask = (
                (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
                & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
                & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
            )
            rows = np.flatnonzero(np.count_nonzero(mask, axis=1) > 5)
            return int(rows[-1] - rows[0] + 1)

        self.assertLessEqual(abs(face_height(narrow) - face_height(broad)), 5)

    def test_face_is_not_oversized_in_upper_chest_crop(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((174, 82, 322, 230), fill=(48, 34, 28))
        draw.rectangle((80, 230, 416, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        rgb = np.asarray(result)
        face = (
            (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
            & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
            & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
        )
        rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
        rendered_face_height = int(rows[-1] - rows[0] + 1)
        self.assertLess(rendered_face_height / result.height, .33)
        self.assertGreater(rendered_face_height / result.height, .24)

    def test_tight_school_id_crop_excludes_long_torso_and_arms(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((174, 82, 322, 230), fill=(48, 34, 28))
        draw.rectangle((80, 215, 416, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            regular = QwenEditPipeline._crop_school_passport_portrait(
                image,
                560,
                720,
                preserve_existing_frame=True,
            )
            result = QwenEditPipeline._crop_school_passport_portrait(
                image,
                560,
                720,
                preserve_existing_frame=True,
                tight_school_id=True,
            )

        def rendered_face_fraction(portrait):
            rgb = np.asarray(portrait)
            face = (
                (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
                & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
                & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
            )
            rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
            return float(rows[-1] - rows[0] + 1) / portrait.height

        self.assertGreater(
            rendered_face_fraction(result),
            rendered_face_fraction(regular) + .01,
        )
        self.assertLessEqual(rendered_face_fraction(result), .32)

    def test_delivery_crop_cannot_preserve_a_long_qwen_torso(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((174, 82, 322, 230), fill=(48, 34, 28))
        draw.rectangle((60, 215, 436, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            preserved = QwenEditPipeline._crop_school_passport_portrait(
                image, 560, 720, preserve_existing_frame=True, tight_school_id=True,
            )
            delivered = QwenEditPipeline._crop_school_passport_portrait(
                image,
                560,
                720,
                preserve_existing_frame=True,
                tight_school_id=True,
                delivery_upper_chest=True,
            )

        def face_fraction(portrait):
            rgb = np.asarray(portrait)
            face = (
                (rgb[:, :, 0] >= 35) & (rgb[:, :, 0] <= 65)
                & (rgb[:, :, 1] >= 20) & (rgb[:, :, 1] <= 48)
                & (rgb[:, :, 2] >= 16) & (rgb[:, :, 2] <= 42)
            )
            rows = np.flatnonzero(np.count_nonzero(face, axis=1) > 5)
            return float(rows[-1] - rows[0] + 1) / portrait.height

        self.assertGreater(face_fraction(delivered), face_fraction(preserved) + .04)

    def test_torso_touching_source_bottom_does_not_create_empty_lower_band(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((174, 82, 322, 230), fill=(48, 34, 28))
        draw.rectangle((80, 215, 416, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        rgb = np.asarray(result)
        lower_subject = np.linalg.norm(
            rgb[int(result.height * .92):].astype(np.float32) - 128.0, axis=2
        ) > 20.0
        self.assertGreater(float(np.mean(lower_subject)), .30)

    def test_already_framed_qwen_portrait_is_resized_without_zooming(self):
        image = Image.new("RGB", (496, 640), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((160, 42, 336, 248), fill=(48, 34, 28))
        # Mark the outer ear/hair edges. A crop or overscan loses these pixels.
        draw.rectangle((150, 125, 159, 175), fill=(12, 190, 35))
        draw.rectangle((337, 125, 346, 175), fill=(12, 190, 35))
        draw.rectangle((42, 230, 454, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(
                image, 560, 720, preserve_existing_frame=True,
            )

        rgb = np.asarray(result)
        foreground = np.linalg.norm(rgb.astype(np.float32) - 255.0, axis=2) > 20.0
        rows, cols = np.where(foreground)
        self.assertGreaterEqual(int(cols.min()), 0)
        self.assertLessEqual(int(cols.max()), result.width - 1)
        self.assertLess(abs(float(np.median(cols)) - result.width * .5), 12.0)
        self.assertEqual(int(rows.max()), result.height - 1)
        ear_markers = (
            (rgb[:, :, 0] < 40)
            & (rgb[:, :, 1] > 140)
            & (rgb[:, :, 2] < 70)
        )
        marker_columns = np.flatnonzero(np.count_nonzero(ear_markers, axis=0) > 3)
        self.assertLess(int(marker_columns.min()), int(result.width * .34))
        self.assertGreater(int(marker_columns.max()), int(result.width * .66))

    def test_existing_qwen_frame_does_not_recrop_when_crown_touches_top(self):
        image = Image.new("RGB", (496, 640), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.ellipse((150, -8, 346, 250), fill=(48, 34, 28))
        draw.rectangle((135, 105, 149, 185), fill=(20, 175, 45))
        draw.rectangle((347, 105, 361, 185), fill=(20, 175, 45))
        draw.rectangle((48, 235, 448, 640), fill=(72, 78, 86))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 70, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(
                image, 560, 720, preserve_existing_frame=True,
            )

        rgb = np.asarray(result)
        ear_markers = (
            (rgb[:, :, 0] < 50)
            & (rgb[:, :, 1] > 125)
            & (rgb[:, :, 2] < 80)
        )
        marker_columns = np.flatnonzero(np.count_nonzero(ear_markers, axis=0) > 3)
        self.assertLess(int(marker_columns.min()), int(result.width * .32))
        self.assertGreater(int(marker_columns.max()), int(result.width * .68))

    def test_crown_touching_source_edge_gets_passport_headroom(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((145, -25, 350, 330), fill=(25, 20, 18))
        draw.rectangle((90, 300, 410, 640), fill=(70, 75, 82))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[172, 36, 150, 150]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        foreground = np.linalg.norm(rgb.astype(np.float32) - 128.0, axis=2) > 20.0
        occupied = np.flatnonzero(np.count_nonzero(foreground, axis=1) > 10)
        self.assertGreaterEqual(int(occupied[0]), 24)
        self.assertLessEqual(int(occupied[0]), 60)

    def test_full_portrait_is_cropped_to_upper_chest(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((150, 25, 345, 270), fill=(45, 32, 28))
        draw.rectangle((95, 255, 400, 440), fill=(75, 80, 88))
        draw.rectangle((95, 440, 400, 640), fill=(210, 30, 30))

        class Detector:
            def detectMultiScale(self, *args, **kwargs):
                return np.asarray([[174, 82, 148, 148]])

        with patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=Detector()):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        rgb = np.asarray(result)
        lower_red = (rgb[int(result.height * .86):, 0] > 170) & (rgb[int(result.height * .86):, 1] < 80)
        self.assertLess(float(np.mean(lower_red)), 0.02)

    def test_detector_miss_never_resizes_the_full_person_frame(self):
        image = Image.new("RGB", (496, 640), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((150, 25, 345, 245), fill=(45, 32, 28))
        draw.rectangle((95, 225, 400, 410), fill=(75, 80, 88))
        draw.rectangle((95, 410, 400, 640), fill=(210, 30, 30))

        class HaarDetector:
            def detectMultiScale(self, *args, **kwargs):
                return np.empty((0, 4), dtype=np.int32)

        class IdentityDetector:
            def get(self, *args, **kwargs):
                return []

        with (
            patch("pipelines.qwen_edit_pipeline.cv2.CascadeClassifier", return_value=HaarDetector()),
            patch("pipelines.photo_restoration._get_insight_app", return_value=IdentityDetector()),
        ):
            result = QwenEditPipeline._crop_school_passport_portrait(image, 560, 720)

        self.assertEqual(result.size, (560, 720))
        rgb = np.asarray(result)
        lower_red = (
            (rgb[int(result.height * .86):, :, 0] > 170)
            & (rgb[int(result.height * .86):, :, 1] < 80)
        )
        self.assertLess(float(np.mean(lower_red)), 0.02)


if __name__ == "__main__":
    unittest.main()
