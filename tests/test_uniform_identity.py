import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from pipelines.uniform_review import check_uniform_identity


class UniformIdentityTests(unittest.TestCase):
    @staticmethod
    def face(embedding, geometry_offset=0.0):
        landmarks = np.stack((
            np.linspace(8.0, 24.0, 106),
            np.linspace(6.0, 26.0, 106),
        ), axis=1)
        landmarks[:, 1] += geometry_offset
        return SimpleNamespace(
            normed_embedding=np.asarray(embedding, dtype=float),
            landmark_2d_106=landmarks,
            kps=np.asarray(((10., 12.), (22., 12.), (16., 17.), (12., 22.), (20., 22.))),
        )

    def check(self, faces, **kwargs):
        detector = Mock()
        detector.get.side_effect = faces
        module = SimpleNamespace(_get_insight_app=lambda: detector)
        with patch.dict(sys.modules, {"pipelines.photo_restoration": module}):
            return check_uniform_identity(
                Image.new("RGB", (32, 32)),
                Image.new("RGB", (32, 32)),
                **kwargs,
            )

    def test_matching_face(self):
        face = self.face([1., 0.])
        self.assertTrue(self.check([[face], [face]])["accepted"])

    def test_high_similarity_regeneration_gets_narrow_geometry_tolerance(self):
        a = self.face([1., 0.])
        b = self.face([1., 0.])
        b.landmark_2d_106[45:75, 0] += 1.5
        result = self.check([[a], [b]])
        self.assertEqual(result["maximum_geometry_rmse"], 0.055)
        self.assertTrue(result["accepted"])

    def test_different_face(self):
        a = self.face([1., 0.])
        b = self.face([0., 1.])
        self.assertFalse(self.check([[a], [b]])["accepted"])

    def test_borderline_embedding_is_not_delivered_as_the_same_person(self):
        a = self.face([1., 0.])
        b = self.face([.69, np.sqrt(1.0 - .69 ** 2)])
        result = self.check([[a], [b]])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["minimum_similarity"], .70)

    def test_changed_face_geometry_is_rejected(self):
        a = self.face([1., 0.])
        b = self.face([1., 0.])
        b.landmark_2d_106[45:75, 0] += 4.0
        result = self.check([[a], [b]])
        self.assertFalse(result["accepted"])
        self.assertGreater(result["geometry_rmse"], result["maximum_geometry_rmse"])

    def test_uniform_delivery_can_require_stricter_geometry(self):
        a = self.face([1., 0.])
        b = self.face([.85, np.sqrt(1.0 - .85 ** 2)])
        b.landmark_2d_106[45:75, 0] += 1.0
        result = self.check([[a], [b]], maximum_geometry_rmse=.028)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["maximum_geometry_rmse"], .028)

    def test_exceptional_similarity_gets_bounded_geometry_tolerance(self):
        a = self.face([1., 0.])
        b = self.face([.95, np.sqrt(1.0 - .95 ** 2)])
        b.landmark_2d_106[45:75, 0] += 1.0
        result = self.check([[a], [b]], maximum_geometry_rmse=.030)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["requested_maximum_geometry_rmse"], .030)
        self.assertEqual(result["maximum_geometry_rmse"], .035)

    def test_identity_failure_reports_the_failed_gate(self):
        a = self.face([1., 0.])
        b = self.face([.69, np.sqrt(1.0 - .69 ** 2)])
        result = self.check([[a], [b]], maximum_geometry_rmse=.030)
        self.assertFalse(result["similarity_ok"])
        self.assertIn("similarity", result["reason"].lower())

    def test_missing_face(self):
        self.assertFalse(self.check([[], []])["accepted"])

    def test_unavailable_verification(self):
        self.assertFalse(self.check(RuntimeError("unavailable"))["accepted"])
