import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from pipelines.uniform_review import check_uniform_identity


class UniformIdentityTests(unittest.TestCase):
    def check(self, faces):
        detector = Mock()
        detector.get.side_effect = faces
        module = SimpleNamespace(_get_insight_app=lambda: detector)
        with patch.dict(sys.modules, {"pipelines.photo_restoration": module}):
            return check_uniform_identity(Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32)))

    def test_matching_face(self):
        face = SimpleNamespace(normed_embedding=np.array([1., 0.]))
        self.assertTrue(self.check([[face], [face]])["accepted"])

    def test_different_face(self):
        a = SimpleNamespace(normed_embedding=np.array([1., 0.]))
        b = SimpleNamespace(normed_embedding=np.array([0., 1.]))
        self.assertFalse(self.check([[a], [b]])["accepted"])

    def test_missing_face(self):
        self.assertFalse(self.check([[], []])["accepted"])

    def test_unavailable_verification(self):
        self.assertFalse(self.check(RuntimeError("unavailable"))["accepted"])
