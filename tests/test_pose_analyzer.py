"""Unit test for pose analyzer, studio reposing, and lighting balancing."""

import os
import json
import shutil
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from pipelines.uniform_swap.pose_analyzer import (
    analyze_portrait_pose,
    apply_studio_reposing,
    apply_studio_lighting,
)


class TestPoseAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("outputs/test_pose_artifacts")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # Use an existing canonical test image if present, or create a mock face
        src_path = Path("T001_source.png")
        if not src_path.exists():
            src_path = Path("inputs/T001_source.png")
        if not src_path.exists():
            # Search workspace
            found = list(Path(".").glob("**/T001*.png"))
            if found:
                src_path = found[0]

        if src_path.exists():
            self.src_img = Image.open(src_path).convert("RGB")
        else:
            # Fallback synthetic portrait
            arr = np.zeros((1024, 800, 3), dtype=np.uint8)
            arr[200:800, 200:600] = (200, 160, 140)
            self.src_img = Image.fromarray(arr)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pose_analysis_and_reposing_artifacts(self):
        pose = analyze_portrait_pose(self.src_img)
        self.assertIn("head_roll", pose)
        self.assertIn("route", pose)
        self.assertIsInstance(pose["kps"], list)
        if pose["kps"]:
            self.assertIsInstance(pose["kps"][0][0], float)

        w, h = self.src_img.size
        # Create dummy identity layer
        id_layer = {
            "full_person_alpha": np.ones((h, w), dtype=np.uint8) * 255,
            "original_face_mask": np.zeros((h, w), dtype=bool),
            "original_hair_mask": np.zeros((h, w), dtype=bool),
            "original_neck_mask": np.zeros((h, w), dtype=bool),
            "original_accessory_mask": np.zeros((h, w), dtype=bool),
        }
        # Mark simple center as face/neck
        id_layer["original_face_mask"][h//4:h//2, w//3:2*w//3] = True
        id_layer["original_neck_mask"][h//2:h//2+100, w//3:2*w//3] = True

        # Test apply_studio_reposing
        reposed_img, new_id_layer, pose_transform = apply_studio_reposing(
            self.src_img,
            id_layer,
            pose,
            debug_dir=self.test_dir,
            job_id="test_job",
        )

        self.assertIsNotNone(reposed_img)
        self.assertIsNotNone(pose_transform)

        # Check that the 9 artifacts exist in debug_dir
        expected_artifacts = [
            "pose_corrected_source.png",
            "corrected_face_rgba.png",
            "corrected_neck_rgba.png",
            "corrected_hair_rgba.png",
            "corrected_accessories_rgba.png",
            "corrected_person_rgba.png",
            "pose_analysis_before.json",
            "pose_analysis_after.json",
            "pose_transform.json",
        ]
        for art in expected_artifacts:
            art_path = self.test_dir / art
            self.assertTrue(art_path.exists(), f"Missing artifact: {art}")

        # Check JSON validity of artifacts
        for j_art in ["pose_analysis_before.json", "pose_analysis_after.json", "pose_transform.json"]:
            with open(self.test_dir / j_art, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)

        # Test apply_studio_lighting
        lit_img = apply_studio_lighting(
            reposed_img,
            face_mask=new_id_layer["original_face_mask"],
            neck_mask=new_id_layer["original_neck_mask"],
        )
        self.assertEqual(lit_img.size, self.src_img.size)


if __name__ == "__main__":
    unittest.main()
