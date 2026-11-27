"""V13.3 Unit Tests for Core Architectural Components:
1. UniformReferenceAnalyzer (15 visual attributes + SHA256 caching)
2. UniformFidelityValidator (Region-aware reference-to-candidate fidelity)
3. RecoveryController (Targeted defect-aware repairs)
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from pipelines.uniform_swap.reference_analyzer import reference_analyzer
from pipelines.uniform_swap.fidelity_validator import uniform_fidelity_validator
from pipelines.uniform_swap.recovery_controller import recovery_controller
from pipelines.uniform_swap.config import UniformSwapConfig


class TestV133Components(unittest.TestCase):
    """Test V13.3 component upgrades and defect recovery."""

    def test_01_reference_analyzer_all_canonical_uniforms(self):
        """Verify dynamic attribute extraction across all 4 canonical uniform references."""
        refs = {
            "REF001_red_collar.png": {"ref_type": "red_collar", "collar_color": "red", "tie": False, "vest": False},
            "REF002_navy_collar.png": {"ref_type": "navy_collar", "collar_color": "navy blue", "tie": False, "vest": False},
            "REF003_tie.png": {"ref_type": "tie", "collar_color": "white", "tie": True, "vest": False},
            "REF004_vest.png": {"ref_type": "vest", "tie": False, "vest": True},
        }

        for fname, expected in refs.items():
            p = Path("test_dataset/uniform_references") / fname
            self.assertTrue(p.exists(), f"Reference {p} must exist")
            spec = reference_analyzer.analyze(p, force_reanalyze=True)
            self.assertEqual(spec["ref_type"], expected["ref_type"], f"{fname} ref_type mismatch")
            if "collar_color" in expected:
                self.assertEqual(spec["collar_color"], expected["collar_color"], f"{fname} collar_color mismatch")
            self.assertEqual(spec["tie"], expected["tie"], f"{fname} tie presence mismatch")
            self.assertEqual(spec["vest"], expected["vest"], f"{fname} vest presence mismatch")
            self.assertTrue(len(spec["sha256"]) == 64, "SHA-256 hash must be 64 hex characters")

    def test_02_fidelity_validator_detects_mismatches(self):
        """Verify UniformFidelityValidator correctly validates matches and catches defects."""
        h, w = 1024, 800
        cand_rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # Plain white shirt
        garment_mask = np.zeros((h, w), dtype=bool)
        garment_mask[450:950, 150:650] = True

        # Test against vest spec: Plain white shirt should FAIL vest check
        vest_spec = {
            "ref_type": "vest",
            "collar_color": "light-blue check",
            "has_vest": True,
            "has_tie": False,
        }
        res_vest = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=cand_rgb,
            garment_mask=garment_mask,
            reference_spec=vest_spec,
            face_bbox=(250, 150, 550, 450),
        )
        self.assertFalse(res_vest["passed"])
        self.assertTrue(any("missing_vest" in d for d in res_vest["defects"]))

        # Test against red collar spec: Plain white collar should FAIL collar color check
        red_spec = {
            "ref_type": "red_collar",
            "collar_color": "red",
            "has_red_collar": True,
            "has_tie": False,
            "has_vest": False,
        }
        res_red = uniform_fidelity_validator.validate_fidelity(
            candidate_rgb=cand_rgb,
            garment_mask=garment_mask,
            reference_spec=red_spec,
            face_bbox=(250, 150, 550, 450),
        )
        self.assertFalse(res_red["passed"])
        self.assertTrue(any("incorrect_collar_color" in d for d in res_red["defects"]))

    def test_03_recovery_controller_shoulder_repair(self):
        """Verify RecoveryController expands narrowed shoulders without re-running diffusion."""
        h, w = 1024, 800
        rgb = np.full((h, w, 3), 100, dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)
        # Narrow shoulder alpha (stops at 30% and 70%)
        alpha[450:950, int(w * 0.32):int(w * 0.68)] = 255

        id_layer = {
            "metrics": {"chin_y": 450, "face_height": 300},
        }
        rgb_rep, alpha_rep, acts = recovery_controller._repair_incomplete_shoulders(
            rgb, alpha, id_layer, w, h
        )
        # Alpha should now extend into left margin <= 0.22*w and right margin >= 0.78*w
        self.assertTrue(np.any(alpha_rep[600:900, :int(w * 0.22)] > 0))
        self.assertTrue(np.any(alpha_rep[600:900, int(w * 0.78):] > 0))
        self.assertTrue(len(acts) > 0)


if __name__ == "__main__":
    unittest.main()
