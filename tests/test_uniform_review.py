import unittest

from pipelines.uniform_review import CHECKS, validate_review, pose_offset


class UniformReviewTests(unittest.TestCase):
    def test_pose_is_scale_invariant(self):
        self.assertAlmostEqual(pose_offset([[10, 10], [30, 10], [22, 20]]),
                               pose_offset([[20, 20], [60, 20], [44, 40]]))

    def test_profile_differs_from_frontal(self):
        front = pose_offset([[10, 10], [30, 10], [20, 20]])
        profile = pose_offset([[10, 10], [30, 10], [32, 20]])
        self.assertGreater(abs(front - profile), 0.18)

    def test_badge_is_a_failure_despite_other_passes(self):
        data = {key: False for key in CHECKS}
        data.update(badge_present=True, correction="Remove chest badge")
        self.assertEqual(validate_review(data)["issues"], ["badge_present"])

    def test_all_explicit_checks_required(self):
        with self.assertRaises(ValueError):
            validate_review({"correction": "Looks fine"})

    def test_string_false_is_not_accepted(self):
        data = {key: False for key in CHECKS}
        data.update(badge_present="false", correction="")
        with self.assertRaises(ValueError):
            validate_review(data)

    def test_clear_candidate(self):
        data = {key: False for key in CHECKS}
        data["correction"] = ""
        self.assertEqual(validate_review(data)["issues"], [])

    def test_structured_correction(self):
        data = {key: False for key in CHECKS}
        data.update(badge_present=True, correction=["Remove the chest badge"])
        self.assertIn("Remove the chest badge", validate_review(data)["correction"])


if __name__ == "__main__":
    unittest.main()
