"""Source-crop arm guard: AST checks, no GPU models loaded."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArmGuideSourceEvidenceTests(unittest.TestCase):
    """job 16682cec delivered a duplicated shoulder: a 90px width step over
    ten rows, with two sleeve hems and two red cuffs per side. The upload was
    a head-and-chest crop containing no arm at all (visible-arm fraction
    0.0000), but a short-sleeved target alone enabled both the synthetic skin
    cue (51686 pixels) and the "generate the arm sections" prompt clause.
    Qwen resolved that unexplained skin band as a second sleeved arm."""

    def setUp(self):
        self.source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.route = next(
            node for node in ast.walk(ast.parse(self.source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_vl_image_edit_uniform_swap"
        )

    def _assignment(self, name):
        return next(
            node for node in ast.walk(self.route)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )

    def _guards(self, name):
        """Every `if` whose body assigns `name`, as source text."""
        found = []
        for node in ast.walk(self.route):
            if not isinstance(node, ast.If):
                continue
            for statement in node.body:
                assigns = [
                    target.id for item in ast.walk(statement)
                    if isinstance(item, ast.Assign)
                    for target in item.targets
                    if isinstance(target, ast.Name)
                ]
                if name in assigns:
                    found.append(ast.get_source_segment(self.source, node.test))
        return found

    def test_arm_evidence_is_measured_from_the_source_parser(self):
        evidence = self._assignment("source_arm_evidence")
        text = ast.get_source_segment(self.source, evidence)
        self.assertIn("source_visible_arm_fraction", text)

    def test_skin_cue_requires_a_visible_source_arm(self):
        guards = self._guards("synthetic_arm_guide")
        gated = [g for g in guards if "source_arm_evidence" in g]
        self.assertTrue(
            gated,
            "the synthetic skin cue must be gated on measured source arm evidence",
        )
        self.assertTrue(any("not source_arm_evidence" not in g for g in gated))

    def test_arm_prompt_has_a_no_arm_branch_for_cropped_uploads(self):
        guards = self._guards("arm_generation_instruction")
        self.assertTrue(
            any("source_arm_evidence" in guard for guard in guards),
            "the arm prompt clause must branch on measured source arm evidence",
        )

    def test_cropped_upload_prompt_forbids_a_second_sleeve(self):
        clause = next(
            node for node in ast.walk(self.route)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "arm_generation_instruction"
                for t in node.targets
            )
            and "no reliable arm pixels" in (ast.get_source_segment(self.source, node) or "")
        )
        text = ast.get_source_segment(self.source, clause)
        for phrase in ("second sleeve", "second cuff", "extra garment layer"):
            self.assertIn(phrase, text)

    def test_no_arm_crop_does_not_outpaint_synthetic_limbs(self):
        route_text = ast.get_source_segment(self.source, self.route)
        self.assertIn("upper_arm_outpaint_region", route_text)
        self.assertIn("requires_exposed_arm_synthesis and not source_arm_evidence", route_text)
        self.assertIn("kept backdrop", route_text)
        self.assertIn("Do not invent arms or widen the body", route_text)

    def test_measured_arm_cue_reaches_qwen_primary_and_edit_mask(self):
        route_text = ast.get_source_segment(self.source, self.route)
        self.assertIn("if np.any(synthetic_arm_guide) and measured_skin_rgb", route_text)
        self.assertIn("primary_person_source = Image.fromarray", route_text)
        self.assertIn("edit_region |= arm_edit_corridor", route_text)
        cue_at = route_text.index("if np.any(synthetic_arm_guide) and measured_skin_rgb")
        qwen_at = route_text.index("generation_input = generation_primary.resize")
        self.assertLess(cue_at, qwen_at)

    def test_arm_instruction_survives_prompt_budget_trimming(self):
        route_text = ast.get_source_segment(self.source, self.route)
        budget_at = route_text.index("generated_prompt = fit_prompt_to_token_budget")
        retained_tail = route_text[budget_at:budget_at + 600]
        self.assertIn("retained_arm_generation_instruction", retained_tail)

    def test_evidence_threshold_is_below_the_synthesis_trigger(self):
        """A source with *some* arm must still get the cue: the evidence floor
        has to sit under the 0.012 fraction that turns synthesis on."""
        evidence = ast.get_source_segment(self.source, self._assignment("source_arm_evidence"))
        floor = float(evidence.split(">=")[1].strip())
        self.assertGreater(floor, 0.0)
        self.assertLess(floor, 0.012)


if __name__ == "__main__":
    unittest.main()
