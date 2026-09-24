"""Lightweight prompt regression tests without loading GPU models."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UniformPromptTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        self.route = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == "qwen_vl_image_edit_uniform_swap")

    def details(self, plan, keys):
        helper = next(n for n in self.route.body if isinstance(n, ast.FunctionDef)
                      and n.name == "observed_details")
        namespace = {"vl_plan": plan}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "helper", "exec"), namespace)
        return namespace["observed_details"](keys)

    def test_observations_include_fabric_and_face(self):
        plan = {"source": "qwen2.5-vl", "fabric_detail": "fine woven charcoal",
                "face_detail": "natural uneven eyebrows", "hair_accessory_details": ["left white clip", "right white clip"]}
        result = self.details(plan, tuple(plan))
        self.assertIn("fine woven charcoal", result)
        self.assertIn("natural uneven eyebrows", result)
        self.assertIn("left white clip, right white clip", result)

    def test_fallback_is_not_template_evidence(self):
        result = self.details({"source": "geometry fallback", "shirt_collar": "mandarin"}, ("shirt_collar",))
        self.assertNotIn("mandarin", result)

    def test_details_reach_generation(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "generated_prompt" for t in n.targets))
        expression = ast.Expression(assignment.value)
        namespace = dict(bg_rgb=(4, 126, 246), fabric_color_instruction="TEST BLACK",
                         hair_color_instruction="TEST HAIR",
                         refinement_prompt=lambda bg, fabric, hair: "TEST REFINEMENT " + str(bg) + " " + fabric + " " + hair)
        result = eval(compile(expression, "prompt", "eval"), namespace)
        self.assertEqual(result, "TEST REFINEMENT (4, 126, 246) TEST BLACK TEST HAIR")

    def test_negative_prompt_does_not_ban_source_traits(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "negative_prompt" for t in n.targets))
        result = ast.literal_eval(assignment.value)
        for trait in ("straight hair", "gray hair", "white dress", "neck shadow"):
            self.assertNotIn(trait, result)

    def test_uniform_construction_observations_are_forwarded(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "garment_details" for t in n.targets))
        keys = ast.literal_eval(assignment.value.args[0])
        for key in ("shirt_color_and_pattern", "outer_garment_color_and_shape",
                    "fabric_detail", "construction_detail", "button_color"):
            self.assertIn(key, keys)

    def test_single_generation_keeps_quality_settings(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "qwen_edit_enhancer"]
        self.assertEqual(len(calls), 1)
        kwargs = {arg.arg: arg.value for arg in calls[0].keywords}
        self.assertEqual(ast.literal_eval(kwargs["steps"]), 20)
        self.assertEqual(ast.literal_eval(kwargs["true_cfg_scale"]), 3.0)
        self.assertFalse(ast.literal_eval(kwargs["reject_identity_failure"]))
        self.assertTrue(ast.literal_eval(kwargs["return_raw_candidate"]))
        self.assertEqual(ast.literal_eval(kwargs["conditioning_max_dimension"]), 384)
        self.assertEqual(eval(compile(ast.Expression(kwargs["conditioning_pixel_budget"]), "budget", "eval")), 384 * 384)
        self.assertTrue(ast.literal_eval(kwargs["preserve_reference_aspect"]))
        references = kwargs["reference_images"]
        self.assertIsInstance(references, ast.List)
        self.assertEqual([item.id for item in references.elts], ["template_pil"])
        self.assertEqual(kwargs["image_input"].id, "rough_composite")

    def test_fabric_finishing_runs_after_background_replacement(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        background = next(node for node in calls if isinstance(node.func, ast.Attribute)
                          and node.func.attr == "_replace_background_with_foreground_matte")
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        self.assertGreater(finishing.lineno, background.lineno)

    def test_custom_prompt_cannot_replace_uniform_instructions(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "generated_prompt" for t in n.targets))
        self.assertIsInstance(assignment.value, ast.JoinedStr)

    def test_raw_candidate_returns_before_identity_fallback(self):
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        enhancer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == "qwen_edit_enhancer")
        branch = next(n for n in enhancer.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == "return_raw_candidate")
        returns = [n for n in branch.body if isinstance(n, ast.Return)]
        self.assertEqual(returns[0].value.id, "qwen_debug_path")
        checks = [n for n in ast.walk(enhancer) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "_has_acceptable_portrait_identity"]
        self.assertLess(branch.lineno, checks[0].lineno)
        defaults = dict(zip([a.arg for a in enhancer.args.args][-len(enhancer.args.defaults):], enhancer.args.defaults))
        self.assertFalse(ast.literal_eval(defaults["return_raw_candidate"]))
        self.assertEqual(ast.literal_eval(defaults["conditioning_max_dimension"]), 256)
        self.assertFalse(ast.literal_eval(defaults["preserve_reference_aspect"]))
        self.assertIsNone(ast.literal_eval(defaults["conditioning_pixel_budget"]))

    def test_both_encoders_receive_same_visual_budget(self):
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_encode_prompt_isolated"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            self.assertEqual(keywords["conditioning_pixel_budget"].id, "conditioning_pixel_budget")

    def test_template_retains_garment_aspect_and_pixels(self):
        from PIL import Image
        from typing import Optional, List
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_prepare_uniform_reference")
        helper.decorator_list = []
        namespace = dict(Image=Image, Optional=Optional, List=List)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "reference", "exec"), namespace)
        template = Image.new("RGBA", (200, 300), (0, 0, 0, 0))
        template.paste((25, 35, 45, 255), (0, 200, 200, 300))
        prepared = namespace[helper.name](template, pad_to_portrait=False)
        self.assertEqual(prepared.size, (200, 108))
        self.assertEqual(prepared.getpixel((0, 107)), (25, 35, 45))
        self.assertEqual(prepared.getpixel((199, 107)), (25, 35, 45))

    def test_nested_uniform_observations_keep_layer_context(self):
        from typing import Any, Dict
        tree = ast.parse((ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8"))
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("flatten_vl_dict", "normalize_uniform_analysis")]
        namespace = dict(Any=Any, Dict=Dict)
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "analysis", "exec"), namespace)
        result = namespace["normalize_uniform_analysis"]({"uniform": {
            "shirt_color_and_pattern": {"color": "white", "pattern": "fine blue checks"},
            "outer_garment_color_and_shape": {"color": "charcoal", "shape": "angled panels"},
            "button_color": {"shirt": "white", "vest": "white"},
            "shirt_collar": "unknown"}})
        self.assertIn("fine blue checks", result["shirt_color_and_pattern"])
        self.assertNotIn("charcoal", result["shirt_color_and_pattern"])
        self.assertIn("angled panels", result["outer_garment_color_and_shape"])
        self.assertIn("vest: white", result["button_color"])
        self.assertEqual(result["shirt_collar"], "unknown")

    def test_isolated_encoder_disables_autograd(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("with torch.inference_mode():\n    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(", source)


if __name__ == "__main__":
    unittest.main()
