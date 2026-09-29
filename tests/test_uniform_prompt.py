"""Lightweight prompt regression tests without loading GPU models."""
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UniformPromptTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        self.tree = tree
        self.route = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == "qwen_vl_image_edit_uniform_swap")

    def uniform_step_selector(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_select_uniform_steps")
        namespace = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "steps", "exec"), namespace)
        return namespace["_select_uniform_steps"]

    def test_adaptive_uniform_steps_scale_with_observed_risk(self):
        select = self.uniform_step_selector()
        simple, _ = select({
            "pose": "front portrait", "hair_accessories": "none",
            "outer_color_confidence": "high", "fabric_detail": "plain woven cotton",
        })
        moderate, _ = select({
            "pose": "front portrait", "hair_accessories": "two clips",
            "outer_color_confidence": "high", "fabric_detail": "plain woven cotton",
        })
        difficult, reasons = select({
            "pose": "front portrait", "hair_accessories": "two flowers",
            "direct_sunlight_present": True, "sunlight_type": "hard direct sun",
            "outer_color_confidence": "low",
            "fabric_detail": "unclear",
        })
        self.assertEqual((simple, moderate, difficult), (8, 12, 20))
        self.assertIn("hard or direct source lighting", reasons)

    def test_soft_indoor_hotspot_and_shared_template_uncertainty_use_twelve_steps(self):
        select = self.uniform_step_selector()
        selected, reasons = select({
            "pose": "front portrait", "hair_accessories": "none",
            "direct_sunlight_present": True, "head_hair_hotspot_present": True,
            "sunlight_type": "soft indoor", "outer_color_confidence": "low",
            "fabric_detail": "unclear",
        })
        self.assertEqual(selected, 12)
        self.assertEqual(reasons, [
            "possible lighting hotspot", "uncertain uniform color or fabric detail",
        ])

    def test_adaptive_uniform_steps_respect_requested_maximum(self):
        select = self.uniform_step_selector()
        selected, _ = select({
            "direct_sunlight_present": True, "hair_accessories": "flowers",
        }, maximum_steps=12)
        self.assertEqual(selected, 12)

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
                         lighting_instruction="TEST LIGHTING",
                         skin_tone_instruction="TEST SKIN",
                         identity_instruction="TEST IDENTITY",
                         collar_constraint="TEST COLLAR",
                         refinement_prompt=lambda bg, fabric, hair, lighting, skin, identity: "TEST REFINEMENT " + str(bg) + " " + fabric + " " + hair + " " + lighting + " " + skin + " " + identity)
        result = eval(compile(expression, "prompt", "eval"), namespace)
        self.assertEqual(result, "TEST REFINEMENT (4, 126, 246) TEST BLACK TEST HAIR TEST LIGHTING TEST SKIN TEST IDENTITY TEST COLLAR")

    def test_vl_skin_tone_is_forwarded_as_a_source_cross_check(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('vl_plan.get("skin_tone")', source)
        self.assertIn("image 1 pixels remain authoritative", source)
        for phrase in ("never lighten", "whiten, tan, warm, cool or recolor skin"):
            self.assertIn(phrase, source)
        self.assertIn('"skin_tone_instruction": skin_tone_instruction', source)

    def test_negative_prompt_does_not_ban_source_traits(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "negative_prompt" for t in n.targets))
        result = ast.literal_eval(assignment.value)
        for trait in ("straight hair", "gray hair", "white dress", "neck shadow"):
            self.assertNotIn(trait, result)
        for source_trait in ("added bows", "added flowers"):
            self.assertNotIn(source_trait, result)

    def test_uniform_construction_observations_are_forwarded(self):
        assignment = next(n for n in self.route.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "garment_details" for t in n.targets))
        keys = ast.literal_eval(assignment.value.args[0])
        for key in ("shirt_color_and_pattern", "outer_garment_color_and_shape"):
            self.assertIn(key, keys)
        for key in ("outer_color_under_neutral_light",):
            self.assertIn(key, keys)

    def test_vl_portrait_details_are_audited_not_prompted(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn('f"VL portrait facts: {portrait_details}', source)
        self.assertIn('"portrait_details": portrait_details', source)
        for key in ("face_detail", "hair_parting", "hair_texture", "hair_color",
                    "hair_accessories", "hair_accessory_details", "visible_wearables"):
            self.assertIn(f'"{key}"', source)
        self.assertIn("outer_neutral_rgb", source)

    def test_identity_review_blocks_mismatched_person(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("[UNIFORM_IDENTITY_REVIEW] Candidate rejected", source)
        self.assertIn("Uniform output was not delivered because the generated face", source)
        self.assertNotIn("Candidate retained for user review", source)

    def test_uniform_delivery_uses_no_secondary_identity_generator(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("_swap_uniform_identity", source)
        self.assertNotIn("inswapper_128", source)

    def test_identity_repair_is_structural_and_preserves_source_chroma(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("alpha = (mask * .36)", source)
        self.assertIn("mean_strength = 1.0 if channel == 0 else 0.35", source)

    def test_uniform_prompt_rejects_cast_only_skin_and_invented_earrings(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('unsafe_cast_terms = ("orange", "yellow"', source)
        for phrase in ("changed ethnicity appearance", "changed ancestry appearance", "invented earrings", "dangling earrings"):
            self.assertIn(phrase, source)

    def test_generation_uses_structured_accessories_not_unreliable_free_text(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn('raw_accessories = vl_plan.get("hair_accessories")', source)
        generation_section = source[source.index("raw_accessories ="):source.index("generated_prompt =")]
        self.assertNotIn("hair_accessory_details", generation_section)

    def test_single_generation_keeps_quality_settings(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "qwen_edit_enhancer"]
        self.assertEqual(len(calls), 1)
        kwargs = {arg.arg: arg.value for arg in calls[0].keywords}
        self.assertEqual(kwargs["steps"].id, "generation_steps")
        self.assertEqual(ast.literal_eval(kwargs["max_sequence_length"]), 384)
        self.assertEqual(ast.literal_eval(kwargs["true_cfg_scale"]), 3.0)
        self.assertFalse(ast.literal_eval(kwargs["reject_identity_failure"]))
        self.assertTrue(ast.literal_eval(kwargs["return_raw_candidate"]))
        self.assertEqual(ast.literal_eval(kwargs["conditioning_max_dimension"]), 384)
        self.assertEqual(eval(compile(ast.Expression(kwargs["conditioning_pixel_budget"]), "budget", "eval")), 384 * 384)
        self.assertTrue(ast.literal_eval(kwargs["preserve_reference_aspect"]))
        references = kwargs["reference_images"]
        self.assertIsInstance(references, ast.List)
        self.assertEqual([item.id for item in references.elts], ["person_pil", "template_pil"])
        self.assertEqual(kwargs["image_input"].id, "person_pil")

    def test_repeated_inputs_use_a_new_auditable_job_seed(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn('zlib.crc32(job_id.encode("utf-8"), uniform_seed)', source)
        self.assertIn('"seed_policy": "stable_content_seed"', source)

    def test_single_pass_edits_original_person_and_uses_template_reference(self):
        assignments = {
            target.id: node.value
            for node in self.route.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        normalized = assignments["conditioning_person"]
        self.assertIsInstance(normalized, ast.Call)
        self.assertEqual(normalized.func.attr, "normalize_uniform_conditioning_light")
        self.assertEqual(normalized.args[0].id, "person_pil")

        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        qwen = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "qwen_edit_enhancer")
        qwen_kwargs = {item.arg: item.value for item in qwen.keywords}
        self.assertEqual(qwen_kwargs["image_input"].id, "person_pil")
        identity = next(node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "check_uniform_identity")
        self.assertEqual(identity.args[0].id, "person_pil")

    def test_wrong_qwen_background_is_corrected_after_generation(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        qwen = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "qwen_edit_enhancer")
        post_generation_replacements = [node for node in calls if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {
                "_replace_background_with_foreground_matte",
                "_replace_smooth_border_background",
            }
            and node.lineno > qwen.lineno
        )]
        self.assertEqual(len(post_generation_replacements), 1)
        replacement = post_generation_replacements[0]
        kwargs = {item.arg: item.value for item in replacement.keywords}
        self.assertEqual(replacement.args[0].id, "cropped")
        self.assertEqual(kwargs["background_color"].id, "background_color")
        self.assertTrue(ast.literal_eval(kwargs["preserve_foreground_rgb"]))
        self.assertTrue(ast.literal_eval(kwargs["strict"]))
        background = next(node for node in calls if isinstance(node.func, ast.Attribute)
                          and node.func.attr == "_has_selected_solid_background")
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        self.assertGreater(finishing.lineno, background.lineno)

    def test_uniform_finishing_uses_source_exposure_and_measured_hair(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        kwargs = {item.arg: item.value for item in finishing.keywords}
        self.assertEqual(kwargs["correct_dark_hair"].id, "measured_dark_hair")
        self.assertEqual(kwargs["face_target_luma"].id, "face_target_luma")

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
