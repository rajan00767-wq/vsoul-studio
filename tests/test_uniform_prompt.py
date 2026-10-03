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

    def enhancement_step_selector(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_select_enhancement_steps")
        namespace = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "steps", "exec"), namespace)
        return namespace["_select_enhancement_steps"]

    def test_adaptive_enhancement_steps_scale_with_observed_risk(self):
        select = self.enhancement_step_selector()
        simple, _ = select({"face_orientation": "front", "hair_accessories": "none"})
        minor, _ = select({
            "face_orientation": "front", "head_hair_hotspot_present": True,
            "sunlight_type": "soft indoor", "hair_accessories": "none",
        })
        moderate, _ = select({
            "face_orientation": "front", "hair_accessories": "two clips",
            "hair_edge_risk": "high",
        })
        difficult, reasons = select({
            "face_orientation": "three-quarter", "hair_accessories": "flowers",
            "direct_sunlight_present": True, "sunlight_type": "hard direct sun",
        })
        self.assertEqual((simple, minor, moderate, difficult), (4, 8, 12, 20))
        self.assertIn("hard or direct source lighting", reasons)

    def test_adaptive_enhancement_steps_respect_requested_maximum(self):
        selected, _ = self.enhancement_step_selector()({
            "face_orientation": "profile", "direct_sunlight_present": True,
            "sunlight_type": "hard direct sun", "hair_accessories": "flowers",
        }, maximum_steps=12)
        self.assertEqual(selected, 12)

    def test_uniform_low_detail_does_not_force_aggressive_regeneration(self):
        selected, reasons = self.uniform_step_selector()({
            "pose": "front", "hair_accessories": "two clips",
            "outer_color_confidence": "uncertain", "fabric_detail": "unclear",
            "source_min_dimension": 390, "source_face_detail_score": 65.3,
        })
        self.assertEqual(selected, 12)
        self.assertNotIn("limited source resolution or facial detail", reasons)

    def test_uniform_route_measures_source_detail_before_step_selection(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        measurement = source.index('vl_plan["source_min_dimension"] = min(source_w, source_h)')
        selection = source.index("generation_steps, step_reasons = _select_uniform_steps")
        self.assertLess(measurement, selection)
        self.assertIn('vl_plan["source_face_detail_score"]', source)

    def test_low_resolution_and_soft_face_limit_regeneration_to_eight_steps(self):
        selected, reasons = self.enhancement_step_selector()({
            "face_orientation": "front", "hair_accessories": "none",
            "source_min_dimension": 390, "source_face_detail_score": 65.3,
        })
        self.assertEqual(selected, 8)
        self.assertIn("limited source resolution or facial detail", reasons)

    def test_enhancement_prompt_requires_photographic_texture(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        for phrase in (
            "photographic restoration, not face recreation",
            "natural pores",
            "small facial asymmetries",
            "exact source face shape, hairline, hair silhouette",
            "airbrushed skin, porcelain skin",
            "Preserve every visible source hair clip",
            "small pale accessories near the crown",
        ):
            self.assertIn(phrase, source)
        self.assertIn("true_cfg_scale=1.02 if generation_steps <= 4 else 2.0", source)
        self.assertIn("do not recreate the person", source)

    def test_portrait_vl_uses_focused_hair_analysis(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn("hair_prompt = {self._HAIR_RETRY_PROMPT!r}", source)
        self.assertIn("portrait.height * 0.72", source)
        self.assertIn('"hair_raw": hair_response', source)
        self.assertIn('result["hair_observations_raw"] = hair_parsed', source)

    def test_enhancement_identity_floor_rejects_borderline_regenerations(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("MIN_PORTRAIT_IDENTITY_SIMILARITY = 0.70", source)
        self.assertIn("MAX_PORTRAIT_LANDMARK_RMSE = 0.045", source)
        self.assertIn("landmark_2d_106", source)
        self.assertIn("and geometry_ok", source)

    def test_adaptive_uniform_steps_scale_with_observed_risk(self):
        select = self.uniform_step_selector()
        simple, _ = select({
            "pose": "front portrait", "hair_accessories": "none",
            "outer_color_confidence": "high", "fabric_detail": "plain woven cotton",
        })
        minor, _ = select({
            "pose": "front portrait", "hair_accessories": "none",
            "head_hair_hotspot_present": True, "sunlight_type": "soft indoor",
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
        self.assertEqual((simple, minor, moderate, difficult), (4, 8, 12, 12))
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
        namespace = dict(qwen_bg_rgb=(128, 128, 128), fabric_color_instruction="TEST BLACK",
                         hair_color_instruction="TEST HAIR",
                         lighting_instruction="TEST LIGHTING",
                         skin_tone_instruction="TEST SKIN",
                         identity_instruction="TEST IDENTITY",
                         collar_constraint="TEST COLLAR",
                         template_generation_details="TEST GARMENT FACTS",
                         neckwear_instruction="TEST NECKWEAR",
                         refinement_prompt=lambda bg, fabric, hair, lighting, skin, identity, **kwargs: "TEST REFINEMENT " + str(bg) + " " + fabric + " " + hair + " " + lighting + " " + skin + " " + identity)
        result = eval(compile(expression, "prompt", "eval"), namespace)
        self.assertEqual(result, "TEST REFINEMENT (128, 128, 128) TEST BLACK TEST HAIR TEST LIGHTING TEST SKIN TEST IDENTITY TEST COLLAR Do not use any text description to infer hair style, clip type, colour, count or position; image 1 pixels are authoritative. Stable template fabric facts (must match image 2 pixels): TEST GARMENT FACTS. TEST NECKWEAR")

    def test_vl_skin_tone_is_forwarded_as_a_source_cross_check(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('vl_plan.get("skin_tone")', source)
        self.assertIn("image 2 pixels remain authoritative", source)
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

    def test_template_pixels_override_vl_for_collar_and_neckline_geometry(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("outer-garment neckline silhouette exactly", source)
        self.assertIn("create a V-shaped edge only when image 2 visibly has one", source)
        self.assertNotIn("Image 3 has a narrow upright band collar", source)

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

    def test_generation_never_uses_vl_accessory_categories_as_identity_facts(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        generation_section = source[source.index("identity_instruction ="):source.index("generated_prompt =")]
        self.assertNotIn("hair_accessories", generation_section)
        self.assertNotIn("hair_accessory_details", generation_section)
        self.assertIn("image 1 pixels are authoritative", source)

    def test_single_generation_keeps_quality_settings(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "qwen_edit_enhancer"]
        self.assertEqual(len(calls), 1)
        kwargs = {arg.arg: arg.value for arg in calls[0].keywords}
        self.assertEqual(kwargs["steps"].id, "generation_steps")
        self.assertEqual(ast.literal_eval(kwargs["max_sequence_length"]), 512)
        self.assertEqual(kwargs["true_cfg_scale"].id, "generation_cfg_scale")
        self.assertFalse(ast.literal_eval(kwargs["reject_identity_failure"]))
        self.assertTrue(ast.literal_eval(kwargs["return_raw_candidate"]))
        self.assertEqual(ast.literal_eval(kwargs["conditioning_max_dimension"]), 320)
        self.assertEqual(eval(compile(ast.Expression(kwargs["conditioning_pixel_budget"]), "budget", "eval")), 320 * 320)
        self.assertEqual(ast.literal_eval(kwargs["max_generation_dimension"]), 640)
        self.assertEqual(ast.literal_eval(kwargs["minimum_generation_dimension"]), 640)
        self.assertTrue(ast.literal_eval(kwargs["preserve_reference_aspect"]))
        references = kwargs["reference_images"]
        self.assertIsInstance(references, ast.List)
        self.assertEqual([item.id for item in references.elts], ["template_conditioning"])
        self.assertEqual(kwargs["image_input"].id, "person_pil")

    def test_uniform_generation_balances_identity_and_complete_garment(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn("generation_steps <= 4 else", source)
        self.assertIn("generation_steps <= 12 else 2.0", source)
        pipeline_source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("central_arm_fraction > 0.015", pipeline_source)
        prompt = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("Keep hands, wrists, props and lower arms outside", prompt)
        self.assertIn("Render the complete uniform seamlessly", prompt)

    def test_visual_reference_gate_makes_hair_and_uniform_review_advisory(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("from pipelines.uniform_review import review_uniform", source)
        self.assertIn('advisory_failures = [', source)
        self.assertIn('("uniform_mismatch", "hair_changed")', source)
        self.assertIn('("face_artifacts", "head_cropped")', source)
        self.assertIn('visual_review["delivery_blocked"]', source)

    def test_repeated_inputs_use_a_new_auditable_job_seed(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('zlib.crc32(job_id.encode("utf-8"), uniform_seed)', source)
        self.assertIn('"seed_policy": "content_seed_with_job_retry_salt"', source)

    def test_uniform_steps_are_not_forced_back_to_twenty(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertNotIn("generation_steps = 20", source)
        self.assertIn("measure_source_person_colors", source)
        self.assertIn('"measured_source_colors": measured_person_colors', source)

    def test_single_pass_edits_exact_background_guide_and_uses_identity_reference(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        qwen = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "qwen_edit_enhancer")
        qwen_kwargs = {item.arg: item.value for item in qwen.keywords}
        self.assertEqual(qwen_kwargs["image_input"].id, "person_pil")
        self.assertEqual(qwen_kwargs["background_color"].id, "background_color")
        identity = next(node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "check_uniform_identity")
        self.assertEqual(identity.args[0].id, "person_pil")

        guide_calls = [node for node in calls if isinstance(node.func, ast.Name)]
        identity_guide = next(node for node in guide_calls if node.func.id == "build_person_backdrop_guide")
        self.assertEqual(identity_guide.args[0].id, "person_pil")
        self.assertEqual(identity_guide.args[1].id, "source_labels")
        background_arg = {
            "garment_on_selected_background": 1,
            "build_person_backdrop_guide": 2,
            "build_rough_composite": 4,
        }
        for function_name, argument_index in background_arg.items():
            guide = next(node for node in guide_calls if node.func.id == function_name)
            self.assertEqual(guide.args[argument_index].id, "qwen_bg_rgb")

    def test_qwen_uses_selected_conditioning_background_from_first_generation(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("qwen_bg_rgb = selected_bg_rgb", source)
        self.assertIn('background_color=background_color', source)
        self.assertIn('"selected_background_rgb": list(selected_bg_rgb)', source)
        self.assertNotIn("gray background, white background", source)

        enhance = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        self.assertIn('qwen_bg_desc = "neutral middle gray #808080"', enhance)
        self.assertIn("candidate_pil, selected_hex", enhance)
        self.assertIn("qwen_pil, selected_hex", enhance)

    def test_uniform_generation_uses_full_reference_then_passport_crop(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("template_conditioning.height * .62", source)
        self.assertNotIn("passport_reference_bottom", source)
        self.assertIn("_crop_school_passport_portrait(generated_candidate, width, height)", source)
        prompt_source = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("shoulders and upper chest", prompt_source)
        self.assertIn("application will crop", prompt_source)

    def test_uniform_prompt_demands_a_plain_passport_backdrop_not_a_studio_scene(self):
        prompt_source = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("BACKGROUND MUST BE ONLY one featureless, perfectly flat field", prompt_source)
        self.assertIn("no gradient, texture or scenery; no shadow, room, wall, window or objects", prompt_source)
        self.assertNotIn("school-ID studio pose", prompt_source)

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
        self.assertEqual(len(post_generation_replacements), 4)
        unconditional_smooth = next(node for node in post_generation_replacements
                                    if node.func.attr == "_replace_smooth_border_background"
                                    and node.args[0].id == "cropped"
                                    and node.lineno < next(
                                        candidate.lineno for candidate in post_generation_replacements
                                        if candidate.func.attr == "_replace_background_with_foreground_matte"
                                        and candidate.args[0].id == "cropped"
                                    ))
        pre_identity_smooth = next(node for node in post_generation_replacements
                                   if node.func.attr == "_replace_smooth_border_background"
                                   and node.args[0].id == "identity_locked")
        pre_identity_matte = next(node for node in post_generation_replacements
                                  if node.func.attr == "_replace_background_with_foreground_matte"
                                  and node.args[0].id == "identity_locked")
        final_smooth = next(node for node in post_generation_replacements
                            if node.func.attr == "_replace_smooth_border_background"
                            and node.args[0].id == "cropped")
        final_matte = next(node for node in post_generation_replacements
                           if node.func.attr == "_replace_background_with_foreground_matte"
                           and node.args[0].id == "cropped")
        self.assertEqual(pre_identity_smooth.args[1].id, "background_color")
        self.assertEqual(unconditional_smooth.args[1].id, "background_color")
        self.assertLess(pre_identity_smooth.lineno, pre_identity_matte.lineno)
        self.assertLess(pre_identity_matte.lineno, final_smooth.lineno)
        final_matte_kwargs = {item.arg: item.value for item in final_matte.keywords}
        self.assertEqual(final_matte_kwargs["background_color"].id, "background_color")
        self.assertTrue(ast.literal_eval(final_matte_kwargs["strict"]))
        background = next(node for node in calls if isinstance(node.func, ast.Attribute)
                          and node.func.attr == "_has_selected_solid_background")
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        self.assertGreater(finishing.lineno, background.lineno)

    def test_uniform_finishing_preserves_qwen_lighting_and_uses_detected_glare(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        kwargs = {item.arg: item.value for item in finishing.keywords}
        self.assertEqual(kwargs["correct_dark_hair"].id, "correct_generated_hair_glare")
        self.assertIsNone(ast.literal_eval(kwargs["face_target_luma"]))

    def test_uniform_delivery_does_not_overprocess_accepted_qwen_face(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        finishing = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "finish_uniform_tones")
        kwargs = {item.arg: item.value for item in finishing.keywords}
        self.assertFalse(ast.literal_eval(kwargs["restore_head_detail"]))
        restorations = [node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "restore_uniform_face_detail"]
        self.assertFalse(restorations)

    def test_qwen_hair_avoids_source_matte_restore_artifacts_after_generation(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        head_restore = [node for node in calls if isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_restore_uniform_hair_accessories"]
        self.assertEqual(len(head_restore), 0)
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Skipped to avoid source-matte hair artifacts", source)

    def test_hair_restore_uses_shared_confident_hair_region(self):
        source = (ROOT / "pipelines/uniform_finishing.py").read_text(encoding="utf-8")
        self.assertIn("common, confident hair interior", source)
        self.assertIn("generated_hair_safe", source)
        self.assertIn("* .38", source)
        self.assertIn("* .72", source)

    def test_flat_qwen_backdrop_does_not_use_segmentation_fallback(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn("if not background_matches and not flat_backdrop:", source)
        self.assertIn("otherwise flat selected backdrop", source)

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

    def test_positive_and_negative_prompts_share_one_encoder_worker(self):
        tree = ast.parse((ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        enhancer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == "qwen_edit_enhancer")
        calls = [n for n in ast.walk(enhancer) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_encode_prompt_isolated"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[1].id, "prompts_to_encode")
        keywords = {k.arg: k.value for k in calls[0].keywords}
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

    def test_vl_workers_use_low_memory_gpu_and_cache_by_image_content(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn('VL_CACHE_VERSION = "content-v3"', source)
        self.assertIn("BitsAndBytesConfig", source)
        self.assertIn('device_map={{"": 0}} if torch.cuda.is_available() else "cpu"', source)
        self.assertIn('_analysis_cache_path("portrait", image)', source)
        self.assertIn('_analysis_cache_path("uniform", person, template)', source)
        self.assertIn("analysis cache hit", source)

    def test_isolated_encoder_disables_autograd(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("with torch.inference_mode():\n    for prompt in prompts:", source)
        self.assertIn('\"encoding\": \"utf-8\"', source)
        self.assertIn('\"errors\": \"replace\"', source)


if __name__ == "__main__":
    unittest.main()
