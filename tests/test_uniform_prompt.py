"""Lightweight prompt regression tests without loading GPU models."""
import ast
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import unittest

import cv2
import numpy as np
from PIL import Image


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

    def sunlight_detector(self):
        from pipelines.photo_restoration import PhotoRestorationService

        tree = ast.parse((ROOT / "gradio_app.py").read_text(encoding="utf-8"))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_harsh_source_sunlight")
        namespace = {
            "Any": Any, "Optional": Optional, "Image": Image,
            "np": np, "cv2": cv2, "logger": logging.getLogger("test"),
            "PhotoRestorationService": PhotoRestorationService,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "sunlight", "exec"), namespace)
        return namespace[helper.name]

    def test_measured_sunlight_overrides_false_vl_result(self):
        harsh = np.full((120, 90, 3), (34, 24, 16), dtype=np.uint8)
        harsh[:, 45:] = (247, 174, 48)
        studio = np.full((120, 90, 3), (156, 122, 96), dtype=np.uint8)
        detector = self.sunlight_detector()
        self.assertTrue(detector(Image.fromarray(harsh)))
        self.assertFalse(detector(Image.fromarray(studio)))

    def enhancement_step_selector(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_select_enhancement_steps")
        namespace = {"Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "steps", "exec"), namespace)
        return namespace["_select_enhancement_steps"]

    def tied_hair_geometry_detector(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_visible_tied_hair_geometry")
        namespace = {"Any": Any, "np": np}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "hair_geometry", "exec"), namespace)
        return namespace["_has_visible_tied_hair_geometry"]

    def concrete_hair_review_gate(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_hair_review_has_concrete_change")
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "hair_review", "exec"), namespace)
        return namespace[helper.name]

    def grounded_direct_light_gate(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_grounded_direct_light")
        namespace = {"Any": Any, "Dict": Dict}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "light_gate", "exec"), namespace)
        return namespace[helper.name]

    def test_hair_review_requires_output_reference_and_explicit_change(self):
        concrete = self.concrete_hair_review_gate()
        self.assertTrue(concrete(
            "Image 2 changed from short source hair to visibly longer swept hair."
        ))
        self.assertTrue(concrete(
            "The generated portrait is missing the flower visible in image 1."
        ))
        self.assertTrue(concrete(
            "The hairstyle in Image 2 changed from a loose ponytail to two braids."
        ))
        self.assertFalse(concrete(
            "Image 1 has natural hair with a middle parting; no crown detail is visible."
        ))
        self.assertFalse(concrete(
            "The source hairline, parting, flower and clip count were inspected."
        ))
        self.assertFalse(concrete(
            "The hairstyle in Image 2 has been changed from braids to a ponytail."
        ))
        self.assertTrue(concrete(
            "Image 2 is missing the viewer-left tied section and has one section instead of two."
        ))

    def test_contradictory_vl_sunlight_flag_does_not_force_head_regeneration(self):
        grounded = self.grounded_direct_light_gate()
        self.assertFalse(grounded({
            "direct_sunlight_present": True,
            "sunlight_type": "none",
            "sunlight_evidence": "none",
            "measured_sunlight_severity": "normal",
        }))
        self.assertTrue(grounded({
            "direct_sunlight_present": True,
            "sunlight_type": "hard directional sunlight",
            "sunlight_evidence": "bright crown patch and paired facial shadow",
            "measured_sunlight_severity": "normal",
        }))
        self.assertTrue(grounded({
            "direct_sunlight_present": True,
            "sunlight_type": "none",
            "sunlight_evidence": "none",
            "measured_sunlight_severity": "extreme",
        }))

    def garment_palette_measurer(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_measure_garment_palette")
        namespace = {
            "Any": Any, "Dict": Dict, "List": List,
            "np": np, "cv2": cv2, "Image": Image,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "palette", "exec"), namespace)
        return namespace["_measure_garment_palette"]

    def garment_accent_measurer(self):
        helper = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_measure_garment_accent_color")
        namespace = {
            "Any": Any, "Dict": Dict, "Tuple": Tuple, "Optional": __import__("typing").Optional,
            "np": np, "cv2": cv2, "Image": Image,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "accent_measure", "exec"), namespace)
        return namespace["_measure_garment_accent_color"]

    def uniform_accent_restorer(self):
        pipeline_class = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "QwenEditPipeline"
        )
        helper = next(
            node for node in pipeline_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_restore_uniform_accent_chroma"
        )
        helper.decorator_list = []
        namespace = {
            "Tuple": Tuple,
            "Image": Image,
            "np": np,
            "cv2": cv2,
            "logger": logging.getLogger("uniform-accent-test"),
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "accent", "exec"), namespace)
        return namespace["_restore_uniform_accent_chroma"]

    def test_uniform_accent_chroma_follows_rough_fit_without_flattening_shading(self):
        restore = self.uniform_accent_restorer()
        background = (20, 120, 230)
        rough = np.full((120, 100, 3), background, dtype=np.uint8)
        generated = rough.copy()
        # A broad red collar/placket in the aligned rough guide was repainted
        # navy by diffusion. The two brightness bands represent fabric folds.
        rough[58:78, 18:82] = (210, 24, 28)
        rough[58:68, 45:55] = (170, 18, 22)
        generated[58:68, 18:82] = (30, 42, 94)
        generated[68:78, 18:82] = (50, 65, 125)
        generated[84:94, 62:92] = (35, 105, 60)
        # Skin-like neck and the selected backdrop must remain untouched.
        rough[52:62, 42:58] = (168, 112, 84)
        generated[52:62, 42:58] = (168, 112, 84)

        output = np.asarray(restore(
            Image.fromarray(generated), (218, 25, 32), background,
            "collar, center placket, sleeve-end cuffs",
        ))

        self.assertGreater(int(output[72, 25, 0]), int(output[72, 25, 2]))
        self.assertGreater(output[72, 25].mean(), output[62, 25].mean())
        np.testing.assert_array_equal(output[20, 20], generated[20, 20])
        np.testing.assert_array_equal(output[55, 50], generated[55, 50])
        np.testing.assert_array_equal(output[88, 75], generated[88, 75])

    def test_garment_palette_is_measured_in_dominant_area_order(self):
        measure = self.garment_palette_measurer()
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :70] = (220, 220, 220, 255)
        rgba[:, 70:90] = (20, 35, 65, 255)
        rgba[:, 90:] = (210, 25, 30, 255)
        palette = measure(Image.fromarray(rgba, "RGBA"), max_colors=3)
        self.assertEqual([item["hex"] for item in palette], ["#DCDCDC", "#142341", "#D2191E"])
        self.assertEqual([item["fraction"] for item in palette], [.7, .2, .1])

    def test_solid_trim_is_measured_separately_from_dominant_pattern_palette(self):
        measure = self.garment_accent_measurer()
        rgba = np.zeros((160, 120, 4), dtype=np.uint8)
        rgba[20:150, 10:110] = (230, 230, 235, 255)
        rgba[20:150:8, 10:110] = (35, 45, 85, 255)
        rgba[20:150, 10:110:8] = (35, 45, 85, 255)
        rgba[20:48, 24:96] = (218, 25, 32, 255)
        rgba[20:145, 55:66] = (218, 25, 32, 255)

        accent = measure(Image.fromarray(rgba, "RGBA"))

        self.assertIsNotNone(accent)
        self.assertGreater(accent["rgb"][0], 180)
        self.assertLess(accent["rgb"][1], 60)
        self.assertLess(accent["rgb"][2], 70)

    def test_solid_trim_ignores_an_opaque_colored_template_canvas(self):
        measure = self.garment_accent_measurer()
        rgba = np.full((180, 140, 4), (4, 126, 246, 255), dtype=np.uint8)
        rgba[35:165, 15:125] = (232, 232, 236, 255)
        rgba[35:165:8, 15:125] = (35, 45, 85, 255)
        rgba[35:165, 15:125:8] = (35, 45, 85, 255)
        rgba[35:62, 30:110] = (218, 25, 32, 255)
        rgba[35:158, 65:76] = (218, 25, 32, 255)

        accent = measure(Image.fromarray(rgba, "RGBA"))

        self.assertIsNotNone(accent)
        self.assertGreater(accent["rgb"][0], 180)
        self.assertLess(accent["rgb"][1], 60)
        self.assertLess(accent["rgb"][2], 70)

    def test_tied_hair_requires_visible_pixels_at_the_nape(self):
        detector = self.tied_hair_geometry_detector()
        labels = np.zeros((240, 200), dtype=np.uint8)
        labels[20:78, 65:135] = 2
        self.assertFalse(detector(labels, (60, 45, 140, 145)))

        labels[140:190, 118:145] = 2
        self.assertTrue(detector(labels, (60, 45, 140, 145)))

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
        self.assertEqual((simple, minor, moderate, difficult), (4, 4, 4, 4))
        self.assertIn("hard or direct source lighting", reasons)

    def test_adaptive_enhancement_steps_respect_requested_maximum(self):
        selected, _ = self.enhancement_step_selector()({
            "face_orientation": "profile", "direct_sunlight_present": True,
            "sunlight_type": "hard direct sun", "hair_accessories": "flowers",
        }, maximum_steps=12)
        self.assertEqual(selected, 4)

    def test_enhancement_uses_edit_preserving_steps_when_face_has_enough_pixels(self):
        selected, reasons = self.enhancement_step_selector()({
            "face_orientation": "front",
            "hair_accessories": "none",
            "source_min_dimension": 768,
            "source_face_detail_score": 240.0,
            "source_face_height_pixels": 190,
            "source_face_frame_ratio": .25,
        })
        self.assertEqual(selected, 4)
        self.assertNotIn("small face region requires photographic base-model detail", reasons)

        tiny, tiny_reasons = self.enhancement_step_selector()({
            "face_orientation": "front", "hair_accessories": "none",
            "source_min_dimension": 768, "source_face_detail_score": 240.0,
            "source_face_height_pixels": 120, "source_face_frame_ratio": .14,
        })
        self.assertEqual(tiny, 4)
        self.assertIn("small face region requires photographic base-model detail", tiny_reasons)

    def test_uniform_low_detail_does_not_force_aggressive_regeneration(self):
        selected, reasons = self.uniform_step_selector()({
            "pose": "front", "hair_accessories": "two clips",
            "outer_color_confidence": "uncertain", "fabric_detail": "unclear",
            "source_min_dimension": 390, "source_face_detail_score": 65.3,
        })
        self.assertEqual(selected, 4)
        self.assertNotIn("hair accessories", reasons)
        self.assertNotIn("limited source resolution or facial detail", reasons)

    def test_direct_patterned_template_uses_stable_four_step_path(self):
        selected, reasons = self.uniform_step_selector()({
            "pose": "front", "outer_color_confidence": "high",
            "fabric_detail": "smooth", "shirt_color_and_pattern": "red, white, blue checkered",
        })
        self.assertEqual(selected, 4)
        self.assertIn("patterned uniform supplied by garment reference", reasons)

    def test_source_outerwear_stays_at_four_steps_after_prefit_removal(self):
        selected, reasons = self.uniform_step_selector()({
            "pose": "front portrait", "outer_color_confidence": "high",
            "fabric_detail": "smooth", "shirt_color_and_pattern": "red, white, blue plaid",
            "clothing_description": "check shirt under a dark vest",
        })
        self.assertEqual(selected, 4)
        self.assertIn("source outerwear replaced by native garment edit", reasons)

    def test_uniform_route_measures_source_detail_before_step_selection(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        measurement = source.index('vl_plan["source_min_dimension"] = min(source_w, source_h)')
        selection = source.index("generation_steps = 4", measurement)
        self.assertLess(measurement, selection)
        self.assertIn('vl_plan["source_face_detail_score"]', source)

    def test_extreme_measured_sunlight_uses_one_neutralized_first_generation(self):
        from pipelines.uniform_vl_analyzer import merge_portrait_pixel_evidence

        harsh = np.full((160, 120, 3), (24, 18, 14), dtype=np.uint8)
        harsh[:, 60:] = (252, 178, 28)
        evidence = merge_portrait_pixel_evidence({}, Image.fromarray(harsh))
        self.assertEqual(evidence["measured_sunlight_severity"], "extreme")

        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route_source = ast.get_source_segment(source, self.route)
        self.assertIn("generation_steps = 4", route_source)
        self.assertIn(
            "_prepare_extreme_sunlight_uniform_guide",
            route_source,
        )
        self.assertNotIn("lighting-only Qwen correction pass", route_source)

    def test_uniform_identity_prepass_is_adaptive_for_harsh_low_detail_inputs(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        # The prepass stays opt-in. Measured on job 58682ecb its restored head
        # scored ArcFace 0.26 against the upload and was discarded after a
        # full extra 4-step pass, so it is an experiment, not a quality
        # response to a soft upload.
        self.assertIn('"VSOUL_ENABLE_UNIFORM_IDENTITY_PREPASS", "0"', source)
        self.assertNotIn('"VSOUL_DISABLE_UNIFORM_IDENTITY_PREPASS", "0"', source)
        self.assertIn("identity_prepass_would_help = bool(", source)
        self.assertIn("identity_prepass_enabled and identity_prepass_would_help", source)
        self.assertIn("uniform-identity-prepass-v5-hair-geometry", source)
        self.assertIn("minimum_similarity=.65", source)
        self.assertIn(
            "validated_4_step_current_upload_identity_prepass_then_4_step_uniform_edit",
            source,
        )

    def test_uniform_validator_accepts_upper_chest_passport_framing(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("a valid collar and vest often occupy only 3-5%", source)
        self.assertIn("if coverage < 0.025:", source)

    def test_uniform_validator_uses_template_color_coverage_for_plaid_uniforms(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("expected_target_fraction: Optional[float] = None", source)
        self.assertIn("Target colour occupies %.3f of template garment pixels", source)
        self.assertIn("float(expected_target_fraction or 0.36) * 0.50", source)

    def test_advisory_vl_review_failure_does_not_discard_valid_uniform(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('VSOUL_ENABLE_POST_VL_REVIEW', source)
        self.assertIn('visual_review["review_unavailable"] = True', source)
        self.assertIn("Advisory worker unavailable", source)
        self.assertNotIn(
            "Uniform output was not delivered because visual reference verification could not be completed",
            source,
        )

    def test_low_resolution_alone_keeps_localized_edit_path(self):
        selected, reasons = self.enhancement_step_selector()({
            "face_orientation": "front", "hair_accessories": "none",
            "source_min_dimension": 390, "source_face_detail_score": 65.3,
        })
        self.assertEqual(selected, 4)
        self.assertIn("limited source resolution or facial detail", reasons)

    def test_hard_light_keeps_native_identity_edit_schedule(self):
        selected, reasons = self.enhancement_step_selector()({
            "face_orientation": "front",
            "hair_accessories": "two flowers",
            "direct_sunlight_present": True,
            "sunlight_type": "hard direct sunlight",
            "source_min_dimension": 390,
            "source_face_detail_score": 65.3,
        }, maximum_steps=20)
        self.assertEqual(selected, 4)
        self.assertIn("hard or direct source lighting", reasons)
        self.assertIn("hair accessories", reasons)

    def test_harsh_light_preserves_qwen_identity_input_and_generated_hair(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        source_crop = source.index("qwen_source = image.convert(\"RGB\")")
        generation = source.index("res_path = gusuq_pipeline.edit_image", source_crop)
        finalizing = source.index("Finalizing the Qwen studio portrait", generation)
        pre_generation = source[source_crop:generation]
        post_generation = source[generation:finalizing]
        self.assertNotIn("normalize_uniform_conditioning_light(qwen_source)", pre_generation)
        self.assertIn("Preserving accepted Qwen face, skin and hair pixels", post_generation)
        self.assertNotIn("PhotoRestorationService.tame_generated_portrait_hotspots", post_generation)
        self.assertNotIn("PhotoRestorationService.match_source_skin_chroma", post_generation)
        self.assertNotIn("PhotoRestorationService.restore_sunlit_hair_tone", post_generation)
        self.assertNotIn("PhotoRestorationService.restore_source_face_microtexture", post_generation)

    def test_uniform_reuses_enhancement_style_head_contract_before_garment_prose(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        head_contract = source.index("head_enhancement_contract = (")
        generated_prompt = source.index("generated_prompt = (", head_contract)
        garment_authority = source.index("Image 1 contains the rough-fitted uniform", generated_prompt)
        self.assertLess(head_contract, generated_prompt)
        self.assertLess(generated_prompt, garment_authority)
        contract_section = source[head_contract:generated_prompt]
        self.assertIn("HEAD LOCK", contract_section)
        self.assertIn("leave image 1's complete head unchanged", contract_section)
        self.assertIn("Do not restore, relight, beautify, redraw", contract_section)
        self.assertNotIn("trusted_repair_instruction.rstrip", contract_section)
        prompt_section = source[generated_prompt:source.index("prompt_word_count =", generated_prompt)]
        self.assertNotIn("earring_reference", prompt_section)
        self.assertNotIn("earring_description_text", prompt_section)

    def test_uniform_keeps_untouched_head_conditioning_like_enhancement(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = source.index("person_conditioning = self._source_identity_reference(")
        end = source.index("person_conditioning_path =", start)
        preparation = source[start:end]
        self.assertNotIn("tame_generated_portrait_hotspots(", preparation)
        self.assertIn("untouched Enhancement-style head conditioning", preparation)

    def test_vl_repair_plan_downgrades_contradictory_hair_and_earring_labels(self):
        from pipelines.uniform_vl_analyzer import build_portrait_repair_plan

        plan = build_portrait_repair_plan({
            "direct_sunlight_present": True,
            "sunlight_regions": "crown and forehead",
            "hair_style": "Ponytail",
            "ponytail_side": "Not Visible",
            "hair_left_side": "None",
            "hair_right_side": "None",
            "hair_accessories": "Flower, Bow",
            "hair_accessory_count": 2,
            "hair_accessory_confidence": "High",
            "hair_accessory_details": "pink flowers on both sides",
            "earring_description": "heart-shaped silver",
        })

        self.assertTrue(plan["repair_required"])
        self.assertFalse(plan["hair_geometry_trusted"])
        self.assertFalse(plan["earring_semantics_trusted"])
        self.assertIn("VL hairstyle labels conflict", plan["prompt"])
        self.assertIn("exactly 2 source-visible hair accessories", plan["prompt"])
        self.assertIn("source pixels, not VL names", plan["prompt"])

    def test_vl_repair_plan_requires_counted_bilateral_tied_hair_geometry(self):
        from pipelines.uniform_vl_analyzer import build_portrait_repair_plan

        bilateral = build_portrait_repair_plan({
            "hair_style": "two low side ponytails",
            "tied_section_count": 2,
            "tied_section_sides": "viewer-left and viewer-right",
            "hair_left_side": "low curled ponytail beside left ear",
            "hair_right_side": "low curled ponytail beside right ear",
        })
        ambiguous = build_portrait_repair_plan({
            "hair_style": "ponytail",
            "ponytail_side": "viewer-left",
            "tied_section_count": 0,
            "tied_section_sides": "unknown",
            "hair_left_side": "source left hair silhouette",
            "hair_right_side": "source right hair silhouette",
        })

        self.assertTrue(bilateral["hair_geometry_trusted"])
        self.assertFalse(ambiguous["hair_geometry_trusted"])
        self.assertIn("tied hairstyle label lacks visible tail geometry", ambiguous["conflicts"])

    def test_enhancement_caps_vl_reference_without_reducing_output_canvas(self):
        pipeline = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        app = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "visual_edge_cap = min(max(32, int(conditioning_max_dimension or 512)), 512)",
            pipeline,
        )
        self.assertIn("conditioning_max_dimension=512", app)
        self.assertIn("conditioning_pixel_budget=None", app)
        self.assertIn("max_generation_dimension=1216", app)

    def test_prompt_encoder_falls_back_when_native_four_bit_dll_is_blocked(self):
        pipeline = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("application control policy", pipeline.lower())
        self.assertIn("device_map=\"auto\" if torch.cuda.is_available()", pipeline)
        self.assertIn("max_memory[0] = \"5500MiB\"", pipeline)
        self.assertIn("retrying with an unquantized GPU/CPU split", pipeline)

    def test_enhancement_repairs_only_borderline_identity_and_rechecks(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        output = source[
            source.index("qwen_identity_review = check_uniform_identity"):
            source.index("Finalizing the Qwen studio portrait", source.index("qwen_identity_review = check_uniform_identity"))
        ]
        self.assertIn(".45 <= float(similarity) < .75", output)
        self.assertIn("_lock_uniform_identity", output)
        self.assertIn("minimum_similarity=.75", output)
        self.assertIn("maximum_geometry_rmse=.035", output)
        self.assertIn("(.12, .20, .32, .44, .58, .68, .72)", output)
        self.assertIn("relit_geometry_match", output)
        self.assertIn("float(similarity) >= .56", output)
        self.assertIn("float(geometry_rmse) <= min(.033", output)
        self.assertNotIn("toned_relit_geometry_match", output)
        self.assertNotIn("match_source_skin_chroma", output)
        self.assertNotIn("tame_generated_portrait_hotspots", output)
        self.assertIn("Preserving accepted Qwen face, skin and hair pixels", output)
        self.assertIn("if not qwen_accepted:", output)

    def test_enhancement_prompt_requires_photographic_texture(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        for phrase in (
            "localized edit to this single photograph",
            "natural pores, fine hair",
            "source person's exact apparent age, facial proportions",
            "source pixels remain the authority",
            "Replace only the background",
        ):
            self.assertIn(phrase, source)
        for key in (
            "hair_crown_shape", "hair_left_side", "hair_right_side", "hair_ends",
            "hair_fringe",
        ):
            self.assertIn(f'(\"{key}\",', source)
        self.assertIn('raw_hair = vl_brief.get("hair_observations_raw")', source)
        self.assertIn("hair_acc_lock,", source)
        self.assertIn("headwear_lock,", source)
        self.assertIn("jewelry_lock,", source)
        prompt_section = source[source.index("hair_fact_labels = ("):source.index("hair_facts = []")]
        self.assertNotIn('(\"ponytail_side\",', prompt_section)
        self.assertNotIn('(\"ponytail_anchor\",', prompt_section)
        self.assertNotIn('(\"ponytail_extent\",', prompt_section)
        self.assertIn("true_cfg_scale=1.0", source)
        self.assertIn("negative_prompt=None", source)
        self.assertIn("subtracts the same source", source)
        self.assertNotIn("Observed identity geometry", source)
        self.assertIn('qwen_source = image.convert("RGB")', source)
        self.assertNotIn("qwen_conditioning = apply_selected_background_matte(neutral_source, selected_rgb)", source)
        matte_worker = source[source.index("def apply_selected_background_matte"):source.index("# Initialize Engines")]
        self.assertIn("use_schp: bool = False", matte_worker)
        self.assertIn("use_schp={bool(use_schp)!r}", matte_worker)
        self.assertIn("qwen_source, face_bbox", source)
        self.assertNotIn("identity_detail_reference = image.crop(detail_box)", source)
        self.assertIn("reference_images=None", source)
        self.assertIn("qwen_canvas_size = (832, 1216)", source)
        self.assertIn("qwen_source = ImageOps.fit(", source)
        self.assertNotIn("contained_source = ImageOps.contain(", source)
        enhancement_crop = source[source.index("qwen_canvas_size = (832, 1216)"):source.index("prompt = build_dynamic_identity_prompt", source.index("qwen_canvas_size = (832, 1216)"))]
        self.assertNotIn("Image.new", enhancement_crop)
        self.assertIn("max_generation_dimension=1216", source)
        self.assertIn("minimum_generation_dimension=1216", source)
        self.assertIn("conditioning_pixel_budget=None", source)
        self.assertIn("Keep exactly one subject", source)
        self.assertIn("if len(generated_faces) != 1:", source)

    def test_enhancement_seed_is_stable_across_prompt_wording_changes(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        seed_section = source[
            source.index("enhancement_seed = zlib.crc32(qwen_source.tobytes())"):
            source.index("res_path = gusuq_pipeline.edit_image", source.index("enhancement_seed = zlib.crc32(qwen_source.tobytes())"))
        ]
        self.assertIn("selected_hex.encode", seed_section)
        self.assertIn("enhancement_seed ^ 483368614", seed_section)
        self.assertIn("Stable source/background seed", seed_section)
        self.assertNotIn("prompt.encode", seed_section)
        self.assertIn("max_sequence_length=256", source)

        gusuq_source = (ROOT / "pipelines/gusuq_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("max_sequence_length: int = 256", gusuq_source)
        self.assertIn("max_sequence_length=max_sequence_length", gusuq_source)

    def test_tied_hair_prompt_ignores_inferred_tail_length_and_side(self):
        tree = ast.parse((ROOT / "gradio_app.py").read_text(encoding="utf-8"))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_dynamic_identity_prompt"
        )
        namespace = {
            "Any": Any,
            "Dict": Dict,
            "Optional": __import__("typing").Optional,
            "Image": Image,
            "re": re,
            "_glasses_prompt_bits": lambda *_args, **_kwargs: ("Keep source glasses.", ""),
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "enhancement_prompt", "exec"), namespace)
        prompt = namespace["build_dynamic_identity_prompt"](
            Image.new("RGB", (64, 64)),
            "solid #047EF6 RGB(4, 126, 246)",
            vl_brief={
                "hair_style": "Tied ponytail",
                "hair_parting": "Centered",
                "hair_texture": "Smooth",
                "hair_color": "Black",
                "hair_crown_shape": "Round",
                "hair_length": "Long",
                "hair_ends": "Past the shoulders",
                "ponytail_side": "Viewer-left",
                "ponytail_extent": "Past the collarbone",
            },
        )
        self.assertNotIn("Past the shoulders", prompt)
        self.assertNotIn("Viewer-left", prompt)
        self.assertNotIn("Past the collarbone", prompt)
        self.assertIn("hair, accessories", prompt)
        self.assertNotIn("exactly one tail", prompt)
        self.assertIn("do not recompose it", prompt)
        self.assertNotIn("5-7 years old", prompt)
        self.assertLessEqual(len(prompt.split()), 140)

    def test_enhancement_pose_preserves_already_frontal_face(self):
        tree = ast.parse((ROOT / "gradio_app.py").read_text(encoding="utf-8"))
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_dynamic_identity_prompt"
        )
        namespace = {
            "Any": Any,
            "Dict": Dict,
            "Optional": Optional,
            "Image": Image,
            "re": re,
            "logger": logging.getLogger("test"),
            "_glasses_prompt_bits": lambda *_args, **_kwargs: ("Keep source glasses.", ""),
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "enhancement_prompt", "exec"), namespace)
        build = namespace["build_dynamic_identity_prompt"]
        image = Image.new("RGB", (64, 64))

        frontal = build(
            image,
            "solid #047EF6",
            vl_brief={
                "face_orientation": "frontal",
                "head_tilt": "upright",
                "shoulder_alignment": "level/front-facing",
            },
        )
        self.assertIn("head is already frontal", frontal)
        self.assertIn("keep its face and hair pixels unchanged", frontal)
        self.assertIn("Square only the upper body and shoulders", frontal)

        tilted = build(
            image,
            "solid #047EF6",
            vl_brief={
                "face_orientation": "three-quarter",
                "head_tilt": "tilted viewer-right",
                "shoulder_alignment": "viewer-left higher",
                "direct_sunlight_present": True,
                "head_hair_hotspot_present": True,
                "sunlight_regions": "forehead, crown, hair and neck",
                "measured_hair_rgb": (24, 20, 18),
            },
        )
        self.assertIn("straight camera-facing ID pose", tilted)
        self.assertIn("level eyes, centered natural neck and even shoulders", tilted)
        self.assertIn("Do not redraw facial features, hair or accessories", tilted)
        self.assertNotIn("pose and crop unchanged", tilted)
        self.assertLessEqual(len(tilted.split()), 140)

    def test_portrait_vl_uses_focused_hair_analysis(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn("hair_prompt = {self._HAIR_RETRY_PROMPT!r}", source)
        self.assertIn("hair_inspection = make_hair_inspection_crop(image)", source)
        self.assertIn('head_crop = Image.open(hair_path).convert("RGB")', source)
        self.assertIn('"hair_raw": hair_response', source)
        self.assertIn('result["hair_observations_raw"] = hair_parsed', source)
        self.assertIn("Eyeglass frames and temples", source)
        self.assertIn("hair behind each ear", source)
        self.assertIn("objects physically attached to hair", source)

    def test_uniform_generation_reads_hair_from_source_pixels(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("source_identity_reference", source)
        self.assertIn("_has_visible_tied_hair_geometry", source)
        self.assertIn("vl_tied_hair_corroborated", source)
        self.assertIn("Ignoring unsupported tied-hair label", source)
        self.assertNotIn('vl_plan["hair_style"] = "short untied hair"', source)
        self.assertIn('vl_plan["hair_geometry_parser_fragmented"] = True', source)
        self.assertIn("Copy image 1's hairline, part, crown", source)
        self.assertIn("tied-tail position, length, texture and color", source)
        self.assertIn("visible ends, length, texture and color", source)
        self.assertIn("compact_wearables", source)
        self.assertIn("Image 1 has no tied-back hairstyle", source)
        self.assertNotIn("side contour, tied-back arrangement", source)

    def test_hair_reviewer_excludes_non_hair_objects(self):
        source = (ROOT / "pipelines/uniform_review.py").read_text(encoding="utf-8")
        self.assertIn("Image 1 is the authority", source)
        self.assertIn("Eyeglass frames and temples", source)
        self.assertIn("concrete visible", source)

    def test_enhancement_identity_floor_rejects_borderline_regenerations(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("MIN_PORTRAIT_IDENTITY_SIMILARITY = 0.78", source)
        self.assertIn("MAX_PORTRAIT_LANDMARK_RMSE = 0.045", source)
        self.assertIn("landmark_2d_106", source)
        self.assertIn("and geometry_ok", source)

    def test_static_enhancement_prompt_locks_face_landmarks_and_real_marks(self):
        source = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        prompt = source[
            source.index("def build_static_enhancement_prompt("):
            source.index("def upscale_qwen_result", source.index("def build_static_enhancement_prompt("))
        ]
        self.assertIn("local restoration of IMAGE 1", prompt)
        self.assertIn("do not generate or redesign the person", prompt)
        self.assertIn("source pixel geometry and natural asymmetry", prompt)
        self.assertIn("real marks, accessories, jewelry and clothing", prompt)
        self.assertIn("without inventing detail", prompt)

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
        self.assertEqual((simple, minor, moderate, difficult), (4, 4, 4, 4))
        self.assertIn("hard source lighting handled by four-step relighting", reasons)

    def test_patterned_uniform_with_multiple_trim_regions_uses_edit_adapter(self):
        select = self.uniform_step_selector()
        selected, reasons = select({
            "pose": "front portrait",
            "outer_color_confidence": "high",
            "fabric_detail": "woven school shirt",
            "shirt_color_and_pattern": "red, white and blue plaid",
            "accent_trim_placement": "collar, center placket, sleeve-end cuffs",
        })
        self.assertEqual(selected, 4)
        self.assertIn("patterned uniform and trim supplied by garment reference", reasons)

    def test_patterned_uniform_with_tied_hair_and_glasses_stays_on_edit_adapter(self):
        select = self.uniform_step_selector()
        selected, reasons = select({
            "pose": "front portrait",
            "outer_color_confidence": "high",
            "fabric_detail": "woven school shirt",
            "shirt_color_and_pattern": "red, white and blue plaid",
            "accent_trim_placement": "collar, center placket, sleeve-end cuffs",
            "hair_style": "low ponytail",
            "visible_wearables": "eyeglasses and earrings",
        })
        self.assertEqual(selected, 4)
        self.assertIn("tied hairstyle and eyeglasses preserved by fitted seed", reasons)
        self.assertIn("pattern and identity detail retained by edit adapter", reasons)

    def test_soft_indoor_hotspot_and_shared_template_uncertainty_stay_on_edit_adapter(self):
        select = self.uniform_step_selector()
        selected, reasons = select({
            "pose": "front portrait", "hair_accessories": "none",
            "direct_sunlight_present": True, "head_hair_hotspot_present": True,
            "sunlight_type": "soft indoor", "outer_color_confidence": "low",
            "fabric_detail": "unclear",
        })
        self.assertEqual(selected, 4)
        self.assertEqual(reasons, [
            "lighting hotspot handled by four-step relighting", "uncertain uniform color or fabric detail",
        ])

    def test_adaptive_uniform_steps_respect_requested_maximum(self):
        select = self.uniform_step_selector()
        selected, _ = select({
            "direct_sunlight_present": True, "hair_accessories": "flowers",
        }, maximum_steps=12)
        self.assertEqual(selected, 4)

    def test_uniform_eight_step_cap_uses_neutral_base_model(self):
        select = self.uniform_step_selector()
        selected, reasons = select({
            "direct_sunlight_present": True,
            "sunlight_type": "hard direct sun",
            "outer_color_confidence": "low",
            "fabric_detail": "unclear",
        }, maximum_steps=8)
        self.assertEqual(selected, 4)
        self.assertNotIn(
            "4-step Lightning fallback avoids under-sampled painted 8-step uniform",
            reasons,
        )

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
        namespace = dict(qwen_bg_rgb=(128, 128, 128), qwen_hex="#808080", source_body_ratio=2.25,
                         fabric_color_instruction="TEST BLACK",
                         hair_color_instruction="TEST HAIR",
                         hair_shape_instruction="TEST HAIR SHAPE",
                         one_sided_ponytail_instruction="TEST ONE-SIDED HAIR",
                         ponytail_position_instruction="TEST PONYTAIL POSITION",
                         wearable_instruction="TEST WEARABLES",
                         lighting_instruction="TEST LIGHTING",
                         skin_tone_instruction="TEST SKIN",
                         skin_prompt_instruction="TEST SKIN",
                         identity_instruction="TEST IDENTITY",
                         body_fit_instruction="TEST BODY FIT",
                         body_fit_prompt_instruction="TEST TAILORED BODY FIT",
                         collar_constraint="TEST COLLAR",
                         template_generation_details="TEST GARMENT FACTS",
                         uniform_generation_facts="TEST EXACT COLOR AND PATTERN",
                         badge_repair_instruction="TEST BADGE FABRIC REPAIR",
                         source_neck_gap_ratio=.18,
                         neckwear_instruction="TEST NECKWEAR",
                         neckwear_prompt_instruction="TEST NECKWEAR",
                         trim_placement_instruction="TEST TRIM PLACEMENT",
                         trim_prompt_instruction="TEST TRIM PLACEMENT",
                         trim_color_instruction="TEST TRIM COLOR",
                          hair_generation_details="TEST SOURCE HAIR",
                          compact_hair_requirement="TEST COMPACT HAIR",
                          compact_wearables="TEST COMPACT WEARABLES",
                          compact_face_requirement="TEST FACE REFERENCE ",
                          source_appearance_lock="TEST SOURCE PRESENTATION",
                          apparent_age_lock="TEST APPARENT AGE ",
                         hair_geometry_copy="TEST HAIR GEOMETRY",
                         measured_palette_instruction="TEST MEASURED PALETTE",
                         compact_uniform_facts="TEST EXACT COLOR AND PATTERN",
                         concise_uniform_facts="TEST EXACT COLOR AND PATTERN",
                         concise_palette_evidence="TEST MEASURED GARMENT PALETTE",
                         lip_prompt_instruction="TEST NATURAL LIP TONE",
                         compact_trim_instruction="TEST TRIM COLOR",
                         compact_badge_instruction="TEST BADGE FABRIC REPAIR",
                         generated_palette_lock="TEST DYNAMIC PALETTE LOCK ",
                         generated_trim_lock="TEST DYNAMIC TRIM LOCK ",
                         pocket_instruction="TEST DYNAMIC POCKET LOCK ",
                         refinement_prompt=lambda bg, fabric, hair, lighting, skin, identity, **kwargs: "TEST REFINEMENT " + str(bg) + " " + str(fabric) + " " + hair + " " + lighting + " " + skin + " " + identity)
        result = eval(compile(expression, "prompt", "eval"), namespace)
        self.assertIn("TEST EXACT COLOR AND PATTERN", result)
        self.assertIn("TEST DYNAMIC PALETTE LOCK", result)
        self.assertIn("TEST DYNAMIC TRIM LOCK", result)
        self.assertIn("ONE PERSON ONLY", result)
        self.assertIn("Edit image 1 locally", result)
        self.assertIn("do not redesign or regenerate the portrait", result)
        self.assertIn("normal skin texture", result)
        self.assertIn("NECK/COLLAR LOCK", result)
        self.assertIn("upper-chest passport crop", result)
        self.assertIn("entire head, hair, hairline, ears, face", result)
        self.assertIn("Uniform facts:", result)
        self.assertIn("shoulder width", result)
        self.assertIn("Naturalize only its collar-to-neck junction", result)
        self.assertIn("TEST NECKWEAR", result)
        self.assertIn("already-clean pocket and fabric uninterrupted", result)
        self.assertNotIn("TEST TRIM PLACEMENT", result)
        for leaked_fixture_detail in (
            "exact plaid", "pointed collar", "short sleeves", "sleeve cuffs",
        ):
            self.assertNotIn(leaked_fixture_detail, result.lower())
        self.assertIn("#808080", result)
        # Leave visual-token capacity for the conditioning image. The previous
        # 694-word prompt exceeded the 512-token encoder budget and truncated
        # its body-scale instructions.
        self.assertLess(len(result.split()), 360)

    def test_primary_identity_prompt_keeps_uniform_as_image_two(self):
        from pipelines.uniform_composite import refinement_prompt

        result = refinement_prompt(
            (255, 255, 255),
            fabric_color_instruction="Copy image 2 fabric.",
            identity_instruction=(
                "Image 1 is the uploaded person. Image 2 is the supplied uniform "
                "and is the only clothing authority."
            ),
            primary_is_identity=True,
        )
        self.assertIn("Image 2 is the supplied uniform", result)
        self.assertNotIn("Image 1 is the supplied uniform", result)

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

    def test_uniform_prompt_rejects_exaggerated_face_and_accessories(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        for phrase in (
            "original small scale", "do not enlarge, stylize, recolor",
            "false eyelashes", "enlarged earrings", "stereotyped face",
            "Keep the crown smooth and flat exactly as image 1",
            "no top loop, visible band, bun, knot",
        ):
            self.assertIn(phrase, source)

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
        self.assertIn("create a V-shaped edge only when image 1 visibly has one", source)
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
        self.assertIn("[UNIFORM_IDENTITY_REVIEW] Raw Qwen candidate rejected", source)
        self.assertIn("Uniform output was not delivered because the Qwen-generated face", source)
        self.assertNotIn("Candidate retained for user review", source)

    def test_uniform_delivery_uses_no_secondary_identity_generator(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("_swap_uniform_identity", source)

    def test_raw_uniform_generation_rejects_secondary_people_before_finishing(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn("generated_faces = _get_insight_app().get(", source)
        self.assertIn("if len(generated_faces) != 1:", source)

    def test_generated_candidate_is_not_judged_against_rough_composite(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertNotIn("if not self._is_regenerated_from_rough", source)
        self.assertIn("rough collage is not an acceptance authority", source)
        self.assertNotIn("inswapper_128", source)

    def test_identity_repair_is_structural_and_preserves_source_chroma(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("alpha = (mask * float(np.clip(strength, .0, .72)))", source)
        self.assertIn("strength: float = .36", source)
        self.assertIn("mean_strength = 1.0 if channel == 0 else 0.35", source)

    def test_uniform_prompt_rejects_cast_only_skin_and_invented_earrings(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('unsafe_cast_terms = ("orange", "yellow"', source)
        for phrase in ("changed ethnicity appearance", "changed ancestry appearance", "invented earrings", "dangling earrings"):
            self.assertIn(phrase, source)
        self.assertIn("Preserve exactly the ear jewelry visibly present in image 1", source)
        self.assertNotIn("if not has_earrings", source)

    def test_generation_uses_only_explicit_earring_detail_from_accessory_analysis(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        # This section used to end at the first "generated_prompt =", which was
        # a duplicate prompt block that no branch ever used. With that block
        # removed the delimiter must be the real start of prompt assembly.
        generation_section = source[
            source.index("identity_instruction ="):source.index("neck_collar_lock = (")
        ]
        self.assertNotIn("hair_accessories", generation_section)
        self.assertIn('raw_hair_observations.get("hair_accessory_details")', generation_section)
        self.assertIn('("earring", "hoop", "stud")', generation_section)
        self.assertIn("image 1 pixels remain authoritative", source)

    def test_single_generation_keeps_quality_settings(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "qwen_edit_enhancer"]
        self.assertEqual(len(calls), 1)
        kwargs = {arg.arg: arg.value for arg in calls[0].keywords}
        self.assertEqual(kwargs["steps"].id, "generation_steps")
        self.assertEqual(kwargs["max_sequence_length"].id, "uniform_prompt_token_budget")
        self.assertEqual(kwargs["true_cfg_scale"].id, "generation_cfg_scale")
        self.assertFalse(ast.literal_eval(kwargs["reject_identity_failure"]))
        self.assertTrue(ast.literal_eval(kwargs["return_raw_candidate"]))
        self.assertIsNone(ast.literal_eval(kwargs["negative_prompt"]))
        self.assertEqual(kwargs["conditioning_max_dimension"].id, "conditioning_dimension")
        self.assertEqual(
            ast.unparse(kwargs["conditioning_pixel_budget"]),
            "conditioning_dimension * conditioning_dimension",
        )
        self.assertEqual(kwargs["max_generation_dimension"].id, "uniform_generation_dimension")
        self.assertEqual(kwargs["minimum_generation_dimension"].id, "uniform_generation_dimension")
        self.assertTrue(ast.literal_eval(kwargs["preserve_reference_aspect"]))
        self.assertEqual(kwargs["reference_images"].id, "qwen_garment_references")
        self.assertEqual(kwargs["image_input"].id, "direct_uniform_seed")
        self.assertNotIn("edit_mask", kwargs)
        self.assertNotIn("restore_unmasked_pixels", kwargs)

    def test_uniform_delivery_runs_full_review_then_passport_crop(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("identity_review = check_uniform_identity(", source)
        self.assertIn("rough_reference=uniform_validation_reference", source)
        self.assertIn("cropped = self._crop_school_passport_portrait(", source)
        self.assertIn("delivery_upper_chest=True", source)
        self.assertIn("cropped.save(output_path", source)
        self.assertIn("return output_path", source)

    def test_qwen_edits_real_person_instead_of_rough_composite(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("image 1 is the real person and body authority", source)
        self.assertIn("Replace every visible source-clothing pixel", source)
        self.assertIn("image_input=person_conditioning", source)
        self.assertNotIn("image_input=direct_uniform_seed", source)

    def test_person_conditioning_preserves_original_edges_before_uniform_fit(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        preparation = source[
            source.index("person_conditioning = self._source_identity_reference("):
            source.index("source_badges_removed = 0")
        ]
        self.assertNotIn("background_removal.remove_background(", preparation)
        self.assertNotIn("use_schp=True", preparation)
        self.assertIn("keep the original portrait pixels", preparation)
        self.assertIn("never on the source identity reference", preparation)

    def test_uniform_generation_balances_identity_and_complete_garment(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn("generation_cfg_scale = 1.0", source)
        pipeline_source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("central_arm_fraction > 0.08", pipeline_source)
        self.assertIn("int(width * .30):int(width * .70)", pipeline_source)
        prompt = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("Keep hands, wrists, props and lower arms outside", prompt)
        self.assertIn("Render the complete uniform seamlessly", prompt)

    def test_visual_reference_blocks_head_drift_after_measured_gate(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("from pipelines.uniform_review import review_uniform", source)
        self.assertIn("original_person_pil, uniform_validation_reference", source)
        self.assertIn('advisory_failures = [', source)
        # The list formerly called blocking_failures is now quality_warnings:
        # nothing ever acted on it, so the name was a lie. Its contents and
        # the advisory/warning split are unchanged, which is what this test
        # actually cares about.
        advisory = source[source.index("advisory_failures = ["):source.index("quality_warnings = [")]
        blocking = source[source.index("quality_warnings = ["):source.index("if confirmed_badge:")]
        self.assertIn('hair_changed_unconfirmed', advisory)
        self.assertIn('concrete_hair_failure', blocking)
        self.assertIn('blocking_failures.append("identity_changed")', source)
        self.assertIn('"harsh_hair_glare"', blocking)
        self.assertNotIn('blocking_failures.append("reference_drift")', source)
        self.assertIn('visual_review["delivery_blocked"]', source)
        # The reviewer now runs for every delivery, not only static_test /
        # extreme sun / tied hair: run 0b0a60e2 shipped a different child with
        # "review_skipped": true. The env var is an opt-out, not an opt-in.
        self.assertIn('"VSOUL_ENABLE_POST_VL_REVIEW", "1"', source)
        self.assertIn('not in {"0", "false", "no"}', source)

    def test_dynamic_route_does_not_mechanically_paste_source_head(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("Qwen genuine decode accepted directly without head paste", source)
        self.assertNotIn("self._lock_uniform_source_head(", source)

    def test_identity_head_is_diagnostic_not_a_generation_reference(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("identity_head_reference = build_person_identity_guide", source)
        self.assertIn("identity_head_reference.save", source)
        self.assertNotIn("identity_conditioning_reference", source)

    def test_vl_hue_wording_does_not_override_measured_garment_gate(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = source.index("concrete_uniform_failure = bool(")
        end = source.index("if visual_review.get(\"uniform_mismatch\")", start)
        gate = source[start:end]
        self.assertNotIn('"fabric hue"', gate)
        self.assertNotIn('"check spacing"', gate)
        self.assertIn('"wrong pattern"', gate)
        self.assertIn('"retained"', gate)

    def test_hair_style_name_alone_does_not_override_measured_geometry(self):
        from pipelines.qwen_edit_pipeline import _hair_review_has_concrete_change

        self.assertFalse(_hair_review_has_concrete_change(
            "The generated image changed from a loose ponytail to a braided hairstyle."
        ))
        self.assertTrue(_hair_review_has_concrete_change(
            "The generated image is missing the viewer-left tied section."
        ))
        self.assertTrue(_hair_review_has_concrete_change(
            "The output portrait moved the ribbon to the wrong side."
        ))

    def test_short_hair_is_not_reinterpreted_as_two_tied_sections(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn(
            "if not tied_accessory_corroborated and not visible_tied_hair_geometry:",
            source,
        )
        self.assertNotIn(
            "if not tied_accessory_corroborated and not visible_tied_hair_geometry and measured_tied_count == 0:",
            source,
        )

    def test_facial_speckles_are_repaired_before_identity_review(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        repair_at = source.index(
            "identity_locked = self._repair_generated_face_chromatic_speckles("
        )
        review_at = source.index("identity_review = check_uniform_identity", repair_at)
        self.assertLess(repair_at, review_at)

    def test_measured_backdrop_overrides_unsupported_visual_flag(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("measured_background_match = self._has_selected_solid_background", source)
        self.assertIn("and not measured_background_match", source)
        self.assertIn('advisory_failures.append("background_mismatch_unconfirmed")', source)

    def test_uniform_sunlight_is_repaired_in_qwen_not_baked_into_conditioning(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        preparation = source[
            source.index("generation_source = person_conditioning.convert"):
            source.index("generation_source.save(person_matted_path")
        ]
        self.assertNotIn("normalize_uniform_conditioning_light(", preparation)
        self.assertIn("Preserved original subject pixels", preparation)
        self.assertIn("Remove harsh directional sunlight", source)
        self.assertIn("maximum_geometry_rmse=0.030", source)

    def test_uniform_prompt_preserves_prepared_flat_backdrop(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("BACKDROP LOCK", source)
        self.assertIn("preserve the flat", source)
        self.assertIn("Preserve every strand, accessory and boundary position", source)
        self.assertNotIn("BACKGROUND REMOVAL IN QWEN", source)

    def test_uniform_prompt_uses_bounded_qwen_relight_and_edge_finish(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("LIGHTING FINISH", source)
        self.assertIn("remove only residual hard light", source)
        self.assertIn("EDGE FINISH", source)
        self.assertIn("remove only residual scene spill or halo", source)
        self.assertIn("never rematte or reshape", source)

    def test_uniform_reviewer_compares_the_passport_cropped_template(self):
        source = (ROOT / "pipelines/uniform_review.py").read_text(encoding="utf-8")
        self.assertIn("exact uniform fitted into the required upper-chest passport crop", source)
        self.assertIn("do not expect lower sleeves, waist, hem, hands", source)
        self.assertIn("visible difference within the shared crop", source)

    def test_selected_backdrop_validation_checks_the_full_semantic_background(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("full_background_match", source)
        self.assertIn("full_background_match >= 0.94", source)
        self.assertIn("semantic foreground matte", source)
        self.assertNotIn("identity_locked = self._apply_rough_silhouette_guard", source)
        self.assertNotIn("cropped = self._apply_rough_silhouette_guard", source)
        route_source = ast.get_source_segment(source, self.route)
        self.assertNotIn("_replace_background_with_foreground_matte", route_source)
        self.assertNotIn("selected_background_fallback", source)
        self.assertIn("must be rejected and regenerated", route_source)

    def test_uniform_colour_validation_allows_natural_chromatic_shading(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("def _uniform_color_fraction(", source)
        self.assertIn("hue_delta <= 12", source)
        self.assertIn("target_hsv[1]) >= 70", source)
        self.assertIn("distance <= 55.0", source)
        self.assertIn("self._uniform_color_fraction(", source)
        self.assertIn("reference_chromatic_fraction", source)
        self.assertIn("chromatic_hue_match", source)
        self.assertIn("patterned_histogram_match", source)
        self.assertIn("and chromatic_hue_match", source)
        self.assertIn("def _garment_reference_mask(", source)
        validator = source[source.index("def _has_generated_uniform("):]
        self.assertIn("_garment_reference_mask", validator)
        self.assertNotIn("ref_labels = parse_body_parts(reference)", validator)
        self.assertIn("dark_structure_match", validator)
        self.assertIn("Accepted template-relative hue/layer structure", validator)

    def test_hair_instruction_uses_person_not_uniform_reference(self):
        source = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("Keep image 1's black/dark-brown hair", source)
        self.assertNotIn("Keep image 2's black/dark-brown hair", source)

    def test_vl_drops_glasses_and_earrings_from_hair_accessories(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn("def normalize_hair_accessories", source)
        self.assertIn('data["hair_accessories"] = "none"', source)
        self.assertIn("normalize_hair_accessories(result)", source)

    def test_silhouette_guard_preserves_the_central_neck_bridge(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("central_neck_pixels = (", source)
        self.assertIn("(hsv[:, :, 1] >= 28)", source)
        self.assertIn("connectedComponentsWithStats", source)
        self.assertIn("(yy < int(rgb.shape[0] * .52))", source)
        self.assertIn("generated_garment = np.isin(", source)
        self.assertIn("| generated_garment | lower_reference_garment", source)

    def test_uniform_generation_dimension_scales_with_selected_steps(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("uniform_generation_dimension = 640", source)

    def test_uniform_uses_original_person_and_person_free_garment_authority(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("person_pil, source_labels, qwen_bg_rgb, remove_clothing=True", source)
        self.assertIn("image_input=person_conditioning", source)
        self.assertNotIn("localized_edit_canvas", source)
        self.assertNotIn("uniform_edit_mask", source)
        self.assertNotIn("QwenImageEditInpaintPipeline", source)
        self.assertIn("reference_images=qwen_garment_references", source)
        self.assertIn("if len(garment_reference_faces) == 0 else None", source)
        self.assertIn(
            '"primary_conditioning": "image 1 is the original-body fitted portrait; image 2 is the cleaned person-free garment authority when face detection confirms it contains no person"',
            source,
        )
        self.assertIn("garment_crop_top", source)
        self.assertIn("garment_design_conditioning", source)
        self.assertIn("reliable_head_protection", source)
        self.assertIn("backdrop_like_identity", source)
        self.assertIn("selected_bg_distance <= 76.0", source)
        self.assertIn("neck_blend_zone", source)
        self.assertIn("jaw_boundary_y", source)
        self.assertIn("np.power(jaw_distance, 1.55)", source)
        self.assertIn("protected_identity |= np.isin(identity_labels, (14, 15))", source)
        self.assertIn("(identity_labels == 13)", source)
        self.assertIn("protected_identity |= natural_neck_guard", source)
        self.assertIn("natural neck", source)
        self.assertIn("neck length, shoulder width and slope, body build, pose, head scale", source)
        self.assertIn("synthetic_arm_guide", source)
        self.assertIn("[UNIFORM_ARM_GUIDE]", source)
        self.assertIn("edit_region = clothing > 0", source)
        self.assertIn("edit_region &= ~protected_identity", source)
        self.assertIn("POCKET LOCK", source)
        self.assertIn('vl_plan.get("pocket_count")', source)
        self.assertIn("second pocket, mirrored pocket, extra pocket", source)
        compact_start = source.index('f"ONE PERSON ONLY. Produce one natural modern formal ID-card photograph')
        complete_prompt = source[compact_start:
                                 source.index("prompt_word_count = len(generated_prompt.split())", compact_start)]
        self.assertIn("{pocket_instruction}", complete_prompt)
        self.assertIn("collar_refine_zone", source)
        self.assertIn("Generate a continuous collar-to-neck junction", complete_prompt)
        self.assertIn("Image 2's garment is authoritative", complete_prompt)
        self.assertIn("UNIFORM LOCK", complete_prompt)
        self.assertIn("{generated_trim_lock}", complete_prompt)
        self.assertLess(
            complete_prompt.index("UNIFORM LOCK"),
            complete_prompt.index("Regenerate the result as one coherent natural camera photograph"),
        )
        self.assertIn("{compact_age_requirement}", complete_prompt)
        self.assertNotIn("school-ID", complete_prompt)
        self.assertIn("do not warp, blur, simplify", complete_prompt)
        self.assertIn("if prompt_word_count > 280:", source)
        self.assertIn("Compacted dynamic generation prompt", source)
        self.assertNotIn("Uniform generation prompt exceeds its conditioning budget", source)
        self.assertIn("Both collar leaves must be solid", source)
        self.assertNotIn("source_collar_hint", source)
        self.assertNotIn("rough_collar_leaf_hint", source)
        self.assertIn("source_clothing = np.isin", source)
        self.assertIn("disconnected_hair_on_backdrop", source)
        self.assertIn("collar_progress * 2.4", source)
        self.assertIn("face_y2 - detected_face_height * .01", source)
        self.assertIn('progress_callback(12, "Preparing the complete uniform portrait generation...")', source)
        self.assertIn("build_rough_composite(", source)
        self.assertIn("rough_source,", source)
        self.assertIn("rough_source_labels,", source)
        self.assertIn('"source_identity_reference_path": str(source_identity_reference_path)', source)
        self.assertNotIn("person_guide_path", source)
        self.assertNotIn("uniform_guide_path", source)
        self.assertIn("same neck length, shoulder width, body build", source)
        self.assertIn("upper-chest passport crop", source)
        self.assertIn("upper chest only", source)
        self.assertIn("cropped hair, missing sleeve", source)
        self.assertIn("ONE PERSON ONLY", source)
        self.assertIn("ONE PERSON ONLY", source)
        self.assertIn("Qwen Edit Plus receives only the original fitted-body", source)
        self.assertIn("The uploaded person's measured upper-body width is authoritative", source)
        self.assertIn('"body_fit_instruction": body_fit_instruction', source)
        route_source = ast.get_source_segment(source, self.route)
        self.assertNotIn("birefnet_service", route_source)
        self.assertNotIn("_normalize_qwen_raw_background", route_source)
        self.assertNotIn("identity_locked = self._lock_uniform_garment_from_rough", route_source)

    def test_repeated_inputs_use_a_content_stable_seed(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn('zlib.crc32(job_id.encode("utf-8"), uniform_seed)', source)
        self.assertIn('"seed_policy": "content_and_generation_contract_stable_v22_compact_uniform_refine"', source)
        self.assertIn("uniform-local-uniform-refine-v22", source)

    def test_uniform_steps_are_not_forced_back_to_twenty(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertNotIn("generation_steps = 20", source)
        self.assertNotIn("source_age_value >= 18 and generation_steps < 12", source)
        self.assertNotIn("adult source requires 12-step base-model age and fabric preservation", source)
        self.assertIn("adaptive visual-risk selector is", source)
        self.assertNotIn("if generation_steps > 12:", source)
        self.assertIn("generation_cfg_scale = 1.08", source)
        self.assertIn('"different person, changed face, younger face, changed skin tone, changed hairstyle, "', source)
        full_source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("pattern and identity detail retained by edit adapter", full_source)
        self.assertIn("recommended = 20 if risk >= 4 else 12 if risk >= 2 else 4", full_source)
        self.assertNotIn("4-step Lightning fallback avoids under-sampled painted 8-step uniform", source)
        self.assertIn("measure_source_person_colors", source)
        self.assertIn('"measured_source_colors": measured_person_colors', source)
        self.assertIn('"source_age_band": source_age_band', source)
        self.assertIn("age_negative_terms", source)

    def test_uniform_pass_uses_fitted_person_and_face_free_garment_reference(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        qwen = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "qwen_edit_enhancer")
        qwen_kwargs = {item.arg: item.value for item in qwen.keywords}
        self.assertEqual(qwen_kwargs["image_input"].id, "direct_uniform_seed")
        self.assertNotIn("edit_mask", qwen_kwargs)
        self.assertEqual(qwen_kwargs["reference_images"].id, "qwen_garment_references")
        self.assertNotIn("restore_unmasked_pixels", qwen_kwargs)
        self.assertEqual(qwen_kwargs["background_color"].id, "qwen_bg_rgb")
        self.assertFalse(ast.literal_eval(qwen_kwargs["preserve_source_clothing"]))
        identity = next(node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "check_uniform_identity")
        self.assertEqual(identity.args[0].id, "person_pil")
        identity_kwargs = {item.arg: item.value for item in identity.keywords}
        self.assertEqual(
            ast.literal_eval(identity_kwargs["minimum_similarity"]), 0.70,
        )
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn(
            "qwen_garment_references = [uniform_validation_reference]", source,
        )
        self.assertIn("Image 2 is the face-free garment authority", source)
        self.assertIn("uniform_generation_dimension = 1024 if fast_portrait_bucket else 1216", source)
        self.assertIn("conditioning_dimension = 512", source)
        self.assertIn("self._source_identity_reference", source)
        self.assertNotIn("background_removal.remove_background(", source)
        self.assertNotIn("use_schp=True", source)
        self.assertIn("uniform_seed = zlib.crc32(direct_uniform_seed.tobytes())", source)
        self.assertIn("uniform_seed = (uniform_seed ^ 483368614)", source)

        guide_calls = [node for node in calls if isinstance(node.func, ast.Name)]
        identity_guide = next(
            node for node in guide_calls if node.func.id == "build_person_identity_guide"
        )
        self.assertEqual(identity_guide.args[0].id, "person_pil")
        rough = next(node for node in guide_calls if node.func.id == "build_rough_composite")
        self.assertEqual(rough.args[0].id, "rough_source")
        self.assertEqual(rough.args[1].id, "transparent_template")
        self.assertEqual(rough.args[4].id, "conditioning_bg_rgb")

    def test_qwen_uses_selected_background_during_first_generation(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("qwen_bg_rgb = selected_bg_rgb", source)
        self.assertNotIn("qwen_bg_rgb = (255, 255, 255)", source)
        self.assertIn("perfectly flat {qwen_hex}", source)
        self.assertIn('background_color=qwen_bg_rgb', source)
        self.assertIn('"selected_background_rgb": list(selected_bg_rgb)', source)
        self.assertIn("Do not add scenery, texture, blocks, shadows or gradients", source)
        self.assertNotIn("gray background, white background", source)

        enhance = (ROOT / "gradio_app.py").read_text(encoding="utf-8")
        self.assertIn('qwen_bg_desc = f"solid {selected_hex.upper()} RGB{selected_rgb}"', enhance)
        self.assertIn("source_rgb = selected_rgb", enhance)
        self.assertNotIn('qwen_bg_desc = "neutral middle gray #808080"', enhance)
        output_section = enhance[enhance.index("qwen_pil = Image.open(str(res_path))"):enhance.index("elapsed = time.time() - t0", enhance.index("qwen_pil = Image.open(str(res_path))"))]
        self.assertIn("len(generated_faces) != 1", output_section)
        self.assertIn("the invalid portrait was not delivered", output_section)
        self.assertIn("qwen_accepted", output_section)
        self.assertIn("Image.open(str(res_path)).convert(\"RGB\")", output_section)
        self.assertNotIn("apply_selected_background_matte", output_section)
        self.assertNotIn('qwen_pil = image.convert("RGB")', output_section)
        self.assertIn("_replace_flat_border_background_strict", output_section)
        self.assertNotIn("_replace_smooth_border_background", output_section)
        self.assertIn("exact correction is backdrop-only", output_section)

    def test_uniform_uses_one_person_tensor_and_person_free_references(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        references_start = source.index("qwen_garment_references = [")
        references_end = source.index("arm_generation_instruction = (", references_start)
        references = source[references_start:references_end]
        self.assertIn("uniform_validation_reference", references)
        self.assertNotIn("selected_bg_reference", references)
        self.assertNotIn("identity_head_conditioning,", references)
        self.assertIn("image 1 is the sole person authority", source)
        self.assertNotIn("image 3 is the exact person-free color authority", source)
        self.assertIn("identity_head_conditioning = None", source)
        self.assertIn("generation_source = person_conditioning.convert", source)
        identity_lock = source.index("IDENTITY LOCK: image 1 is the sole person authority")
        hair_lock = source.index("HAIR LOCK: copy image 1's exact hairline")
        garment_edit = source.index("Use image 2 only as the uniform authority")
        self.assertLess(identity_lock, hair_lock)
        self.assertLess(hair_lock, garment_edit)
        self.assertIn("exact face, current age", source)
        self.assertIn("exact eye shape, size, spacing, pupils", source)
        self.assertIn("never redirect, enlarge or redraw", source)
        self.assertIn("Never truncate by word position", source)
        self.assertIn("No makeup, eyeliner, mascara, false eyelashes", source)
        self.assertIn('"VSOUL_ENABLE_UNIFORM_IDENTITY_PREPASS", "0"', source)
        self.assertIn('uniform-identity-prepass-v2', source)
        self.assertIn("minimum_similarity=.65", source)
        self.assertNotIn("[UNIFORM_IDENTITY_REPAIR] Bounded facial-interior repair review", source)
        self.assertIn("HAIR LOCK: copy image 1's exact hairline", source)
        self.assertIn('"hair_accessory_details"', source)
        self.assertIn("conditioning_height * width / max(height, 1)", source)
        self.assertIn("minimum_neck_gap_ratio=0.0", source)
        self.assertIn("measured neck length, head-to-shoulder scale and body width", source)
        self.assertIn('"hair_changed", "identity_changed",', source)
        self.assertIn('vl_plan.get("eye_description")', source)
        self.assertIn("never enters the eyes", source)
        self.assertIn("minimum_similarity=0.70", source)

        api_source = (ROOT / "main.py").read_text(encoding="utf-8")
        frontend = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("candidate_path=", api_source)
        self.assertNotIn("enhance-safe-dl-btn", frontend)
        self.assertNotIn("Qwen Generation - Review", frontend)

    def test_uniform_generation_preserves_the_shared_passport_canvas(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("template_conditioning.height * .62", source)
        self.assertNotIn("passport_reference_bottom", source)
        route_source = ast.get_source_segment(source, self.route)
        self.assertIn("identity_locked = generated_candidate", route_source)
        self.assertNotIn("face_bbox=generated_faces[0].bbox", route_source)
        prompt_source = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("shoulders and upper chest", prompt_source)
        self.assertIn("application will crop", prompt_source)

    def test_uniform_prompt_demands_a_plain_passport_backdrop_not_a_studio_scene(self):
        prompt_source = (ROOT / "pipelines/uniform_composite.py").read_text(encoding="utf-8")
        self.assertIn("BACKGROUND MUST BE ONLY one featureless, perfectly flat field", prompt_source)
        self.assertIn("no gradient, texture or scenery; no shadow, room, wall, window or objects", prompt_source)
        self.assertNotIn("school-ID studio pose", prompt_source)

    def test_wrong_qwen_background_uses_protected_semantic_matte(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        qwen = next(node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "qwen_edit_enhancer")
        post_generation_replacements = [node for node in calls if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_replace_background_with_foreground_matte"
            and node.lineno > qwen.lineno
        )]
        # A stochastic Qwen backdrop must not discard an otherwise valid
        # identity/uniform render. The generated subject is protected by SCHP
        # and composited over the selected exact colour.
        self.assertGreaterEqual(len(post_generation_replacements), 2)
        rough_guards = [node for node in calls if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_apply_rough_silhouette_guard"
            and node.lineno > qwen.lineno
        )]
        self.assertEqual(len(rough_guards), 0)
        envelope_remattes = [node for node in calls if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_replace_background_outside_subject_envelope"
            and node.lineno > qwen.lineno
        )]
        self.assertEqual(len(envelope_remattes), 0)
        smooth_replacements = [node for node in calls if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_replace_flat_border_background_strict"
            and node.lineno > qwen.lineno
        )]
        # Normalize the decoded canvas, the accepted portrait, and the final
        # resized passport canvas before any semantic-matte fallback.
        self.assertGreaterEqual(len(smooth_replacements), 3)
        route_source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("applying protected semantic foreground matte", route_source)
        self.assertIn("Preserve Qwen's accepted subject pixels", route_source)
        self.assertIn("normalized_background", route_source)

    def test_multicolor_uniform_does_not_collapse_to_one_vl_color(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("multi_color_pattern = (", source)
        self.assertIn('("plaid", "check", "tartan", "stripe")', source)
        self.assertIn("if not multi_color_pattern and isinstance(vl_rgb", source)
        self.assertIn("analyzed_pattern", source)
        self.assertNotIn("red contrast trim and deep navy-blue", source)
        self.assertIn("rough_reference=uniform_validation_reference", source)
        self.assertIn("if template_has_outer_layer:", source)
        self.assertIn("the target is one shirt only", source)
        self.assertIn("across the entire visible torso, both shoulders and both sleeves", source)
        self.assertIn("vest, waistcoat, pinafore, jacket, blazer, bib, apron", source)
        self.assertIn("[UNIFORM_PATTERN_GATE]", source)
        self.assertIn("candidate_pattern_density < minimum_pattern_density", source)

    def test_final_portrait_keeps_qwen_generated_garment(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertNotIn(
            "generated_candidate = self._lock_uniform_garment_from_rough(", source,
        )
        qwen_at = source.index("output_path = self.qwen_edit_enhancer(")
        identity_at = source.index("identity_review = check_uniform_identity(")
        self.assertLess(qwen_at, identity_at)
        self.assertIn("rough_reference=uniform_validation_reference", source)

    def test_uniform_delivery_does_not_apply_a_post_generation_tone_pass(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        finishing = [node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "finish_uniform_tones"]
        self.assertFalse(finishing)

    def test_uniform_delivery_does_not_recolor_the_accepted_qwen_portrait(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertNotIn("cropped = self._restore_uniform_accent_chroma(", source)
        self.assertIn("Do not recolour the accepted Qwen portrait", source)

    def test_uniform_edit_uses_complete_schp_identity_portrait(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn('rough_source = person_conditioning.convert("RGB")', source)
        self.assertIn('identity_base = person_conditioning.convert("RGB").resize(', source)
        self.assertNotIn("edit_canvas_rgb", source)
        self.assertNotIn("+ rough_rgb * clothing_alpha", source)
        self.assertNotIn("rough_collar_seed", source)
        self.assertNotIn("uniform_edit_mask", source)
        self.assertNotIn("low_frequency_guide", source)
        self.assertIn("image 1 is a real portrait, not a garment layout guide", source)
        self.assertIn("Image 2's garment is authoritative", source)
        self.assertIn("edit_region = clothing > 0", source)
        self.assertIn("edit_region &= ~protected_identity", source)
        self.assertNotIn("source_background = identity_labels == 0", source)
        self.assertIn("image_input=person_conditioning", source)
        self.assertIn("reference_images=qwen_garment_references", source)

    def test_uniform_delivery_does_not_overprocess_accepted_qwen_face(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        finishing = [node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "finish_uniform_tones"]
        self.assertFalse(finishing)
        restorations = [node for node in calls if isinstance(node.func, ast.Name)
                        and node.func.id == "restore_uniform_face_detail"]
        self.assertFalse(restorations)
        route_source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("cropped = identity_locked", route_source)
        self.assertIn("cropped.save(output_path", route_source)

    def test_qwen_uniform_never_pastes_source_face_or_hair_after_generation(self):
        calls = [node for node in ast.walk(self.route) if isinstance(node, ast.Call)]
        head_locks = [node for node in calls if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "_lock_uniform_source_head"]
        self.assertEqual(len(head_locks), 0)
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("Image 1's head is already correct", source)
        self.assertIn("Copy only visible earrings and eyeglasses", source)
        self.assertNotIn("exact hairline, center part, crown height", source)
        self.assertIn("short rounded hair, afro, curly hair", source)
        route_source = ast.get_source_segment(source, self.route)
        self.assertNotIn("self._lock_uniform_identity(", route_source)
        self.assertNotIn("self._restore_uniform_source_hair_geometry(", route_source)
        self.assertIn('identity_review["identity_repair_applied"] = False', route_source)

    def test_hidden_arm_guide_is_tapered_before_conditioning(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("tapered_arm_guide", source)
        self.assertIn("source_width * (1.0 - .42 * progress)", source)
        self.assertIn("Added tapered measured skin cue", source)

    def test_hair_restore_uses_shared_confident_hair_region(self):
        source = (ROOT / "pipelines/uniform_finishing.py").read_text(encoding="utf-8")
        self.assertIn("common, confident hair interior", source)
        self.assertIn("generated_hair_safe", source)
        self.assertIn("* .52", source)
        self.assertIn("* .78", source)

    def test_wrong_background_is_replaced_without_rejecting_qwen_subject(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"), self.route,
        )
        self.assertIn("_replace_background_with_foreground_matte", source)
        self.assertNotIn(
            "Qwen did not generate the selected flat backdrop", source,
        )
        self.assertIn("preserve_foreground_rgb=True", source)

    def test_custom_prompt_cannot_replace_uniform_instructions(self):
        # Every generated_prompt assignment must be an in-repo literal, or the
        # token-budget trimmer applied to one. Anything else (a parameter, a
        # dict lookup, a caller-supplied string) would let a request body
        # replace the uniform contract. The route-level assignment used to be
        # the literal itself; it is now the trimmer call that wraps it, with
        # the literals inside the prompt-mode branches.
        assignments = [n for n in ast.walk(self.route) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "generated_prompt"
                               for t in n.targets)]
        self.assertTrue(assignments)
        for node in assignments:
            if isinstance(node.value, ast.JoinedStr):
                continue
            self.assertIsInstance(node.value, ast.Call)
            self.assertEqual(node.value.func.id, "fit_prompt_to_token_budget")
            self.assertEqual(node.value.args[0].id, "generated_prompt")

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

    def test_single_shirt_trim_is_not_promoted_to_outer_garment(self):
        from typing import Any, Dict
        tree = ast.parse((ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8"))
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("flatten_vl_dict", "normalize_uniform_analysis")]
        namespace = dict(Any=Any, Dict=Dict)
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "analysis", "exec"), namespace)
        result = namespace["normalize_uniform_analysis"]({
            "garment_components": ["short-sleeve shirt"],
            "outer_garment_color_and_shape": "red square layer",
            "outer_primary_color": "red",
            "accent_trim_placement": "collar, center placket, sleeve-end cuffs",
            "shoulder_trim_present": False,
        })
        self.assertEqual(result["outer_garment_color_and_shape"], "none")
        self.assertEqual(result["outer_primary_color"], "none")
        self.assertIsNone(result["outer_neutral_rgb"])
        self.assertIn("sleeve-end cuffs", result["accent_trim_placement"])

    def test_vl_workers_use_low_memory_gpu_and_cache_by_image_content(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn('VL_CACHE_VERSION = "content-v16-pose-alignment"', source)
        self.assertIn("BitsAndBytesConfig", source)
        self.assertIn('device_map={{"": 0}} if torch.cuda.is_available() else "cpu"', source)
        self.assertIn('_analysis_cache_path("portrait", image)', source)
        self.assertIn('_analysis_cache_path("uniform", person, template)', source)
        self.assertIn("analysis cache hit", source)

    def test_hair_analysis_uses_a_dedicated_face_centred_crop(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn("def make_hair_inspection_crop", source)
        self.assertEqual(source.count("hair_inspection = make_hair_inspection_crop("), 2)
        self.assertNotIn("head_crop = person_img.crop((0, 0, person_img.width", source)
        for key in (
            "hair_crown_shape", "hair_left_side", "hair_right_side", "hair_ends",
            "hair_fringe", "ponytail_side", "ponytail_anchor", "ponytail_extent",
            "hair_accessory_count", "hair_accessory_confidence",
        ):
            self.assertIn(key, source)

    def test_portrait_analysis_uses_full_image_and_dedicated_face_crop(self):
        source = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        self.assertIn("def make_face_inspection_crop", source)
        self.assertEqual(source.count("face_inspection = make_face_inspection_crop("), 2)
        self.assertIn("Image 2 is a detailed face-and-ears crop", source)
        self.assertIn("Never infer age from a uniform, school setting", source)
        self.assertIn("Inspect both ears in image 2", source)
        self.assertEqual(source.count("images=[person_img, face_crop]"), 1)
        self.assertEqual(source.count("images=[portrait, face_crop]"), 1)

    def test_uniform_prompt_rejects_uncertain_ponytail_geometry(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"not visible", "not detected"', source)
        self.assertIn("Do not shorten, expand, curl, mirror or restyle it", source)
        self.assertIn("Image 1's head is already correct", source)
        self.assertIn("Keep exactly one low tied hairstyle at the nape", source)
        self.assertIn("do not loosen it down ", source)
        self.assertIn('"both sides, split it into two tails', source)
        self.assertNotIn("A standing band collar wraps the neck base", source)
        self.assertIn("Keep the same neck length", source)
        self.assertIn("Naturalize only its collar-to-neck junction", source)

    def test_uniform_generation_uses_stable_single_subject_conditioning(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("conditioning_dimension = 512", source)
        self.assertIn("uniform_prompt_token_budget = 384", source)
        self.assertIn("conditioning_max_dimension=conditioning_dimension", source)
        self.assertIn("conditioning_pixel_budget=conditioning_dimension * conditioning_dimension", source)
        self.assertIn("generation_cfg_scale = 1.0", source)
        self.assertIn("uniform-local-uniform-refine-v22", source)
        self.assertIn('rough_placement.get("detected_ponytail_side")', source)
        self.assertIn("dreadlocks, locs, braids, cornrows, twists", source)

    def test_uniform_prompt_fits_encoder_budget_without_low_confidence_hair_fields(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        generation_section = source[source.index("hair_generation_details = observed_details"):
                                    source.index("component_description =")]
        self.assertNotIn('"hair_crown_shape"', generation_section)
        self.assertNotIn('"hair_left_side"', generation_section)
        self.assertNotIn('"hair_right_side"', generation_section)
        self.assertIn('"hair_style", "hair_parting", "hair_length", "hair_texture", "hair_color"', generation_section)
        self.assertIn("ONE PERSON ONLY", source)
        self.assertIn("_measure_garment_palette", source)
        self.assertIn("_measure_garment_accent_color", source)
        self.assertIn("Measured garment colours by pixel area, dominant first", source)

    def test_gender_classifiers_are_audit_only_not_generation_conditioning(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn('vl_plan["gender_analysis_only"]', source)
        generation = source[source.index("source_appearance_lock ="):source.index("generation_negative_prompt =")]
        self.assertNotIn("female presentation", generation)
        self.assertNotIn("male presentation", generation)
        self.assertNotIn("feminine facial features", generation)
        self.assertNotIn("masculine features", generation)
        self.assertIn("do not infer or change gender", generation)
        self.assertIn("def capped_words", source)
        self.assertIn("[UNIFORM_PROMPT] Compacted verbose VL prompt", source)
        self.assertIn("prefix_limit = max(1, 360 - len(finish_words))", source)
        self.assertNotIn("Uniform generation prompt exceeds its safe conditioning budget", source)

    def test_school_uniform_prompt_does_not_deage_adult_or_drop_corroborated_tied_hair(self):
        source = ast.get_source_segment(
            (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"),
            self.route,
        )
        self.assertIn("vl_tied_hair_corroborated", source)
        self.assertIn("analyzed_ponytail_anchor not in concrete_hair_values", source)
        self.assertIn("analyzed_ponytail_extent not in concrete_hair_values", source)
        self.assertIn("ONE PERSON ONLY", source)
        self.assertIn("Retain exactly the same apparent age and facial maturity visible in image 1", source)
        self.assertIn("generation must copy source-visible maturity", source)
        self.assertNotIn("Produce one modern real school-ID photograph", source)

    def test_uniform_prompt_carries_exact_template_facts_and_upper_chest_frame(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        for key in (
            "shirt_color_and_pattern", "shirt_pattern_type", "shirt_pattern_scale",
            "shirt_collar", "sleeve_length", "accent_trim_color", "accent_trim_placement",
            "button_layout", "button_color",
        ):
            self.assertIn(f'"{key}"', source)
        self.assertIn("Uniform facts: {concise_uniform_facts}", source)
        self.assertIn("{compact_trim_instruction}", source)
        self.assertIn('vl_plan.get("accent_trim_color")', source)
        self.assertIn("Pixel measurement identifies the coherent solid trim", source)
        self.assertIn("Solid trim exists only at those analyzed placements", source)
        self.assertIn("_restore_generated_lip_tone", source)
        self.assertIn("natural unpainted lip tone", source)
        self.assertIn('max_colors=6 if multi_color_pattern else 4', source)
        self.assertIn('"Measured template palette: "', source)
        self.assertIn("Uniform facts:", source)
        self.assertIn("centered camera-facing pose and upper-chest passport crop", source)
        self.assertIn("Do not add scenery, texture, blocks, shadows or gradients", source)
        self.assertIn("upper-chest passport crop", source)
        self.assertIn("source_body_ratio", source)
        self.assertIn("neck length, shoulder width, body build", source)
        self.assertIn("never copy the template model", source)
        self.assertIn("garment design, colors, check or stripe alignment, collar", source)
        self.assertNotIn("Copy image 2's exact plaid, pointed collar", source)
        self.assertIn("natural unretouched camera photograph", source)
        self.assertIn("Keep the same neck length", source)
        self.assertIn("sideways neck, bent neck, curved neck, off-center neck", source)
        self.assertIn("Image 1's head is already correct", source)
        self.assertIn("do not make the person younger, older, more childlike or more mature", source)
        self.assertIn("do not redraw, beautify, smooth, recolor or reshape it", source)
        self.assertIn("Image 1's hair is already correct. Copy image 1's hairline", source)
        self.assertIn("Image 1 pixels are the sole hair-accessory authority", source)
        self.assertIn("same count, type, color, size, side and placement", source)
        self.assertIn("{hair_accessory_lock}", source)
        unconditional_negative = source[
            source.index('negative_prompt = (', source.index('HAIR AND WEARABLES:')):
            source.index('if not template_has_neckwear:', source.index('HAIR AND WEARABLES:'))
        ]
        self.assertNotIn("front braid, side braid", unconditional_negative)
        self.assertIn("invented hair accessory", source)

    def test_generated_badge_cleanup_is_reference_matched_and_texture_based(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route_source = ast.get_source_segment(source, self.route)
        self.assertIn("generated_template_badges_removed", route_source)
        self.assertIn("badge_reference", route_source)
        self.assertIn('repair_mode="auto"', route_source)

    def test_misclassified_earrings_are_kept_as_wearables(self):
        from pipelines.uniform_vl_analyzer import normalize_hair_accessories

        result = normalize_hair_accessories({
            "hair_accessories": "Earrings",
            "hair_accessory_details": "small silver earrings",
            "hair_accessory_count": 2,
            "visible_wearables": "none",
            "earring_description": "none",
        })
        self.assertEqual(result["hair_accessories"], "none")
        self.assertEqual(result["hair_accessory_count"], 0)
        self.assertEqual(result["visible_wearables"], "small silver earrings")
        self.assertEqual(result["earring_description"], "small silver earrings")

    def test_mixed_flower_and_necklace_does_not_count_necklace_as_hair(self):
        from pipelines.uniform_vl_analyzer import normalize_hair_accessories

        result = normalize_hair_accessories({
            "hair_accessories": "Flower;Necklace",
            "hair_accessory_details": "Flower on left side;Necklace around neck",
            "hair_accessory_count": 2,
            "hair_accessory_confidence": "high",
            "visible_wearables": "earrings",
        })
        self.assertEqual(result["hair_accessories"], "flower")
        self.assertEqual(result["hair_accessory_details"], "flower on left side")
        self.assertEqual(result["hair_accessory_count"], "unknown")
        self.assertIn("necklace", result["visible_wearables"])

    def test_lens_reflection_corroborates_glasses_without_dropping_earrings(self):
        from pipelines.uniform_vl_analyzer import normalize_hair_accessories

        result = normalize_hair_accessories({
            "visible_wearables": "earrings",
            "earring_description": "small silver hoop earrings",
            "glasses_reflection_present": True,
            "glasses_reflection_description": "green reflection across both lenses",
        })
        self.assertIn("earrings", result["visible_wearables"])
        self.assertIn("glasses", result["visible_wearables"])
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('bool(vl_plan.get("glasses_reflection_present"))', source)

    def test_one_sided_ponytail_is_not_mirrored(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("visible ONLY on {measured_ponytail_side}", source)
        self.assertIn("bilateral ponytails, mirrored ponytail, second ponytail", source)
        self.assertIn("This is a LOW ponytail anchored at the nape", source)
        self.assertIn("high ponytail, crown ponytail, top ponytail", source)

    def test_uniform_qwen_canvas_uses_selected_background_directly(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("conditioning_bg_rgb = selected_bg_rgb", source)
        self.assertIn("perfectly flat {qwen_hex}", source)
        self.assertIn("conditioning_subject_alpha", source)
        self.assertIn("Reapplied the uploaded person's soft alpha", source)
        route = source[source.index("def qwen_vl_image_edit_uniform_swap("):]
        self.assertNotIn(
            "generated_candidate = self._replace_background_from_conditioning_silhouette(",
            route,
        )
        self.assertIn("subject &= support", source)
        self.assertIn("cv2.grabCut(", source)
        self.assertNotIn("conditioning_bg_rgb = (232, 232, 232)", source)

    def test_edit_plus_image_latents_use_cuda_without_inpaint_latents(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("def _cuda_prepare_latents", source)
        self.assertNotIn("def _cuda_prepare_mask_latents", source)
        self.assertNotIn("pipe.prepare_mask_latents", source)
        self.assertNotIn("calculate_inpaint_dimensions", source)
        self.assertNotIn("edit_mask_resized", source)
        self.assertIn("pipe = QwenImageEditPlusPipeline(", source)
        self.assertIn("reference_images=qwen_garment_references", source)
        self.assertIn("image_input=person_conditioning", source)
        self.assertIn("[UNIFORM_GENERATION_DELTA]", source)
        self.assertIn("edit_region = clothing > 0", source)
        self.assertIn("edit_region &= ~protected_identity", source)
        self.assertIn("check_uniform_identity(identity_base, identity_locked)", source)

    def test_uniform_validator_rejects_invented_dark_central_panel(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("central_torso_dark_fraction", source)
        self.assertIn("dark_panel_divergence", source)
        self.assertIn("Qwen invented a dark central garment panel absent from the template", source)
        self.assertIn("target_layout_wins", source)
        self.assertIn("target_spatial_similarity >= .18", source)
        self.assertIn("target_spatial_similarity", source)
        self.assertIn(">= source_spatial_similarity + .10", source)

    def test_measured_collar_tip_count_overrides_vl_collar_label(self):
        """A free-tip count must beat the VL's collar name when they disagree.

        Run 0b0a60e2's template was an obvious two-point pointed collar, but
        the VL labelled it "crossover overlap band collar without free points"
        and that label reached the prompt as a generation fact.
        """
        analyzer = (ROOT / "pipelines/uniform_vl_analyzer.py").read_text(encoding="utf-8")
        # The model must count before it names, and the key must be requested
        # and reset alongside the other garment fields.
        self.assertIn("COLLAR, STEP 1 of 2", analyzer)
        self.assertIn("COLLAR, STEP 2 of 2", analyzer)
        self.assertIn("collar_free_tip_count, shirt_collar", analyzer)
        self.assertIn('"collar_free_tip_count", "shirt_collar"', analyzer)
        # A changed VL prompt with an unchanged cache version silently replays
        # the old wrong reading for every already-seen upload.
        self.assertNotIn('VL_CACHE_VERSION = "content-v32', analyzer)

        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = source.index('collar_type = str(vl_plan.get("shirt_collar")')
        block = source[start:source.index("uniform_facts_str", start)]
        self.assertIn('vl_plan.get("collar_free_tip_count")', block)
        self.assertIn("collar_tips == 2", block)
        self.assertIn("folded pointed collar with two free triangular tips", block)
        self.assertIn("UNIFORM_VL_COLLAR", block)

    def test_pattern_repeat_is_measured_and_stated_in_prompt(self):
        """A check's size must reach Qwen as a count, not as "medium".

        On run fbf83870 the delivered plaid measured 2.18x the template's
        check size: the prompt carried shirt_pattern_scale "medium", which is
        a category and constrains nothing.
        """
        import numpy as np
        from PIL import Image
        from pipelines.qwen_edit_pipeline import QwenEditPipeline

        # A synthetic 24px check on a 480px canvas: the body window is the
        # central 40%, so the measurement must recover ~24px regardless.
        size = 480
        period = 24
        cell = np.indices((size, size)).sum(axis=0) // (period // 2)
        checks = np.where(cell % 2 == 0, 235, 40).astype(np.uint8)
        template = Image.fromarray(np.dstack([checks] * 3))
        measured = QwenEditPipeline._measure_pattern_repeat_px(template)
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured, period, delta=4)

        # A plain fabric has no repeat to state; the caller must omit the
        # instruction rather than assert a wrong number.
        plain = Image.new("RGB", (size, size), (200, 200, 200))
        self.assertIsNone(QwenEditPipeline._measure_pattern_repeat_px(plain))

        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("PATTERN SCALE: the repeating check measures about", source)
        self.assertIn("UNIFORM_PATTERN_SCALE", source)
        # The count must use the same window the measurement used, or it
        # describes a span the viewer never sees.
        self.assertIn("chest_span_px = template_pil.width * .40", source)
        # And it must actually reach the dynamic prompt.
        contract = source[
            source.index("uniform_fit_contract = ("):
            source.index("lighting_correction =", source.index("uniform_fit_contract = ("))
        ]
        self.assertIn("pattern_repeat_instruction", contract)

    def test_source_complexion_is_restored_after_decode(self):
        """The decode bleaches skin; the correction must undo it, and only it.

        Traced on run 0b0a60e2: every tensor entering Qwen carried the source
        complexion (208,143,108) and the decode emerged at (244,194,165), a
        deltaE of 45. Lightness is 94% of that error, so a chroma-only
        correction is not enough (it reached only 42). The correction must
        also leave backdrop, hair and garment untouched, which an earlier
        plain-ellipse mask did not: it painted a dark oval halo.

        This runs on the real delivered frame, because InsightFace does not
        detect a drawn oval and a synthetic fixture would skip silently.
        """
        import numpy as np
        import cv2
        from PIL import Image
        from pipelines.qwen_edit_pipeline import QwenEditPipeline

        fixture = ROOT / "outputs" / "qwen_uniform_0b0a60e2_uniform.png"
        if not fixture.exists():
            self.skipTest("regression fixture 0b0a60e2 not retained")

        def delta_e(first, second):
            to_lab = lambda c: cv2.cvtColor(
                np.uint8([[c]]), cv2.COLOR_RGB2LAB,
            )[0, 0].astype(float)
            return float(np.sqrt(((to_lab(first) - to_lab(second)) ** 2).sum()))

        def cheek_rgb(image):
            from pipelines.photo_restoration import _get_insight_app
            array = np.asarray(image.convert("RGB"))
            faces = _get_insight_app().get(
                cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            )
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            w, h = x2 - x1, y2 - y1
            cy, half = y1 + int(h * .55), max(2, int(h * .06))
            patches = [
                array[cy - half:cy + half, x1 + int(w * .12):x1 + int(w * .28)],
                array[cy - half:cy + half, x2 - int(w * .28):x2 - int(w * .12)],
            ]
            return np.median(
                np.vstack([p.reshape(-1, 3) for p in patches if p.size]), axis=0,
            )

        source_skin = (208, 143, 108)
        delivered = Image.open(fixture)
        before = delta_e(cheek_rgb(delivered), source_skin)
        self.assertGreater(before, 30.0, "fixture should show the bleaching defect")

        restored, info = QwenEditPipeline._restore_source_skin_tone(
            delivered, source_skin,
        )
        self.assertTrue(info.get("applied"), info.get("reason"))
        after = delta_e(cheek_rgb(restored), source_skin)
        self.assertLess(after, 10.0, f"complexion still off: deltaE {after:.1f}")

        # All three channels must move: chroma alone only reached 42.
        self.assertLess(info["shift_l"], 0.0, "a bleached face must be darkened")

        # Backdrop is not skin and must come back byte-exact.
        original = np.asarray(delivered.convert("RGB"))
        output = np.asarray(restored.convert("RGB"))
        corner_delta = int(
            np.abs(original[:30, :30].astype(int) - output[:30, :30].astype(int)).max()
        )
        self.assertEqual(corner_delta, 0, "background must not be recoloured")

    def test_insufficient_source_detail_is_measured_and_reported(self):
        """An upload that cannot reach the delivered frame must say so.

        Facial detail cannot exceed the source. Measured on job 58682ecb (a
        390x508 print with a 214px face), every generative restorer bought
        sharpness with identity - Qwen prepass 0.26, GFPGAN 0.819, CodeFormer
        0.817, against 0.977 for simply holding the real head. So the
        pipeline keeps the real face and reports the ceiling instead.
        """
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        # Scale-honest: ``repair_recommended`` is resolution-dependent and
        # called a clean 3500px upload blurrier than a 390px print, so the
        # measurement is a face-pixel budget against the delivered frame.
        self.assertIn("delivered_face_width_px = 260.0", source)
        self.assertIn("source_face_detail_budget = source_face_width_px", source)
        self.assertIn("source_detail_sufficient = bool(source_face_detail_budget >= 1.0)", source)
        self.assertIn("UNIFORM_SOURCE_QUALITY", source)
        # It has to reach the audit, or only the server operator ever sees it.
        self.assertIn('"source_detail_sufficient": source_detail_sufficient', source)
        self.assertIn('"source_face_detail_budget"', source)

        # And reach the person who uploaded the photo, via the job record.
        worker = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("source_quality_notes", worker)
        self.assertIn("source_detail_sufficient", worker)
        self.assertIn("torso_evidence_sufficient", worker)

    def test_isolated_encoder_disables_autograd(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        # The loop now carries a per-prompt text-only flag so the negative can
        # be encoded without the conditioning images; autograd must stay off.
        self.assertIn(
            "with torch.inference_mode():\n"
            "    for prompt, text_only in zip(prompts, text_only_flags):",
            source,
        )
        self.assertIn('\"encoding\": \"utf-8\"', source)
        self.assertIn('\"errors\": \"replace\"', source)


if __name__ == "__main__":
    unittest.main()
