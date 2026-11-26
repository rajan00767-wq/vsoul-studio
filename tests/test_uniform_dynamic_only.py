import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DynamicUniformOnlyTests(unittest.TestCase):
    def setUp(self):
        self.pipeline_source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(self.pipeline_source)
        self.route = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_vl_image_edit_uniform_swap"
        )
        self.route_source = ast.get_source_segment(self.pipeline_source, self.route)

    def test_uniform_generator_exposes_no_static_or_manual_prompt_mode(self):
        arguments = [arg.arg for arg in self.route.args.args]
        self.assertNotIn("prompt", arguments)
        self.assertNotIn("prompt_mode", arguments)
        self.assertNotIn("static_test", self.route_source)
        self.assertNotIn("_build_universal_static_uniform_prompt", self.pipeline_source)

    def test_uniform_generator_uses_one_vl_analysis_and_one_qwen_edit(self):
        self.assertEqual(self.route_source.count("uniform_vl_analyzer.analyze("), 1)
        self.assertEqual(self.route_source.count("self.qwen_edit_enhancer("), 1)
        self.assertIn('vl_plan.get("source") != "qwen2.5-vl"', self.route_source)

    def test_diffusion_receives_identity_only_person_and_one_garment_reference(self):
        self.assertIn("identity_only_authority = (", self.route_source)
        self.assertIn("generation_primary = Image.fromarray(identity_only_rgb", self.route_source)
        self.assertIn("identity_only_upper_chest_region = (", self.route_source)
        self.assertIn("neck_collar_seam_region = (", self.route_source)
        self.assertIn("edit_region |= neck_collar_seam_region", self.route_source)
        self.assertIn("| neck_collar_seam_region", self.route_source)
        self.assertIn("source_garment_authority = (", self.route_source)
        self.assertIn("identity_only_rgb = primary_rgb.copy()", self.route_source)
        self.assertIn("identity_only_rgb[garment_removal_region]", self.route_source)
        self.assertNotIn("identity_only_rgb = np.empty_like(primary_rgb)", self.route_source)
        self.assertIn(
            "never a generative pixel authority",
            self.route_source,
        )
        self.assertNotIn("generation_primary = generation_source", self.route_source)
        self.assertIn(
            "qwen_garment_references = [uniform_validation_reference]",
            self.route_source,
        )
        self.assertNotIn(
            "qwen_garment_references.append(fitted_garment_reference)",
            self.route_source,
        )
        self.assertIn("Using compact dynamic VL contract", self.route_source)

    def test_neck_seam_conditioning_repairs_backdrop_holes_dynamically(self):
        self.assertIn("seam_seed_holes = neck_collar_seam_region", self.route_source)
        self.assertIn("seed_bg_distance <= 38.0", self.route_source)
        self.assertIn("if collar_leaves_are_solid and measured_accent", self.route_source)
        self.assertIn('measured_accent["rgb"]', self.route_source)
        self.assertIn("cv2.inpaint(", self.route_source)
        self.assertIn("[UNIFORM_NECK_SEAM]", self.route_source)

    def test_fitted_uniform_prompt_forbids_duplicate_layers(self):
        self.assertIn("Keep one shirt and one sleeve per side", self.route_source)
        self.assertIn("second cuff or duplicate layer", self.route_source)
        self.assertIn("IMAGE 2 supplies fabric and construction only", self.route_source)

    def test_uniform_replacement_does_not_latent_hold_source_clothing(self):
        tree = ast.parse(self.pipeline_source)
        enhancer = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_edit_enhancer"
        )
        enhancer_source = ast.get_source_segment(self.pipeline_source, enhancer)
        self.assertIn("if preserve_source_clothing:", enhancer_source)
        self.assertIn(
            "source clothing is not latent-held",
            enhancer_source,
        )
        self.assertIn("preserve_source_clothing=False", self.route_source)

    def test_dynamic_prompt_keeps_person_and_garment_authorities_separate(self):
        self.assertIn("IMAGE 2 supplies fabric and construction only", self.route_source)
        self.assertIn("IMAGE 1 is the sole person authority", self.route_source)
        self.assertIn("image 2 uniform", self.route_source)
        self.assertNotIn("Preserve image 1's exact multicolor fabric palette", self.route_source)

    def test_uniform_conditioning_uses_decontaminated_fused_matte(self):
        self.assertIn("conditioning_birefnet_rgba = None", self.route_source)
        self.assertIn("Using fused BiRefNet+SCHP alpha", self.route_source)
        self.assertIn("if conditioning_birefnet_rgba is not None:", self.route_source)

    def test_garment_edit_has_dynamic_edge_band_and_looser_passport_crop(self):
        self.assertIn("garment_edge_px = max(7", self.route_source)
        self.assertIn("edit_region |= garment_edge_band & garment_edge_corridor", self.route_source)
        self.assertIn("delivery_face_fraction if delivery_face_fraction is not None else .405", self.pipeline_source)
        self.assertIn("dynamic_delivery_face_fraction = .445", self.route_source)
        self.assertIn("delivery_has_extended_hair", self.route_source)
        self.assertIn("delivery_face_fraction=dynamic_delivery_face_fraction", self.route_source)

    def test_garment_edge_redraw_is_bounded_by_measured_person_width(self):
        self.assertIn(
            "measured_body_half_width = detected_face_width * source_body_ratio * .5",
            self.route_source,
        )
        self.assertIn(
            "garment_edit_half_width = measured_body_half_width + garment_seam_allowance",
            self.route_source,
        )
        self.assertIn(
            "np.abs(mask_x - face_center_x) <= garment_edit_half_width",
            self.route_source,
        )
        self.assertNotIn(
            "np.abs(mask_x - face_center_x) <= detected_face_width * 1.55",
            self.route_source,
        )
        self.assertIn(
            "edit_region &= ~(lower_uniform_zone & ~measured_body_corridor)",
            self.route_source,
        )
        self.assertIn(
            "source_body_ratio = float(np.clip(raw_body_ratio, 1.35, 2.80))",
            self.route_source,
        )

    def test_lower_hair_uses_soft_hold_while_accessories_remain_hard_held(self):
        self.assertIn("hard_protect_region_mask", self.route_source)
        self.assertIn(
            "hard_protect_region_mask=generation_hard_identity_protection",
            self.route_source,
        )
        self.assertIn("complete_head_guard |= side_hair_guard", self.route_source)
        self.assertIn("edit_region &= ~complete_head_guard", self.route_source)
        self.assertNotIn("hard_identity_protection |= side_hair_guard", self.route_source)
        self.assertIn("hard_identity_protection |= detected_acc_mask_canvas", self.route_source)
        self.assertIn(
            "upload_confirmed_clothing | source_collar_residual",
            self.route_source,
        )
        self.assertIn(
            "hard_identity_protection &= ~hard_garment_overlap",
            self.route_source,
        )
        hard_at = self.route_source.index("hard_identity_protection = np.zeros_like")
        hard_block = self.route_source[hard_at:hard_at + 900]
        self.assertNotIn("hard_identity_protection |= neck_skin_corridor", hard_block)
        self.assertNotIn("hard_identity_protection |= natural_neck_guard", hard_block)

    def test_target_collar_is_locally_seeded_at_measured_neck_seat(self):
        self.assertIn("collar_conditioning_region = (", self.route_source)
        self.assertIn("& ~neck_skin_corridor", self.route_source)
        self.assertIn("direct_uniform_seed.resize", self.route_source)
        self.assertIn("[UNIFORM_COLLAR_CUE]", self.route_source)
        self.assertIn("collar_tip_rise = (", self.route_source)
        self.assertIn('("pointed", "folded", "lapel")', self.route_source)

    def test_uniform_delivery_never_copies_source_pixels_over_qwen_neck(self):
        self.assertNotIn(
            "identity_locked = self._bridge_neck_collar_gap(",
            self.route_source,
        )
        self.assertIn(
            "coherent Qwen-generated neck and collar junction",
            self.route_source,
        )

    def test_source_neck_measurement_is_used_without_lengthening_floor(self):
        self.assertIn(
            "premeasured_source_neck_ratio = self.measure_exposed_neck(",
            self.route_source,
        )
        self.assertIn(
            "target_neck_gap_ratio=premeasured_source_neck_ratio",
            self.route_source,
        )
        self.assertIn(
            "direct_source_neck_ratio = premeasured_source_neck_ratio",
            self.route_source,
        )
        self.assertIn(
            'rough_placement["neck_seat_ratio"] = measured_neck_gap_ratio',
            self.route_source,
        )
        self.assertIn("np.clip(raw_neck_ratio, 0.04, 0.32)", self.route_source)
        self.assertNotIn("np.clip(raw_neck_ratio, 0.16, 0.32)", self.route_source)

    def test_patterned_uniform_does_not_force_pattern_onto_solid_collar(self):
        self.assertIn('"accent_trim_color", "accent_trim_placement"', self.route_source)
        self.assertIn("collar_uses_pattern_with_piping", self.route_source)
        self.assertIn(
            "both collar leaves are solid accent fabric with no body pattern",
            self.route_source,
        )
        self.assertIn("accent_fraction >= 0.14", self.route_source)
        self.assertIn("largest_component_fraction >= 0.045", self.route_source)
        self.assertIn("[UNIFORM_COLLAR_SURFACE]", self.route_source)
        self.assertIn(
            "if collar_uses_pattern_with_piping else",
            self.route_source,
        )
        self.assertNotIn(
            "if (patterned_uniform or multi_color_pattern) else",
            self.route_source,
        )

    def test_exact_garment_facts_precede_body_and_hair_contracts(self):
        facts_at = self.route_source.index("Exact observed garment facts:")
        body_at = self.route_source.index("BODY FIT: use the measured")
        self.assertLess(facts_at, body_at)
        self.assertIn("retained_tied_geometry_lock", self.route_source)
        self.assertIn("Preserve every separately visible source hair accessory", self.route_source)
        self.assertIn("image 1—not a numeric ", self.route_source)

    def test_low_detail_head_is_regenerated_as_one_coherent_region(self):
        self.assertIn(
            "regenerate_low_detail_head = not source_detail_sufficient",
            self.route_source,
        )
        self.assertIn(
            "edit_region |= full_head_regeneration_region",
            self.route_source,
        )
        self.assertIn(
            "never paste the source face as a rectangular block",
            self.route_source,
        )
        # Foreground clipping must happen first; otherwise it erases the
        # immediate-backdrop part of the coherent head regeneration region.
        clip_at = self.route_source.index("edit_region &= is_real_foreground")
        open_at = self.route_source.index("edit_region |= full_head_regeneration_region")
        self.assertLess(clip_at, open_at)

    def test_all_source_clothing_inside_measured_torso_is_editable(self):
        self.assertIn(
            "residual_source_clothing = (source_clothing > 0) & torso_corridor",
            self.route_source,
        )
        self.assertIn("edit_region |= residual_source_clothing", self.route_source)
        self.assertIn("f\"{arm_generation_instruction} \"", self.route_source)

    def test_dynamic_prompt_makes_person_body_geometry_authoritative(self):
        self.assertIn("BODY FIT: use the measured", self.route_source)
        self.assertIn(
            "Keep one shirt and one sleeve per side",
            self.route_source,
        )
        self.assertIn("passport_pose_instruction", self.route_source)
        self.assertIn("do not generate arms, hands, waist or lower body", self.route_source)

    def test_dynamic_prompt_retains_literal_tied_hair_geometry(self):
        self.assertIn("compact_hair_generation_lock", self.route_source)
        self.assertIn(
            "visually copy image 1's literal strand organization",
            self.route_source,
        )
        self.assertIn("tied_geometry_lock", self.route_source)
        self.assertIn("hair_endpoint_authority", self.route_source)
        budget_at = self.route_source.index(
            "generated_prompt = fit_prompt_to_token_budget"
        )
        retained_tail = self.route_source[budget_at:budget_at + 1000]
        self.assertIn("retained_hair_generation_lock", retained_tail)
        self.assertNotIn('f"{arm_generation_instruction}"', retained_tail)
        self.assertIn("retained_arm_generation_instruction", retained_tail)

    def test_dynamic_hair_contract_does_not_trust_lossy_texture_label(self):
        hair_keys_at = self.route_source.index("hair_fact_keys = [")
        hair_keys = self.route_source[hair_keys_at:hair_keys_at + 260]
        self.assertNotIn('"hair_style"', hair_keys)
        self.assertNotIn('"hair_texture"', hair_keys)
        self.assertIn("Image 1 pixels override every text label", self.route_source)

    def test_head_edge_diffusion_cannot_grow_hair_into_backdrop(self):
        ring_at = self.route_source.index("head_edge_diffusion_ring = (")
        ring = self.route_source[ring_at:ring_at + 360]
        self.assertIn("head_subject", ring)
        self.assertIn("& ~head_edge_inner", ring)
        self.assertNotIn("head_edge_outer", ring)

    def test_clean_selected_backdrop_bypasses_destructive_rgb_rematting(self):
        self.assertIn(
            "conditioning_already_has_selected_backdrop = (",
            self.route_source,
        )
        self.assertIn(
            "rough_source = clean_flat_person_conditioning.copy()",
            self.route_source,
        )
        self.assertIn(
            "retained original conditioning RGB and used ",
            self.route_source,
        )
        self.assertNotIn(
            "hard_identity_protection &= ~head_edge_diffusion_ring",
            self.route_source,
        )

    def test_below_jaw_source_hair_is_protected_from_garment_edit(self):
        self.assertIn(
            'rough_placement.get("measured_hair_end_ratio")',
            self.route_source,
        )
        self.assertIn(
            "parsed_identity_hair | recovered_identity_hair",
            self.route_source,
        )
        self.assertIn("& ~source_collar_residual", self.route_source)
        self.assertIn("protected_identity |= side_hair_guard", self.route_source)
        self.assertIn("complete_head_guard |= side_hair_guard", self.route_source)

    def test_source_collar_under_hair_is_released_for_uniform_diffusion(self):
        self.assertIn("source_collar_residual = (", self.route_source)
        self.assertIn(
            "protected_identity &= ~source_collar_residual",
            self.route_source,
        )
        self.assertIn(
            "complete_head_guard &= ~source_collar_residual",
            self.route_source,
        )
        self.assertIn(
            "recovered_identity_hair &= ~source_collar_residual",
            self.route_source,
        )
        self.assertIn("edit_region |= source_collar_residual", self.route_source)

    def test_head_guard_uses_detected_person_hair_not_rectangle(self):
        guard_at = self.route_source.index("complete_head_guard = (")
        guard = self.route_source[guard_at:guard_at + 420]
        self.assertIn("protected_identity", guard)
        self.assertIn("& is_real_foreground", guard)
        self.assertNotIn("~(clothing > 0)", guard)

    def test_tied_hair_accessories_are_not_pasted_over_wrong_geometry(self):
        self.assertIn(
            "tied_hair_requires_coherent_generation = trusted_tied_section_count > 0",
            self.route_source,
        )
        self.assertEqual(
            self.route_source.count("and not tied_hair_requires_coherent_generation"),
            2,
        )
        self.assertIn(
            "source-pixel accessory paste disabled",
            self.route_source,
        )

    def test_uniform_uses_native_diffusion_backdrop_without_rematting(self):
        self.assertIn("allow_uniform_background_postprocess = False", self.route_source)
        self.assertIn("normalize_decoded_background=False", self.route_source)
        self.assertIn(
            "Qwen diffusion did not generate the selected flat backdrop",
            self.route_source,
        )

    def test_template_pixels_are_the_dynamic_garment_color_authority(self):
        self.assertIn(
            "IMAGE 2 is the sole garment color, pattern, trim and construction authority",
            self.route_source,
        )
        self.assertIn("do not recolor, brighten, darken", self.route_source)
        self.assertIn("generation_cfg_scale = 1.20", self.route_source)

    def test_uniform_boundary_has_a_two_latent_soft_transition(self):
        self.assertIn(
            "else max(20.0, min(h, w) * .025)",
            self.pipeline_source,
        )
        self.assertIn("6.0 if preserve_source_clothing", self.pipeline_source)

    def test_uniform_requests_share_one_queue_worker(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        routes = (ROOT / "api/routes.py").read_text(encoding="utf-8")
        frontend = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertEqual(main.count('elif job_type == "uniform_swap":'), 1)
        self.assertNotIn('job_type == "qwen_uniform_swap"', main)
        self.assertNotIn('"type": "qwen_uniform_swap"', routes)
        self.assertNotIn("uniform-prompt-mode", frontend)
        self.assertNotIn("Simple Static Prompt Test", frontend)

    def test_legacy_uniform_generator_is_removed(self):
        self.assertNotIn("def qwen_edit_uniform_swap(", self.pipeline_source)


if __name__ == "__main__":
    unittest.main()
