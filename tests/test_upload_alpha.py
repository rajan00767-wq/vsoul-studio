"""Transparent uploads must be composited onto white, never onto black."""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from pipelines.qwen_edit_pipeline import QwenEditPipeline, _recover_unparsed_neck

ROOT = Path(__file__).resolve().parents[1]

class UploadAlphaTests(unittest.TestCase):
    def cutout(self, size=(64, 80)):
        """A red subject on a fully transparent field whose stored RGB is black."""
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        subject = Image.new("RGBA", (size[0] // 2, size[1] // 2), (200, 40, 40, 255))
        image.paste(subject, (size[0] // 4, size[1] // 4))
        return image

    def test_transparent_field_becomes_white_not_black(self):
        flattened = QwenEditPipeline._flatten_upload_alpha(self.cutout())
        pixels = np.asarray(flattened)
        self.assertEqual(flattened.mode, "RGB")
        # Corners were transparent-over-black; convert("RGB") would leave them 0.
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            self.assertEqual(tuple(pixels[corner]), (255, 255, 255))

    def test_opaque_subject_pixels_are_untouched(self):
        source = self.cutout()
        flattened = np.asarray(QwenEditPipeline._flatten_upload_alpha(source))
        original = np.asarray(source)
        opaque = original[:, :, 3] == 255
        self.assertTrue(opaque.any())
        self.assertTrue((flattened[opaque] == original[:, :, :3][opaque]).all())

    def test_partial_alpha_blends_toward_white(self):
        image = Image.new("RGBA", (8, 8), (0, 0, 0, 128))
        blended = np.asarray(QwenEditPipeline._flatten_upload_alpha(image))
        # Half-transparent black over white sits mid-grey, not at 0.
        self.assertTrue(100 <= int(blended[0, 0][0]) <= 155)

    def test_opaque_rgb_upload_is_passed_through_unchanged(self):
        image = Image.new("RGB", (8, 8), (10, 20, 30))
        result = np.asarray(QwenEditPipeline._flatten_upload_alpha(image))
        self.assertEqual(tuple(result[0, 0]), (10, 20, 30))

    def test_fully_opaque_rgba_keeps_its_own_pixels(self):
        image = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
        result = np.asarray(QwenEditPipeline._flatten_upload_alpha(image))
        self.assertEqual(tuple(result[0, 0]), (10, 20, 30))

    def test_uniform_swap_flattens_alpha_before_measuring_the_person(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_vl_image_edit_uniform_swap"
        )
        body = ast.get_source_segment(source, route)
        # The flatten must happen before any measurement, parse or conditioning
        # build; a bare convert("RGB") here silently reintroduces the black field.
        self.assertIn("person_pil = self._flatten_upload_alpha(_person_upload)", body)
        self.assertNotIn('person_pil = self._to_pil(person_input).convert("RGB")', body)
        self.assertLess(
            body.index("_flatten_upload_alpha"),
            body.index("_restore_dead_and_hot_pixels"),
        )
        # Downstream wall-keying needs to know the white backdrop was synthetic.
        self.assertIn("self._last_upload_alpha_flattened = bool(", body)
        self.assertLess(
            body.index("_last_upload_alpha_flattened"),
            body.index("person_pil = self._flatten_upload_alpha"),
        )

class RawCandidateRewriteTests(unittest.TestCase):
    """The enhancer's raw artifact is read and then rewritten by the same routine."""

    def route_source(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        route = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_vl_image_edit_uniform_swap"
        )
        return ast.get_source_segment(source, route)

    def test_decoded_candidate_releases_the_file_handle(self):
        body = self.route_source()
        # A bare Image.open(...).convert(...) leaves PIL lazily attached to the
        # file that is rewritten a few hundred lines later.
        self.assertIn("with Image.open(output_path) as _decoded_candidate:", body)
        self.assertNotIn('Image.open(output_path).convert("RGB").resize(', body)

    def test_raw_artifact_is_rewritten_atomically(self):
        body = self.route_source()
        # Overwriting the just-read path in place failed on Windows with
        # "[Errno 22] Invalid argument" and discarded a finished generation.
        self.assertNotIn('generated_candidate.save(output_path, format="PNG")', body)
        self.assertIn("os.replace(_normalized_raw_tmp, output_path)", body)
        rewrite = body.index("_normalized_raw_tmp = output_path.with_name")
        replace = body.index("os.replace(_normalized_raw_tmp, output_path)")
        self.assertLess(rewrite, replace)

class SyntheticBackdropGuardTests(unittest.TestCase):
    """A composited white field is not a photographed wall."""

    def lock_source(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = source.index("def _lock_uniform_source_head(")
        return source[start:source.index("def _parse_bg_color(", start)]

    def test_head_lock_executes_with_either_backdrop_origin(self):
        """Call it for real: a source-text assertion missed a NameError here.

        The guard was first written as ``getattr(self, ...)`` inside a
        ``@staticmethod``, which has no ``self``. Every structural test still
        passed and the job failed at runtime with "name 'self' is not defined".
        """
        source = Image.new("RGB", (240, 300), (190, 150, 120))
        generated = Image.new("RGB", (240, 300), (200, 160, 130))
        labels = np.zeros((300, 240), dtype=np.uint8)
        labels[40:150, 80:160] = 13
        labels[20:60, 70:170] = 2
        labels[180:, 40:200] = 5
        for synthetic in (False, True):
            with self.subTest(synthetic=synthetic):
                image, info = QwenEditPipeline._lock_uniform_source_head(
                    source, generated, labels, (255, 255, 255),
                    upload_backdrop_is_synthetic=synthetic,
                )
                self.assertEqual(image.size, generated.size)
                self.assertIn("applied", info)

    def test_wall_key_is_skipped_for_cutout_uploads(self):
        body = self.lock_source()
        # Keying the synthetic white removed ~30k real antialiased hair pixels
        # and left the restored head sitting inside a grey envelope.
        self.assertIn("upload_backdrop_is_synthetic", body)
        self.assertIn(
            "if float(upload_bg_lab[0]) > 60.0 and not upload_backdrop_is_synthetic:",
            body,
        )
        self.assertNotIn("if float(upload_bg_lab[0]) > 60.0:" + chr(10), body)

    def test_flag_defaults_off_so_photographed_walls_still_key(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("self._last_upload_alpha_flattened: bool = False", source)
        # An opaque upload must leave the guard enabled.
        opaque = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
        self.assertFalse(bool(opaque.getchannel("A").getextrema()[0] < 255))
        cutout = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
        self.assertTrue(bool(cutout.getchannel("A").getextrema()[0] < 255))

class QueueGpuReleaseTests(unittest.TestCase):
    """Every queued job must hand the GPU back before the next one starts."""

    def main_source(self):
        return (ROOT / "main.py").read_text(encoding="utf-8")

    def test_job_execution_releases_cuda_in_a_finally(self):
        source = self.main_source()
        tree = ast.parse(source)
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_execute_queued_job"
        )
        tries = [node for node in fn.body if isinstance(node, ast.Try)]
        self.assertEqual(len(tries), 1)
        # A finally (not a success-path call) is required: the cancelled branch
        # returns early and failures raise, and both still hold GPU memory.
        self.assertTrue(tries[0].finalbody)
        body = ast.get_source_segment(source, tries[0])
        self.assertIn("_release_cuda_between_jobs(job_id)", body)

    def test_release_helper_empties_the_cache(self):
        source = self.main_source()
        start = source.index("def _release_cuda_between_jobs(")
        helper = source[start:source.index("async def _queue_worker_loop", start)]
        for call in ("gc.collect()", "torch.cuda.synchronize()", "torch.cuda.empty_cache()"):
            self.assertIn(call, helper)
        # Reclamation must never mask the job's own result.
        self.assertIn("except Exception", helper)

    def test_server_sets_expandable_segments_before_importing_torch(self):
        source = self.main_source()
        alloc = source.index("PYTORCH_CUDA_ALLOC_CONF")
        self.assertIn("expandable_segments:True", source)
        # torch is imported lazily inside helpers; the env var must precede
        # every one of those import sites to take effect.
        first_torch_import = source.index("import torch")
        self.assertLess(alloc, first_torch_import)

class UnparsedNeckRecoveryTests(unittest.TestCase):
    """LIP-20 has no neck class, so a labels!=0 foreground drops the neck."""

    def fixture(self):
        rgb = np.full((300, 240, 3), 255, dtype=np.uint8)
        rgb[40:150, 80:160] = (176, 112, 79)    # face
        rgb[150:215, 100:140] = (168, 104, 74)  # neck, skin coloured
        rgb[215:, 40:200] = (20, 30, 90)        # collar
        parsed = np.zeros((300, 240), dtype=bool)
        parsed[40:150, 80:160] = True           # parser sees only face...
        parsed[215:, 40:200] = True             # ...and clothing
        return rgb, parsed, (80.0, 40.0, 160.0, 150.0)

    def test_neck_pixels_are_added_to_the_foreground(self):
        rgb, parsed, bbox = self.fixture()
        recovered = _recover_unparsed_neck(rgb, parsed, bbox)
        # The corridor reaches .42 face-heights below the chin (row 196 here),
        # which is where a seated collar covers the rest of the neck.
        column = slice(110, 130)
        self.assertFalse(parsed[155:195, column].any(), "fixture already had a neck")
        self.assertTrue(
            recovered[155:195, column].all(),
            "neck still missing from the foreground mask",
        )

    def test_existing_foreground_is_never_removed(self):
        rgb, parsed, bbox = self.fixture()
        recovered = _recover_unparsed_neck(rgb, parsed, bbox)
        self.assertTrue(recovered[parsed].all())

    def test_non_skin_pixels_in_the_corridor_stay_out(self):
        rgb, parsed, bbox = self.fixture()
        # A blue lanyard hanging in the same corridor must not be recovered.
        rgb[160:200, 150:160] = (20, 30, 200)
        recovered = _recover_unparsed_neck(rgb, parsed, bbox)
        self.assertFalse(recovered[165:195, 151:159].any())

    def test_bad_face_box_is_a_no_op(self):
        rgb, parsed, _ = self.fixture()
        self.assertTrue(
            np.array_equal(_recover_unparsed_neck(rgb, parsed, None), parsed)
        )

class CancelHonouredTests(unittest.TestCase):
    """Cancel is cooperative: every progress callback must observe it."""

    def callbacks(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        fn = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_execute_queued_job"
        )
        return source, [
            node for node in ast.walk(fn)
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_cb", "_uniform_cb"}
        ]

    def test_every_progress_callback_checks_for_cancellation(self):
        source, callbacks = self.callbacks()
        self.assertGreaterEqual(len(callbacks), 3, "expected one callback per job type")
        for node in callbacks:
            body = ast.get_source_segment(source, node)
            with self.subTest(line=node.lineno):
                # Writing status="processing" without first checking the record
                # overwrites the user's cancel and the job runs to completion.
                self.assertIn('get("status") == "cancelled"', body)
                self.assertIn("JobCancelledError", body)

    def test_cancel_check_precedes_the_status_write(self):
        source, callbacks = self.callbacks()
        for node in callbacks:
            body = ast.get_source_segment(source, node)
            with self.subTest(line=node.lineno):
                self.assertLess(
                    body.index('get("status") == "cancelled"'),
                    body.index("_js.update_job("),
                    "cancel checked after the status write, too late to matter",
                )

class IdentityGateTests(unittest.TestCase):
    """A borderline embedding must not silence the visual reviewer."""

    def route(self):
        source = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = source.index("measured_high_confidence_identity = bool(")
        end = source.index("visual_review[\"advisory_failures\"]", start)
        return source, source[start:end]

    def test_high_confidence_uses_the_shared_identity_floor(self):
        _, body = self.route()
        # .70 is the delivery floor. Reusing it here meant anything delivered
        # also outranked the reviewer, so a .739 decode of a visibly different
        # child shipped with "identity_changed" downgraded to advisory.
        self.assertIn("MIN_PORTRAIT_IDENTITY_SIMILARITY", body)
        self.assertNotIn('identity_review["similarity"]) >= .70', body)

    def test_identity_changed_is_not_overruled_by_mere_acceptance(self):
        _, body = self.route()
        # Target the blocking decision, not the earlier advisory downgrade.
        start = body.index('blocking_failures.append("identity_changed")')
        clause = body[body.rindex('visual_review.get("identity_changed")', 0, start):start]
        self.assertIn("not measured_high_confidence_identity", clause)
        # OR-ing "accepted" made the block unreachable for every delivered job.
        self.assertNotIn("identity_review.get(\"accepted\") or", clause)

    def test_shared_floor_is_stricter_than_the_delivery_floor(self):
        source, _ = self.route()
        line = next(
            l for l in source.splitlines()
            if l.startswith("MIN_PORTRAIT_IDENTITY_SIMILARITY")
        )
        floor = float(line.split("=")[1].strip())
        self.assertGreater(floor, 0.70)

class NegativeGuidanceTests(unittest.TestCase):
    """CFG needs a negative branch, and it must not subtract the images."""

    def source(self):
        return (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")

    def test_negative_prompt_is_encoded_without_conditioning_images(self):
        src = self.source()
        # Attaching the images to the negative made CFG subtract the garment
        # embedding, which erased the plaid and rebuilt the source vest. That
        # is why the uniform route used to pass negative_prompt=None at all.
        self.assertIn("image=None if text_only else images", src)
        self.assertIn("for prompt, text_only in zip(prompts, text_only_flags)", src)
        self.assertIn(
            "text_only_prompts=[False] + ([True] if effective_negative else [])", src
        )

    def test_text_only_flag_changes_the_embedding_cache_key(self):
        src = self.source()
        self.assertIn(
            "cache_hasher.update(bytes(1 if flag else 0 for flag in text_only_flags))",
            src,
        )

    def test_uniform_route_actually_passes_its_negative_prompt(self):
        src = self.source()
        route = next(
            node for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.FunctionDef)
            and node.name == "qwen_vl_image_edit_uniform_swap"
        )
        body = ast.get_source_segment(src, route)
        # The route built a negative prompt, wrote it to the audit JSON, then
        # discarded it - so none of its identity bans reached the model.
        self.assertIn("negative_prompt=generation_negative_prompt", body)
        # Scope the check to the main generation call. The opt-in identity
        # prepass is a separate route and still runs without guidance.
        main_call = body[body.index("output_path = self.qwen_edit_enhancer("):]
        self.assertNotIn("negative_prompt=None", main_call)

    def test_dynamic_uniform_cfg_exceeds_one_so_negative_branch_exists(self):
        src = self.source()
        route = src[src.index("def qwen_vl_image_edit_uniform_swap("):]
        line = next(l for l in route.splitlines() if "generation_cfg_scale = " in l)
        value = float(line.split("=")[1].strip())
        # diffusers skips the negative branch entirely at true_cfg_scale <= 1.
        self.assertGreater(value, 1.0)
        self.assertLessEqual(value, 2.0, "strong guidance redraws the face")

    def test_mismatched_flag_count_is_rejected(self):
        from pipelines.qwen_edit_pipeline import _encode_prompt_isolated
        with self.assertRaises(ValueError):
            _encode_prompt_isolated(
                Image.new("RGB", (8, 8)), ["a", "b"], text_only_prompts=[True],
            )

class GarmentReferenceSafetyTests(unittest.TestCase):
    """Qwen receives one clean person and one face-free garment image."""

    def test_rough_composite_is_not_a_visual_authority(self):
        src = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("generation_primary = primary_person_source", src)
        self.assertIn(
            "qwen_garment_references = [uniform_validation_reference]", src,
        )
        self.assertNotIn(
            "qwen_garment_references.append(fitted_garment_reference)", src,
        )
        self.assertNotIn("fitted_garment_reference.alpha_composite", src)

class DeliveryCropHeadroomTests(unittest.TestCase):
    """A chest-cropped upload must not be answered by zooming the head."""

    def crop_source(self):
        src = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        start = src.index("maximum_real_crop_h = max(1, int(source_h - desired_top))")
        return src[start:src.index("pad_top = max(0, -desired_top)", start)]

    def test_flat_backdrop_is_extended_before_clamping_the_crop(self):
        body = self.crop_source()
        # Clamping tightened the frame to an effective .433 face fraction and
        # cut headroom from .124 to .079 - the head read as zoomed.
        self.assertIn("pad_bottom = int(crop_h - maximum_real_crop_h)", body)
        self.assertIn("bottom_is_backdrop", body)

    def test_padding_only_when_the_bottom_edge_is_really_backdrop(self):
        body = self.crop_source()
        # Padding under a garment that still fills the bottom row would end
        # the shirt in a hard horizontal line.
        self.assertIn("bottom_strip", body)
        self.assertIn("else:", body)
        self.assertIn("crop_h = min(source_h, crop_h)", body)
        self.assertLess(body.index("bottom_is_backdrop"), body.index("crop_h = min(source_h, crop_h)"))

    def test_clamp_fallback_still_recomputes_its_framing(self):
        body = self.crop_source()
        tail = body[body.index("crop_h = min(source_h, crop_h)"):]
        for recomputed in ("crop_w =", "target_headroom =", "desired_top =", "desired_left ="):
            self.assertIn(recomputed, tail)

if __name__ == "__main__":
    unittest.main()
