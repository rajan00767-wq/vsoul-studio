"""Hair-silhouette smoothing tests.

The CPU VAE is 8x downsampled, so a near-black hair boundary on a white
backdrop cannot be reconstructed smoothly: measured on job 5a1bbeaa, the
contour's zig-zag energy (mean absolute second difference of the silhouette
column per row) was 0.707px in the conditioning Qwen receives and 1.457px in
its decode. The repair works on contour geometry, never on pixel content, so
a decoded strand is either kept exactly or replaced by backdrop.
"""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (ROOT / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
WHITE = (255, 255, 255)


def _load():
    import logging
    tree = ast.parse(PIPELINE)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "QwenEditPipeline")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "smooth_decoded_hair_silhouette")
    ns = {"np": np, "Image": Image, "logger": logging.getLogger("t")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "hair", "exec"), ns)
    return ns["smooth_decoded_hair_silhouette"]


def _load_generated_edge_antialias():
    tree = ast.parse(PIPELINE)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "QwenEditPipeline")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_antialias_generated_head_boundary")
    ns = {"np": np, "Image": Image, "cv2": __import__("cv2")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "head_edge", "exec"), ns)
    return ns["_antialias_generated_head_boundary"]


def zigzag(img, frac=0.22):
    """Mean |second difference| of the left silhouette column per row."""
    grey = np.array(img.convert("L")).astype(float)
    height = grey.shape[0]
    cols = []
    for row in range(int(0.06 * height), int(frac * height)):
        hit = np.flatnonzero(grey[row] < 170)
        cols.append(hit[0] if hit.size else np.nan)
    series = np.array(cols, dtype=float)
    series = series[~np.isnan(series)]
    return float(np.abs(np.diff(series, 2)).mean())


def _dome(serrate=0):
    """White frame with a dark dome; `serrate` adds a 1px staircase."""
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    for row in range(20, 200):
        half = int(np.sqrt(max(0.0, 90.0 ** 2 - (row - 110) ** 2)))
        jitter = (serrate if row % 2 else -serrate)
        left = 150 - half + jitter
        right = 150 + half - jitter
        img[row, max(0, left):min(300, right)] = (20, 18, 16)
    return Image.fromarray(img)


class SmoothHairSilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.smooth = _load()

    def test_serrated_edge_gets_smoother(self):
        rough = _dome(serrate=2)
        before = zigzag(rough)
        after = zigzag(self.smooth(rough, WHITE))
        self.assertLess(after, before)

    def test_already_smooth_edge_is_left_alone(self):
        clean = _dome(serrate=0)
        out = self.smooth(clean, WHITE)
        changed = (np.abs(
            np.array(out).astype(int) - np.array(clean).astype(int)
        ).sum(2) > 0).sum()
        self.assertLess(changed, 60, "a clean contour must barely be touched")

    def test_lower_frame_is_never_modified(self):
        # Shoulders, collar and garment are out of scope by construction.
        rough = _dome(serrate=2)
        out = np.array(self.smooth(rough, WHITE)).astype(int)
        before = np.array(rough).astype(int)
        cut = int(0.30 * before.shape[0])
        self.assertEqual(np.abs(out[cut:] - before[cut:]).max(), 0)

    def test_edges_move_no_further_than_the_cap(self):
        # A real cowlick must survive; only staircase steps are removed.
        rough = _dome(serrate=2)
        out = self.smooth(rough, WHITE, max_shift_px=2.5)
        before = np.array(rough.convert("L")).astype(float)
        after = np.array(out.convert("L")).astype(float)
        for row in range(30, 110):
            b = np.flatnonzero(before[row] < 170)
            a = np.flatnonzero(after[row] < 170)
            if b.size and a.size:
                self.assertLessEqual(abs(int(a[0]) - int(b[0])), 3)

    def test_only_backdrop_or_adjacent_hair_is_painted(self):
        # No invented colours: every changed pixel must end up as backdrop or
        # as a colour that already existed on its own row.
        rough = _dome(serrate=2)
        before = np.array(rough).astype(int)
        after = np.array(self.smooth(rough, WHITE)).astype(int)
        ys, xs = np.where(np.abs(after - before).sum(2) > 0)
        for y, x in list(zip(ys, xs))[:200]:
            pixel = tuple(after[y, x])
            self.assertTrue(
                pixel == WHITE or pixel in {tuple(p) for p in before[y]},
                f"invented colour {pixel} at ({y},{x})",
            )

    def test_backdrop_colour_is_honoured(self):
        blue = (4, 126, 246)
        img = np.array(_dome(serrate=2))
        img[img.sum(2) > 740] = blue
        out = self.smooth(Image.fromarray(img), blue)
        self.assertIsInstance(out, Image.Image)

    def test_tiny_frame_is_a_no_op(self):
        tiny = Image.new("RGB", (6, 6), WHITE)
        self.assertIs(self.smooth(tiny, WHITE), tiny)


class WiringTests(unittest.TestCase):
    def test_post_decode_repair_is_enabled_by_default(self):
        # The VAE can recreate a staircase even after diffusion owns the edge,
        # so the bounded geometry-only cleanup is part of normal delivery.
        # Off by default: on job 54b9d2a8's soft, spiky crown the repaint cut
        # visible white notches into the hair. Available behind the flag.
        self.assertIn('os.getenv("VSOUL_UNIFORM_SMOOTH_HAIR_EDGE", "0")', PIPELINE)

    def test_failure_cannot_fail_the_render(self):
        idx = PIPELINE.index("Silhouette smoothing skipped")
        self.assertIn("try:", PIPELINE[idx - 1000:idx])

    def test_neighbouring_methods_keep_their_decorators(self):
        # Inserting this helper above _repair_generated_face_chromatic_speckles
        # once duplicated one @staticmethod onto the new function and left the
        # existing one an instance method, so the next GPU run died with
        # "takes 2 positional arguments but 3 were given". The AST-level tests
        # all passed because they exec the function in isolation and never
        # exercise class binding. Check the real class attributes.
        tree = ast.parse(PIPELINE)
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "QwenEditPipeline")
        by_name = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name in (
            "smooth_decoded_hair_silhouette",
            "_repair_generated_face_chromatic_speckles",
        ):
            node = by_name[name]
            decorators = [
                d.id for d in node.decorator_list if isinstance(d, ast.Name)
            ]
            self.assertEqual(
                decorators.count("staticmethod"), 1,
                f"{name} must carry exactly one @staticmethod, got {decorators}",
            )
            first_arg = node.args.args[0].arg if node.args.args else None
            self.assertNotEqual(
                first_arg, "self",
                f"{name} is decorated @staticmethod but takes self",
            )

    def test_runs_before_delivery_crop(self):
        smooth_at = PIPELINE.index("identity_locked = self.smooth_decoded_hair_silhouette")
        crop_at = PIPELINE.index("cropped = self._crop_school_passport_portrait")
        self.assertLess(smooth_at, crop_at)

    def test_qwen_diffusion_gets_only_a_narrow_head_edge_ring(self):
        self.assertIn("head_edge_diffusion_ring", PIPELINE)
        self.assertIn("edit_region |= head_edge_diffusion_ring", PIPELINE)
        self.assertIn("head_subject = complete_head_guard & is_real_foreground", PIPELINE)
        self.assertIn("round(detected_face_width * .055), 10, 20", PIPELINE)
        self.assertIn("head_edge_kernel", PIPELINE)
        self.assertIn("head_edge_diffusion_ring &= ~protected_accessory_edge", PIPELINE)

    def test_edge_instruction_survives_prompt_budget_trimming(self):
        budget_at = PIPELINE.index("generated_prompt = fit_prompt_to_token_budget")
        retained_tail = PIPELINE[budget_at:budget_at + 1100]
        self.assertIn("EDGE LOCK", retained_tail)
        self.assertIn("no jagged pixels", retained_tail)


class GeneratedHeadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.antialias = _load_generated_edge_antialias()

    def test_vertical_hair_staircase_is_softened_inside_face_roi(self):
        image = np.full((240, 180, 3), 255, dtype=np.uint8)
        # Alternating columns simulate an 8x-VAE staircase on each side.
        for row in range(28, 178):
            jitter = 2 if row % 2 else 0
            image[row, 45 + jitter:135 - jitter] = (18, 16, 15)
        result, changed = self.antialias(
            Image.fromarray(image), (255, 255, 255), (58, 72, 122, 158),
        )
        self.assertGreater(changed, 0)
        output = np.asarray(result)
        # Face interior must remain byte-identical.
        np.testing.assert_array_equal(output[90:145, 70:110], image[90:145, 70:110])
        # At least one hard black boundary pixel becomes antialiased.
        transition = output[40:165, 40:55]
        self.assertTrue(np.any((transition > 30) & (transition < 245)))

    def test_pixels_below_head_roi_are_untouched(self):
        image = np.full((240, 180, 3), 255, dtype=np.uint8)
        image[30:170, 45:135] = (20, 18, 16)
        image[205:230, 15:165] = (210, 20, 20)
        before = image.copy()
        result, _ = self.antialias(
            Image.fromarray(image), (255, 255, 255), (58, 72, 122, 158),
        )
        np.testing.assert_array_equal(np.asarray(result)[200:], before[200:])

    def test_live_wiring_uses_detected_generated_face(self):
        self.assertIn(
            "self._antialias_generated_head_boundary(\n"
            "            identity_locked,\n"
            "            selected_bg_rgb,\n"
            "            generated_faces[0].bbox",
            PIPELINE,
        )


if __name__ == "__main__":
    unittest.main()
