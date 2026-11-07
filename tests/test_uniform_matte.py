import ast
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import cv2


class UniformMatteTests(unittest.TestCase):
    def raw_background_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_normalize_qwen_raw_background")
        scope = dict(Image=Image, np=np, cv2=cv2, io=io, Union=Union, Tuple=Tuple)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "raw_background", "exec"), scope)
        return scope[method.name]

    def background_check_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_selected_solid_background")
        scope = dict(Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), "background_check", "exec"), scope)
        return scope[method.name]

    def flat_background_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_flat_border_background")
        scope = dict(
            Image=Image, np=np, cv2=cv2, logger=Mock(), Optional=Optional, Union=Union, Tuple=Tuple,
            QwenEditPipeline=SimpleNamespace(_parse_bg_color=lambda color: color),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), "flat_background", "exec"), scope)
        return scope[method.name]

    def smooth_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_smooth_border_background")
        scope = dict(Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "smooth", "exec"), scope)
        return scope[method.name]

    def strict_flat_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_flat_border_background_strict")
        scope = dict(
            Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple,
            logger=Mock(),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), "strict_flat", "exec"), scope)
        return scope[method.name]

    def test_strict_flat_background_keeps_blue_clothing_and_pale_accessory(self):
        source_bg = (56, 134, 194)
        selected_bg = (4, 126, 246)
        pixels = np.full((96, 72, 3), source_bg, dtype=np.uint8)
        # A blue garment reaches the lower canvas edge. Its subtle weave gives
        # it texture, so it must not be confused with the generated backdrop.
        for y in range(50, 96):
            for x in range(8, 64):
                delta = 18 if (x + y) % 2 else -18
                pixels[y, x] = np.clip(np.array(source_bg) + (delta, delta, delta), 0, 255)
        # A pale flower overlaps the upper border-connected background.
        pixels[12:25, 4:18] = (224, 218, 228)
        original_garment = pixels[50:, 8:64].copy()
        original_flower = pixels[12:25, 4:18].copy()
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)

        result = np.asarray(
            self.strict_flat_helper()(owner, Image.fromarray(pixels), selected_bg)
        )

        self.assertTrue(np.all(result[0, 0] == selected_bg))
        self.assertTrue(np.array_equal(result[50:, 8:64], original_garment))
        self.assertTrue(np.array_equal(result[12:25, 4:18], original_flower))

    def tiny_hole_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_repair_tiny_backdrop_holes_in_head_neck")
        scope = dict(
            Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple,
            logger=Mock(),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), "tiny_holes", "exec"), scope)
        return scope[method.name]

    def test_tiny_hair_backdrop_hole_uses_subject_color_not_backdrop(self):
        target = (4, 126, 246)
        pixels = np.full((80, 80, 3), target, dtype=np.uint8)
        labels = np.zeros((80, 80), dtype=np.uint8)
        labels[14:58, 18:62] = 2
        labels[34:66, 27:53] = 13
        pixels[14:58, 18:62] = (22, 20, 18)
        pixels[34:66, 27:53] = (118, 75, 54)
        pixels[24:26, 38:40] = target
        labels[24:26, 38:40] = 0
        # Dark antialias blends retain the backdrop hue but can be more than
        # 100 RGB-distance units from the requested colour.
        pixels[28:30, 44:46] = (34, 70, 111)
        labels[28:30, 44:46] = 0
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.tiny_hole_helper()(owner, Image.fromarray(pixels), target)
        repaired = np.asarray(result)
        self.assertLess(int(repaired[24:26, 38:40, 2].max()), 100)
        self.assertLess(int(repaired[28:30, 44:46, 2].max()), 100)

    def envelope_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_background_outside_subject_envelope")
        scope = dict(Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), "envelope", "exec"), scope)
        return scope[method.name]

    def helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_background_with_foreground_matte")
        scope = dict(Image=Image, np=np, cv2=cv2, Optional=Optional, Tuple=Tuple, Union=Union, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), "matte", "exec"), scope)
        return scope[method.name]

    def test_exact_background_and_original_opaque_pixels(self):
        source = Image.new("RGB", (32, 32), (240, 240, 240))
        labels = np.zeros((32, 32), dtype=np.uint8)
        labels[8:24, 8:24] = 13
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.helper()(owner, source, (4, 126, 246), preserve_foreground_rgb=True, strict=True)
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((16, 16)), (240, 240, 240))
        parser.assert_called_once()

    def test_textured_scene_matte_preserves_compact_hair_accessory(self):
        target = (4, 126, 246)
        height, width = 128, 96
        yy, xx = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, 0] = 165 + ((xx * 3 + yy) % 35)
        pixels[:, :, 1] = 182 + ((xx + yy * 2) % 30)
        pixels[:, :, 2] = 150 + ((xx * 2 + yy * 3) % 40)
        labels = np.zeros((height, width), dtype=np.uint8)
        labels[14:68, 25:71] = 2
        labels[38:82, 31:65] = 13
        labels[78:128, 12:84] = 5
        pixels[14:68, 25:71] = (25, 22, 20)
        pixels[38:82, 31:65] = (154, 101, 76)
        pixels[78:128, 12:84] = (45, 55, 80)
        # SCHP commonly reports a flower/bow as background. It overlaps the
        # hair edge and must survive replacement of the textured scenery.
        labels[20:42, 66:86] = 0
        pixels[20:42, 66:86] = (236, 126, 184)
        pixels[26:36, 72:80] = (250, 226, 236)
        original_flower = pixels[20:42, 66:86].copy()
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.helper()(
                owner, Image.fromarray(pixels), target,
                preserve_foreground_rgb=True, strict=True,
            ))
        self.assertTrue(np.all(result[0, 0] == target))
        self.assertTrue(np.array_equal(result[23:39, 69:83], original_flower[3:19, 3:17]))

    def test_schp_foreground_is_never_removed_by_scene_colour_cleanup(self):
        target = (4, 126, 246)
        pixels = np.full((96, 80, 3), (226, 226, 224), dtype=np.uint8)
        labels = np.zeros((96, 80), dtype=np.uint8)
        labels[12:48, 24:56] = 2
        labels[38:68, 28:52] = 13
        labels[66:96, 12:68] = 5
        pixels[12:48, 24:56] = (24, 22, 20)
        pixels[38:68, 28:52] = (154, 101, 76)
        pixels[66:96, 12:68] = (35, 48, 76)
        # Simulate difficult edge pixels that match the sampled scene colour
        # even though SCHP correctly classifies them as hair and collar.
        pixels[20:43, 24] = (226, 226, 224)
        pixels[66, 20:60] = (226, 226, 224)
        original = pixels.copy()
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.helper()(
                owner, Image.fromarray(pixels), target,
                preserve_foreground_rgb=True, strict=True,
            ))
        semantic_core = cv2.erode(
            (labels != 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1,
        ).astype(bool)
        self.assertTrue(np.array_equal(result[semantic_core], original[semantic_core]))
        self.assertTrue(np.all(result[0, 0] == target))

    def test_thin_bright_glasses_detail_is_not_treated_as_background_edge(self):
        target = (4, 126, 246)
        pixels = np.full((96, 80, 3), (226, 226, 224), dtype=np.uint8)
        labels = np.zeros((96, 80), dtype=np.uint8)
        labels[15:48, 24:56] = 2
        labels[32:70, 28:52] = 13
        labels[68:96, 10:70] = 5
        pixels[15:48, 24:56] = (24, 22, 20)
        pixels[32:70, 28:52] = (154, 101, 76)
        pixels[68:96, 10:70] = (35, 48, 76)
        labels[43:45, 30:50] = 4
        pixels[43:45, 30:50] = (205, 225, 218)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.helper()(
                owner, Image.fromarray(pixels), target,
                preserve_foreground_rgb=True, strict=True,
            ))
        self.assertTrue(np.all(result[43:45, 30:50] == (205, 225, 218)))

    def test_person_matte_does_not_blend_over_semantic_diagonal_boundary(self):
        size = 80
        target = (4, 126, 246)
        pixels = np.full((size, size, 3), (242, 242, 242), dtype=np.uint8)
        labels = np.zeros((size, size), dtype=np.uint8)
        for row in range(12, 68):
            left = 12 + row // 5
            labels[row, left:68] = 13
            pixels[row, left:68] = (72, 48, 38)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.helper()(
                owner, Image.fromarray(pixels), target,
                preserve_foreground_rgb=True, strict=True,
            ))
        target_array = np.asarray(target, dtype=np.uint8)
        subject_array = np.asarray((72, 48, 38), dtype=np.uint8)
        semantic_core = cv2.erode(
            (labels != 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1,
        ).astype(bool)
        self.assertTrue(np.all(result[semantic_core] == subject_array))
        self.assertTrue(np.all(result[0, 0] == target_array))

    def test_selected_background_check_ignores_internal_white_plaid_holes(self):
        pixels = np.full((120, 100, 3), (4, 126, 246), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[28:120, 18:82] = 5
        pixels[28:120, 18:82] = (25, 35, 70)
        # Simulate SCHP reporting bright checks inside the shirt as background.
        labels[62:110:12, 30:70] = 0
        pixels[62:110:12, 30:70] = (248, 248, 248)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            accepted = self.background_check_helper()(
                owner, Image.fromarray(pixels), (4, 126, 246),
            )
        self.assertTrue(accepted)

    def test_selected_background_check_accepts_exact_border_with_parser_missed_hair(self):
        target = (4, 126, 246)
        pixels = np.full((120, 100, 3), target, dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[20:72, 30:70] = 2
        labels[66:120, 12:88] = 5
        pixels[20:72, 30:70] = (24, 22, 20)
        pixels[66:120, 12:88] = (90, 94, 102)
        # A disconnected ponytail/accessory can be missed by SCHP while the
        # actual backdrop remains exact along both borders and top corners.
        pixels[30:82, 72:88] = (58, 55, 52)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            accepted = self.background_check_helper()(
                owner, Image.fromarray(pixels), target,
            )
        self.assertTrue(accepted)

    def test_selected_background_check_rejects_textured_side_border(self):
        target = (4, 126, 246)
        pixels = np.full((120, 100, 3), target, dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[24:120, 18:82] = 5
        pixels[24:120, 18:82] = (80, 84, 92)
        pixels[20:80, :3] = (180, 170, 145)
        pixels[20:80, -3:] = (160, 175, 150)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            accepted = self.background_check_helper()(
                owner, Image.fromarray(pixels), target,
            )
        self.assertFalse(accepted)

    def test_selected_background_rejects_interior_scenery_behind_exact_frame(self):
        target = (4, 126, 246)
        pixels = np.full((120, 100, 3), target, dtype=np.uint8)
        # Simulate a border-only correction around a cloud scene. A faulty
        # parser labels the clouds as foreground, so geometry must still fail.
        pixels[8:42, 8:24] = (218, 232, 244)
        pixels[12:46, 76:92] = (190, 215, 235)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[8:46, 8:24] = 5
        labels[12:46, 76:92] = 5
        labels[28:120, 28:72] = 13
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            accepted = self.background_check_helper()(
                owner, Image.fromarray(pixels), target,
            )
        self.assertFalse(accepted)

    def test_transparent_export_keeps_subject_and_clears_background(self):
        source = Image.new("RGB", (32, 32), (255, 255, 255))
        pixels = np.asarray(source).copy()
        pixels[8:24, 8:24] = (28, 24, 20)
        labels = np.zeros((32, 32), dtype=np.uint8)
        labels[8:24, 8:24] = 13
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.helper()(
                owner, Image.fromarray(pixels), (255, 255, 255),
                preserve_foreground_rgb=True, strict=True, transparent=True,
            )
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((16, 16)), (28, 24, 20, 255))

    def test_raw_background_removes_white_from_soft_subject_edge(self):
        pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
        pixels[8:24, 8:24] = (20, 20, 20)
        pixels[7, 8:24] = (138, 138, 138)
        pixels[6, 8:24] = (235, 235, 235)
        alpha = np.zeros((32, 32), dtype=np.uint8)
        alpha[8:24, 8:24] = 255
        alpha[7, 8:24] = 128
        alpha[6, 8:24] = 245
        cutout = Image.fromarray(np.dstack((pixels, alpha)), "RGBA")
        with io.BytesIO() as buffer:
            cutout.save(buffer, format="PNG")
            cutout_bytes = buffer.getvalue()
        remover = SimpleNamespace(remove_background=Mock(return_value=cutout_bytes))
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.birefnet_service": SimpleNamespace(background_removal=remover)}):
            result = np.asarray(self.raw_background_helper()(
                owner, Image.fromarray(pixels), (4, 126, 246),
            ))
        self.assertTrue(np.array_equal(result[0, 0], (4, 126, 246)))
        self.assertTrue(np.array_equal(result[12, 12], (20, 20, 20)))
        self.assertLess(int(result[7, 12, 0]), 20)
        self.assertGreater(int(result[7, 12, 2]), 120)
        self.assertLess(int(result[6, 12, 0]), 90)
        self.assertGreater(int(result[6, 12, 2]), 200)

    def test_raw_background_removes_fully_opaque_pale_outer_fringe(self):
        pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
        pixels[8:24, 8:24] = (20, 20, 20)
        pixels[7, 8:24] = (236, 236, 236)
        alpha = np.zeros((32, 32), dtype=np.uint8)
        alpha[7:24, 8:24] = 255
        cutout = Image.fromarray(np.dstack((pixels, alpha)), "RGBA")
        with io.BytesIO() as buffer:
            cutout.save(buffer, format="PNG")
            cutout_bytes = buffer.getvalue()
        remover = SimpleNamespace(remove_background=Mock(return_value=cutout_bytes))
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.birefnet_service": SimpleNamespace(background_removal=remover)}):
            result = np.asarray(self.raw_background_helper()(
                owner, Image.fromarray(pixels), (4, 126, 246),
            ))
        self.assertLess(int(result[7, 12, 0]), 90)
        self.assertGreater(int(result[7, 12, 2]), 200)
        self.assertTrue(np.array_equal(result[12, 12], (20, 20, 20)))

    def test_failure_does_not_silently_return_wrong_background(self):
        parser = Mock(side_effect=RuntimeError("model unavailable"))
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            with self.assertRaisesRegex(RuntimeError, "raw Qwen output is retained"):
                self.helper()(None, Image.new("RGB", (32, 32)), (4, 126, 246),
                              preserve_foreground_rgb=True, strict=True)

    def test_exact_background_preserves_saturated_hair_accessory(self):
        pixels = np.full((120, 100, 3), (8, 91, 164), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[24:82, 30:72] = 2
        labels[62:120, 18:84] = 5
        pixels[24:82, 30:72] = (22, 20, 18)
        pixels[62:120, 18:84] = (238, 238, 238)
        # The outer half of this red bow is not covered by the semantic hair
        # label, matching the failure seen with real school accessories.
        pixels[10:38, 8:38] = (210, 18, 30)
        pixels[19:27, 18:27] = (224, 216, 190)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.helper()(
                owner, Image.fromarray(pixels), (4, 126, 246),
                preserve_foreground_rgb=True, strict=True,
            )
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((12, 14)), (210, 18, 30))
        self.assertEqual(result.getpixel((21, 22)), (224, 216, 190))

    def test_smooth_replacement_preserves_pale_hair_flower(self):
        pixels = np.full((80, 64, 3), (158, 158, 158), dtype=np.uint8)
        pixels[28:54, :8] = (126, 126, 126)
        pixels[10:60, 14:50] = (24, 22, 20)
        pixels[12:28, 6:22] = (224, 204, 215)
        pixels[17:22, 11:16] = (170, 150, 55)
        labels = np.zeros((80, 64), dtype=np.uint8)
        labels[10:60, 14:50] = 2
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((3, 40)), (4, 126, 246))
        self.assertEqual(result.getpixel((8, 14)), (224, 204, 215))
        self.assertEqual(result.getpixel((32, 30)), (24, 22, 20))

    def test_smooth_replacement_removes_wall_beside_hair(self):
        pixels = np.full((100, 80, 3), (245, 245, 242), dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.uint8)
        labels[12:62, 22:58] = 2
        labels[55:100, 10:70] = 5
        pixels[12:62, 22:58] = (24, 22, 20)
        pixels[55:100, 10:70] = (235, 235, 235)
        # A darker patch of the same wall beside the hair used to survive as a
        # large gray halo because it was inside a broad accessory zone.
        pixels[20:55, 2:22] = (218, 218, 214)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((8, 32)), (4, 126, 246))
        self.assertEqual(result.getpixel((40, 30)), (24, 22, 20))
        self.assertEqual(result.getpixel((40, 80)), (235, 235, 235))
        # A gray uniform touching the lower border must not become a blue hole.
        self.assertEqual(result.getpixel((40, 99)), (235, 235, 235))

    def test_smooth_replacement_removes_flat_bottom_wedge_mislabeled_as_arm(self):
        pixels = np.full((100, 80, 3), (232, 232, 232), dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.uint8)
        labels[12:58, 24:56] = 2
        labels[52:100, 10:70] = 5
        pixels[12:58, 24:56] = (24, 22, 20)
        pixels[52:100, 10:70] = (35, 48, 76)
        # The generated gray backdrop touches both lower corners, but SCHP
        # incorrectly labels the strips as arms because they touch sleeves.
        labels[82:100, :10] = 15
        labels[82:100, 70:] = 14
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((4, 95)), (4, 126, 246))
        self.assertEqual(result.getpixel((75, 95)), (4, 126, 246))
        self.assertEqual(result.getpixel((40, 90)), (35, 48, 76))

    def test_smooth_replacement_removes_flat_shoulder_halo_mislabeled_as_shirt(self):
        pixels = np.full((120, 100, 3), (232, 232, 232), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[18:66, 30:70] = 2
        labels[66:120, 12:88] = 5
        pixels[18:66, 30:70] = (24, 22, 20)
        pixels[66:120, 12:88] = (35, 48, 76)
        # SCHP sometimes extends the shirt label into a perfectly flat strip
        # of the old backdrop along the outer shoulder, producing a white halo.
        labels[72:112, 8:12] = 5
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        corrected_halo = np.asarray(result.getpixel((9, 90)), dtype=np.float32)
        self.assertLess(
            np.linalg.norm(corrected_halo - np.asarray((4, 126, 246), dtype=np.float32)),
            16.0,
        )
        self.assertEqual(result.getpixel((40, 90)), (35, 48, 76))

    def test_subject_envelope_replaces_textured_backdrop_without_portrait_holes(self):
        height, width = 120, 100
        yy, xx = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, 0] = 70 + (xx * 7 + yy * 3) % 95
        pixels[:, :, 1] = 105 + (xx * 3 + yy * 5) % 105
        pixels[:, :, 2] = 125 + (xx * 5 + yy * 2) % 120
        labels = np.zeros((height, width), dtype=np.uint8)
        labels[14:68, 30:70] = 2
        labels[62:120, 10:90] = 5
        pixels[14:68, 30:70] = (24, 22, 20)
        pixels[62:120, 10:90] = (185, 28, 46)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        target = (4, 126, 246)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.envelope_helper()(
                owner, Image.fromarray(pixels), target,
            ))
        self.assertTrue(np.all(result[0, 0] == target))
        self.assertTrue(np.all(result[45, 8] == target))
        self.assertTrue(np.all(result[40, 40] == (24, 22, 20)))
        self.assertTrue(np.all(result[90, 50] == (185, 28, 46)))

    def test_subject_envelope_keeps_head_disconnected_from_larger_uniform(self):
        height, width = 140, 100
        pixels = np.full((height, width, 3), (170, 180, 190), dtype=np.uint8)
        labels = np.zeros((height, width), dtype=np.uint8)
        labels[12:58, 30:70] = 13
        labels[7:25, 26:74] = 2
        labels[76:140, 8:92] = 5
        pixels[12:58, 30:70] = (145, 92, 68)
        pixels[7:25, 26:74] = (25, 22, 20)
        pixels[58:76, 43:57] = (132, 82, 62)  # neck missed by parser
        pixels[76:140, 8:92] = (185, 28, 46)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        target = (4, 126, 246)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.envelope_helper()(
                owner, Image.fromarray(pixels), target,
            ))
        self.assertTrue(np.all(result[35, 50] == (145, 92, 68)))
        self.assertTrue(np.all(result[15, 35] == (25, 22, 20)))
        self.assertTrue(np.all(result[66, 50] == (132, 82, 62)))
        self.assertTrue(np.all(result[30, 5] == target))

    def test_smooth_replacement_preserves_neck_mislabeled_as_background(self):
        pixels = np.full((120, 100, 3), (248, 248, 248), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[12:62, 30:70] = 2
        labels[62:120, 12:88] = 5
        pixels[12:62, 30:70] = (22, 20, 18)
        pixels[62:120, 12:88] = (32, 42, 72)
        # A parser crack from the backdrop into the side of the neck must not
        # turn skin into a coloured notch or a dark compositing seam.
        pixels[54:80, 42:58] = (154, 101, 76)
        labels[54:80, 42:58] = 0
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 64)), (154, 101, 76))
        self.assertEqual(result.getpixel((50, 72)), (154, 101, 76))

    def test_smooth_replacement_preserves_unlabelled_lower_sleeve(self):
        pixels = np.full((120, 100, 3), (200, 195, 188), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[12:65, 30:70] = 2
        labels[62:120, 4:96] = 5
        labels[72:112, 4:24] = 0
        pixels[12:65, 30:70] = (25, 23, 21)
        # Simulate an internal SCHP crack through a white sleeve.
        pixels[72:112, 4:24] = (242, 242, 240)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((10, 95)), (242, 242, 240))

    def test_smooth_replacement_removes_upper_backdrop_falsely_labelled_as_arm(self):
        pixels = np.full((120, 100, 3), (210, 210, 214), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[18:72, 28:72] = 2
        labels[72:, 12:88] = 5
        pixels[18:72, 28:72] = (24, 22, 20)
        pixels[72:, 12:88] = (238, 238, 238)
        # Simulate SCHP's real failure: neutral backdrop beside the hair is
        # incorrectly classified as an arm in the upper half of the portrait.
        labels[28:68, 72:92] = 14
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((82, 42)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 42)), (24, 22, 20))
        self.assertEqual(result.getpixel((50, 100)), (238, 238, 238))

    def test_smooth_replacement_removes_flat_backdrop_inside_clothing_hull(self):
        pixels = np.full((120, 100, 3), (250, 250, 250), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[15:62, 34:66] = 2
        labels[62:120, 4:42] = 5
        labels[62:120, 58:96] = 5
        pixels[15:62, 34:66] = (24, 22, 20)
        pixels[62:120, 4:42] = (28, 44, 82)
        pixels[62:120, 58:96] = (28, 44, 82)
        # The flat white wedge lies inside the convex garment hull but is still
        # connected backdrop, not fabric.
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 75)), (4, 126, 246))
        self.assertEqual(result.getpixel((20, 90)), (28, 44, 82))

    def test_chromatic_backdrop_halo_is_not_protected_as_shirt(self):
        pixels = np.full((120, 100, 3), (18, 184, 220), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[18:64, 34:66] = 13
        labels[64:120, 12:88] = 5
        pixels[18:64, 34:66] = (145, 100, 77)
        pixels[64:120, 12:88] = (35, 48, 76)
        # Generated cyan halo is incorrectly included in the shirt label and
        # convex hull, but retains the border backdrop's hue and chroma.
        labels[68:105, 8:14] = 5
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        corrected_halo = np.asarray(result.getpixel((10, 85)), dtype=np.float32)
        self.assertLess(
            np.linalg.norm(corrected_halo - np.asarray((4, 126, 246), dtype=np.float32)),
            35.0,
        )
        self.assertEqual(result.getpixel((50, 40)), (145, 100, 77))
        self.assertEqual(result.getpixel((40, 90)), (35, 48, 76))

    def test_chromatic_backdrop_removes_local_pale_semantic_fringe(self):
        target = (4, 126, 246)
        pixels = np.full((100, 80, 3), target, dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.uint8)
        labels[16:66, 25:55] = 2
        labels[60:100, 12:68] = 5
        pixels[16:66, 25:55] = (24, 22, 20)
        pixels[60:100, 12:68] = (35, 48, 76)
        # A two-pixel decoder rim remains semantic background. It is locally
        # adjacent to the blue backdrop but cannot be found by hue matching.
        pixels[30:58, 23:25] = (232, 232, 234)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), target)
        self.assertEqual(result.getpixel((24, 42)), target)
        self.assertEqual(result.getpixel((30, 42)), (24, 22, 20))
        self.assertEqual(result.getpixel((40, 85)), (35, 48, 76))

    def test_lower_garment_silhouette_discards_one_pixel_decoder_fringe(self):
        target = (4, 126, 246)
        pixels = np.full((120, 100, 3), target, dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[12:58, 34:66] = 13
        labels[58:120, 17:83] = 5
        pixels[12:58, 34:66] = (140, 92, 70)
        pixels[58:120, 18:82] = (88, 42, 34)
        # This cyan-white decoded sample is attached to the real shirt and can
        # therefore survive both a colour flood fill and a semantic mask.
        pixels[78:108, 17] = (126, 196, 238)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), target)
        self.assertEqual(result.getpixel((17, 90)), target)
        self.assertEqual(result.getpixel((24, 90)), (88, 42, 34))
        self.assertEqual(result.getpixel((50, 35)), (140, 92, 70))

    def test_tiny_detached_crown_sliver_is_removed(self):
        pixels = np.full((120, 100, 3), (18, 184, 220), dtype=np.uint8)
        labels = np.zeros((120, 100), dtype=np.uint8)
        labels[18:70, 30:70] = 2
        pixels[18:70, 30:70] = (22, 20, 18)
        pixels[8:10, 43:57] = (130, 105, 88)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 8)), (4, 126, 246))
        self.assertEqual(result.getpixel((50, 35)), (22, 20, 18))

    def test_smooth_replacement_decontaminates_gray_edge(self):
        pixels = np.full((80, 64, 3), (210, 210, 214), dtype=np.uint8)
        labels = np.zeros((80, 64), dtype=np.uint8)
        labels[12:62, 18:46] = 2
        pixels[12:62, 18:46] = (25, 23, 21)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.smooth_helper()(
                owner, Image.fromarray(pixels), (4, 126, 246),
            ))
        # The transition must move toward the requested blue without creating
        # the white/gray blend produced by ordinary alpha compositing.
        edge_pixel = result[30, 17]
        self.assertGreater(int(edge_pixel[2]), int(edge_pixel[0]) + 80)
        self.assertLess(int(edge_pixel[0]), 100)

    def test_smooth_replacement_substitutes_old_backdrop_in_antialiased_edge(self):
        old_background = np.array((220, 220, 220), dtype=np.uint8)
        target = np.array((4, 126, 246), dtype=np.uint8)
        subject = np.array((20, 18, 16), dtype=np.uint8)
        pixels = np.full((80, 64, 3), old_background, dtype=np.uint8)
        labels = np.zeros((80, 64), dtype=np.uint8)
        labels[12:68, 20:44] = 2
        pixels[12:68, 20:44] = subject
        # Simulate a half-covered hair pixel produced by antialiasing against
        # Qwen's old gray backdrop. It is deliberately outside the eroded hair
        # core so the narrow fringe correction is exercised.
        pixels[12:68, 19] = ((subject.astype(np.uint16) + old_background) // 2).astype(np.uint8)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = np.asarray(self.smooth_helper()(
                owner, Image.fromarray(pixels), tuple(target),
            ))
        edge_pixel = result[40, 19]
        expected = ((subject.astype(np.uint16) + target) // 2).astype(np.uint8)
        # The edge should contain the new blue backdrop, not a second blend of
        # the already gray-contaminated source pixel.
        self.assertLess(np.linalg.norm(edge_pixel.astype(float) - expected.astype(float)), 45.0)
        self.assertLess(int(edge_pixel[0]), 70)
        self.assertEqual(tuple(result[40, 30]), tuple(subject))

    def test_smooth_replacement_preserves_bright_pixels_inside_hair_core(self):
        pixels = np.full((100, 80, 3), (210, 210, 214), dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.uint8)
        labels[15:70, 22:58] = 2
        pixels[15:70, 22:48] = (24, 22, 20)
        # A pale strand/accessory inside the parsed hair core must survive. The
        # background pass is not allowed to infer subject identity from luma.
        pixels[20:68, 48:58] = (150, 149, 151)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((53, 40)), (150, 149, 151))
        self.assertEqual(result.getpixel((35, 40)), (24, 22, 20))

    def test_smooth_replacement_preserves_enclosed_forehead_fringe(self):
        pixels = np.full((100, 80, 3), (210, 210, 214), dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.uint8)
        # SCHP can report fine light-brown forehead strands as background. An
        # enclosed region must not be painted solely from colour similarity.
        labels[12:72, 18:62] = 2
        pixels[12:72, 18:62] = (24, 22, 20)
        labels[28:62, 44:54] = 0
        pixels[28:62, 44:54] = (168, 145, 126)
        parser = Mock(return_value={"labels": labels})
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.schp_service": SimpleNamespace(parse=parser)}):
            result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((49, 45)), (168, 145, 126))
        self.assertEqual(result.getpixel((35, 45)), (24, 22, 20))

    def test_rejects_same_hue_with_wrong_brightness(self):
        height, width = 120, 80
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            value = 105 + row * 80 // height
            hsv = np.full((width, 1, 3), (105, 235, value), dtype=np.uint8)
            pixels[row] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(width, 3)
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        self.assertFalse(self.background_check_helper()(owner, Image.fromarray(pixels), (4, 126, 246)))

    def test_rejects_different_saturated_background_hue(self):
        pixels = np.full((120, 80, 3), (40, 190, 75), dtype=np.uint8)
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        self.assertFalse(self.background_check_helper()(owner, Image.fromarray(pixels), (4, 126, 246)))

    def test_flat_background_gate_rejects_textured_scenery(self):
        flat = Image.new("RGB", (80, 120), (2, 111, 189))
        texture = np.zeros((120, 80, 3), dtype=np.uint8)
        yy, xx = np.indices((120, 80))
        texture[:, :, 0] = 110 + (xx * 5 + yy * 3) % 100
        texture[:, :, 1] = 95 + (xx * 2 + yy * 7) % 110
        texture[:, :, 2] = 80 + (xx * 9 + yy * 4) % 120
        helper = self.flat_background_helper()
        self.assertTrue(helper(flat))
        self.assertFalse(helper(Image.fromarray(texture)))

    def test_flat_background_gate_rejects_wrong_selected_color(self):
        helper = self.flat_background_helper()
        wrong = Image.new("RGB", (80, 120), (205, 201, 174))
        correct = Image.new("RGB", (80, 120), (4, 126, 246))
        self.assertFalse(helper(wrong, (4, 126, 246)))
        self.assertTrue(helper(correct, (4, 126, 246)))

    def test_source_matte_clears_border_colour_components_reconnected_by_torso_hull(self):
        source = (Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("connected_scene_leak", source)
        self.assertIn("component & exterior_scene", source)
        self.assertIn("protected_scene_detail", source)
        self.assertIn("side_neck_corridor", source)
        self.assertIn("source_colour_fringe", source)
