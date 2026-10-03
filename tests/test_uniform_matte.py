import ast
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
    def background_check_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_selected_solid_background")
        scope = dict(Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "background_check", "exec"), scope)
        return scope[method.name]

    def flat_background_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_has_flat_border_background")
        scope = dict(Image=Image, np=np)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "flat_background", "exec"), scope)
        return scope[method.name]

    def smooth_helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_smooth_border_background")
        scope = dict(Image=Image, np=np, cv2=cv2, Union=Union, Tuple=Tuple)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "smooth", "exec"), scope)
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
