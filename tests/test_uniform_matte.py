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

    def test_smooth_replacement_preserves_pale_hair_flower(self):
        pixels = np.full((80, 64, 3), (158, 158, 158), dtype=np.uint8)
        pixels[28:54, :8] = (126, 126, 126)
        pixels[10:60, 14:50] = (24, 22, 20)
        pixels[12:28, 6:22] = (224, 204, 215)
        pixels[17:22, 11:16] = (170, 150, 55)
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        result = self.smooth_helper()(owner, Image.fromarray(pixels), (4, 126, 246))
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((3, 40)), (4, 126, 246))
        self.assertEqual(result.getpixel((8, 14)), (224, 204, 215))
        self.assertEqual(result.getpixel((32, 30)), (24, 22, 20))

    def test_accepts_same_hue_studio_gradient(self):
        height, width = 120, 80
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            value = 105 + row * 80 // height
            hsv = np.full((width, 1, 3), (105, 235, value), dtype=np.uint8)
            pixels[row] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(width, 3)
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        self.assertTrue(self.background_check_helper()(owner, Image.fromarray(pixels), (4, 126, 246)))

    def test_rejects_different_saturated_background_hue(self):
        pixels = np.full((120, 80, 3), (40, 190, 75), dtype=np.uint8)
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        self.assertFalse(self.background_check_helper()(owner, Image.fromarray(pixels), (4, 126, 246)))
