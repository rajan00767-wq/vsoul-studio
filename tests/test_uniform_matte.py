import ast
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import unittest
from unittest.mock import Mock, patch

from PIL import Image


class UniformMatteTests(unittest.TestCase):
    def helper(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "pipelines/qwen_edit_pipeline.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_replace_background_with_foreground_matte")
        scope = dict(Image=Image, io=io, Optional=Optional, Tuple=Tuple, Union=Union, logger=Mock())
        exec(compile(ast.Module(body=[method], type_ignores=[]), "matte", "exec"), scope)
        return scope[method.name]

    def test_exact_background_and_original_opaque_pixels(self):
        source = Image.new("RGB", (32, 32), (240, 240, 240))
        foreground = Image.new("RGBA", source.size, (200, 0, 0, 0))
        foreground.putpixel((16, 16), (200, 0, 0, 255))
        stream = io.BytesIO()
        foreground.save(stream, format="PNG")
        service = Mock()
        service.remove_background.return_value = stream.getvalue()
        owner = SimpleNamespace(_parse_bg_color=lambda color: color)
        with patch.dict(sys.modules, {"pipelines.birefnet_service": SimpleNamespace(background_removal=service)}):
            result = self.helper()(owner, source, (4, 126, 246), preserve_foreground_rgb=True, strict=True)
        self.assertEqual(result.getpixel((0, 0)), (4, 126, 246))
        self.assertEqual(result.getpixel((16, 16)), (240, 240, 240))
        self.assertFalse(service.remove_background.call_args.kwargs["use_schp"])
        service.unload.assert_called_once()

    def test_failure_does_not_silently_return_wrong_background(self):
        service = Mock()
        service.remove_background.side_effect = RuntimeError("model unavailable")
        with patch.dict(sys.modules, {"pipelines.birefnet_service": SimpleNamespace(background_removal=service)}):
            with self.assertRaisesRegex(RuntimeError, "raw Qwen output is retained"):
                self.helper()(None, Image.new("RGB", (32, 32)), (4, 126, 246),
                              preserve_foreground_rgb=True, strict=True)
        service.unload.assert_called_once()
