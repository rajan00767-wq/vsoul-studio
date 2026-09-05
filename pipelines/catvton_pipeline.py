"""
CatVTON Virtual Try-On Pipeline
Lightweight, parameter-efficient virtual try-on diffusion pipeline using spatial concatenation.
Integrates with VTONCompositor for authentic biometric identity lock & studio compositing.
"""

from __future__ import annotations

import inspect
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from pipelines.vton_compositor import VTONCompositor
from utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
CATVTON_DIR = MODELS_DIR / "catvton"
OUTPUTS_DIR = ROOT_DIR / "outputs"


class SkipAttnProcessor(torch.nn.Module):
    """Bypasses cross-attention for pure spatial concatenation condition in CatVTON."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return hidden_states


class CatVTONPipeline:
    """
    CatVTON: Concatenation-based Try-On diffusion pipeline.
    Drapes uniform/garment onto the target subject and then applies VTONCompositor
    for face preservation, seamless collar blending, and studio background.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CatVTONPipeline, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self._is_loaded = False
        self.compositor = VTONCompositor()

    def is_available(self) -> bool:
        """True when local CatVTON attention + SD-inpaint UNet weights exist."""
        attn = CATVTON_DIR / "mix-48k-1024" / "attention" / "model.safetensors"
        unet_dir = CATVTON_DIR / "base_inpainting" / "unet"
        unet_w = (unet_dir / "diffusion_pytorch_model.bin").exists() or (
            unet_dir / "diffusion_pytorch_model.safetensors"
        ).exists()
        return attn.is_file() and (unet_dir / "config.json").is_file() and unet_w

    def _local_vae_dir(self) -> str:
        bundled = CATVTON_DIR / "vae"
        if (bundled / "config.json").is_file():
            return str(bundled)
        hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--sd-vae-ft-mse" / "snapshots"
        if hub.is_dir():
            for snap in hub.iterdir():
                if (snap / "config.json").is_file() and (
                    (snap / "diffusion_pytorch_model.safetensors").is_file()
                    or (snap / "diffusion_pytorch_model.bin").is_file()
                ):
                    return str(snap)
        return "stabilityai/sd-vae-ft-mse"

    def _load_model(self):
        """Loads CatVTON VAE, Scheduler, and Inpainting UNet with CatVTON attention weights."""
        if self._is_loaded and self.unet is not None:
            self.unet.to(self.device)
            self.vae.to(self.device)
            return

        with self._lock:
            if self._is_loaded and self.unet is not None:
                self.unet.to(self.device)
                self.vae.to(self.device)
                return

            logger.info("[CatVTON] Initializing CatVTON pipeline on %s (%s)...", self.device, self.dtype)
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from safetensors.torch import load_file

            if not self.is_available():
                raise FileNotFoundError(
                    "CatVTON weights missing. Need models/catvton/mix-48k-1024/attention/model.safetensors "
                    "and models/catvton/base_inpainting/unet/diffusion_pytorch_model.bin"
                )

            base_dir = CATVTON_DIR / "base_inpainting"
            attn_ckpt_path = CATVTON_DIR / "mix-48k-1024" / "attention" / "model.safetensors"
            vae_src = self._local_vae_dir()
            local_vae = Path(vae_src).is_dir()

            self.scheduler = DDIMScheduler.from_pretrained(
                str(base_dir), subfolder="scheduler", local_files_only=True
            )
            self.vae = AutoencoderKL.from_pretrained(
                vae_src, torch_dtype=self.dtype, local_files_only=local_vae
            ).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(
                str(base_dir),
                subfolder="unet",
                torch_dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)

            attn_procs = {}
            for name in self.unet.attn_processors.keys():
                if "attn2" in name:
                    attn_procs[name] = SkipAttnProcessor()
            self.unet.set_attn_processor({**self.unet.attn_processors, **attn_procs})

            raw = load_file(str(attn_ckpt_path))
            ids = sorted({int(k.split(".")[0]) for k in raw})
            attns = [m for n, m in self.unet.named_modules() if n.endswith(".attn1")]
            if len(attns) != len(ids):
                raise RuntimeError(
                    f"CatVTON attn mismatch: unet has {len(attns)} attn1 blocks, ckpt has {len(ids)}"
                )
            for attn, idx in zip(attns, ids):
                def _copy(param, key, _raw=raw):
                    t = _raw[key].to(device=param.device, dtype=param.dtype)
                    if tuple(t.shape) != tuple(param.shape):
                        raise ValueError(f"{key} shape {tuple(t.shape)} != {tuple(param.shape)}")
                    param.data.copy_(t)

                _copy(attn.to_q.weight, f"{idx}.to_q.weight")
                _copy(attn.to_k.weight, f"{idx}.to_k.weight")
                _copy(attn.to_v.weight, f"{idx}.to_v.weight")
                _copy(attn.to_out[0].weight, f"{idx}.to_out.0.weight")
                _copy(attn.to_out[0].bias, f"{idx}.to_out.0.bias")
            logger.info("[CatVTON] Injected self-attention weights into %d UNet blocks", len(attns))

            self.unet.eval()
            self.vae.eval()
            self._is_loaded = True
            logger.info("[CatVTON] Models loaded from local disk (%s, vae=%s)", base_dir, vae_src)

    def offload(self) -> None:
        """Free GPU VRAM so Qwen-Image-Edit 2509 can run after try-on."""
        if self.unet is not None:
            self.unet.to("cpu")
        if self.vae is not None:
            self.vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[CatVTON] Offloaded UNet/VAE to CPU")

    @staticmethod
    def _rgb_on_white(image: Image.Image) -> Image.Image:
        """Keep template alpha/transparency as white — never flatten to black."""
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.split()[-1])
            return canvas
        return image.convert("RGB")

    @staticmethod
    def _checker_pixels(rgb: np.ndarray) -> np.ndarray:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        chroma = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        mean = rgb.mean(axis=2)
        return (chroma < 12) & (mean > 158)

    def _prepare_garment(self, uniform_pil: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """
        CatVTON condition image: the uploaded uniform, large on a white canvas.
        Do not SCHP-keep-only-clothing — navy vest is often labeled background and gets deleted.
        """
        rgb = np.array(self._rgb_on_white(uniform_pil))
        chk = self._checker_pixels(rgb)
        # Transparent canvas becomes pure neutral white.  Keep the very pale
        # blue gingham shirt rather than treating it as empty canvas.
        near_white = (rgb.mean(axis=2) > 250) & (rgb.max(axis=2) - rgb.min(axis=2) < 4)
        keep = ~(chk | near_white)

        schp = self.compositor._get_schp()
        if schp is not None:
            try:
                labels = schp.parse(rgb)["labels"]
                # Drop only a template model's head. Keep vest/shirt even if SCHP says BG.
                keep[np.isin(labels, [2, 4, 13])] = False
                cloth = np.isin(labels, [5, 6, 7, 10, 11, 12])
                keep = keep | cloth
                keep[chk] = False
            except Exception as e:
                logger.warning("[CatVTON] SCHP on garment skipped: %s", e)

        keep = keep.astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, k, iterations=1)

        canvas = np.full_like(rgb, 255)
        canvas[keep > 0] = rgb[keep > 0]
        kept = int(keep.sum())
        total = int(keep.size)
        logger.info("[CatVTON] Template garment kept %d px (%.1f%%)", kept, 100.0 * kept / max(1, total))

        ys, xs = np.where(keep > 0)
        if len(xs) > 200:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            pad = int(0.04 * max(y1 - y0 + 1, x1 - x0 + 1, 32))
            y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
            y1, x1 = min(canvas.shape[0], y1 + pad + 1), min(canvas.shape[1], x1 + pad + 1)
            crop = Image.fromarray(canvas[y0:y1, x0:x1], mode="RGB")
        else:
            logger.warning("[CatVTON] Template garment mask empty — using full image on white")
            crop = Image.fromarray(canvas, mode="RGB")
        # Fill most of 768×1024 so CatVTON reads the real vest/shirt, not a tiny postage stamp.
        return self._resize_and_padding(crop, size, fill=0.94)

    @staticmethod
    def _resize_and_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        w, h = image.size
        target_w, target_h = size
        if w / h < target_w / target_h:
            new_w = w
            new_h = w * target_h // target_w
        else:
            new_h = h
            new_w = h * target_w // target_h
        image = image.crop(((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2))
        return image.resize(size, Image.LANCZOS)

    @staticmethod
    def _resize_and_padding(
        image: Image.Image,
        size: Tuple[int, int],
        fill: float = 1.0,
    ) -> Image.Image:
        w, h = image.size
        target_w, target_h = size
        if w / h < target_w / target_h:
            new_h = max(1, int(target_h * fill))
            new_w = max(1, w * new_h // h)
        else:
            new_w = max(1, int(target_w * fill))
            new_h = max(1, h * new_w // w)
        if new_w > target_w or new_h > target_h:
            scale = min(target_w / new_w, target_h / new_h)
            new_w = max(1, int(new_w * scale))
            new_h = max(1, int(new_h * scale))
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", size, (255, 255, 255))
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return canvas

    @staticmethod
    def _clothing_keep(rgb: np.ndarray) -> np.ndarray:
        """Keep vest/gingham; drop white canvas, checkerboard, and neck-hole fill."""
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        chroma = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        mean = rgb.mean(axis=2)
        pale_blue = (b > r + 5) & (b > g + 2) & (mean > 205)
        whiteish = ((rgb.min(axis=2) > 242) & ~pale_blue) | ((chroma < 8) & (mean > 210))
        return ((~whiteish).astype(np.uint8)) * 255

    def _tint_worn_garment(
        self,
        worn_pil: Image.Image,
        garment_pil: Image.Image,
        mask_pil: Image.Image,
    ) -> Image.Image:
        """Keep CatVTON folds/pose; pull fabric colours from the downloaded template (not a pixel paste)."""
        worn = np.array(worn_pil.convert("RGB"))
        garment = np.array(garment_pil.convert("RGB").resize(worn_pil.size, Image.LANCZOS))
        body = np.array(mask_pil.convert("L").resize(worn_pil.size, Image.NEAREST)) > 127
        keep = self._clothing_keep(garment) > 127
        region = body
        if int(keep.sum()) > 400:
            region = body
        if int(region.sum()) < 400 or int(keep.sum()) < 400:
            return worn_pil
        w_lab = cv2.cvtColor(worn, cv2.COLOR_RGB2LAB).astype(np.float32)
        g_lab = cv2.cvtColor(garment, cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            mu_g = float(g_lab[:, :, c][keep].mean())
            sd_g = float(g_lab[:, :, c][keep].std()) + 1e-5
            mu_w = float(w_lab[:, :, c][region].mean())
            sd_w = float(w_lab[:, :, c][region].std()) + 1e-5
            mapped = (w_lab[:, :, c] - mu_w) / sd_w * (0.5 * sd_w + 0.5 * sd_g) + (0.35 * mu_w + 0.65 * mu_g)
            w_lab[:, :, c] = np.where(region, np.clip(mapped, 0, 255), w_lab[:, :, c])
        out = cv2.cvtColor(w_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        logger.info("[CatVTON] Worn-garment colour matched to uploaded template")
        return Image.fromarray(out, mode="RGB")

    def _blend_template_chroma(
        self,
        worn_pil: Image.Image,
        garment_pil: Image.Image,
        mask_pil: Image.Image,
        person_pil: Image.Image,
    ) -> Image.Image:
        """Keep CatVTON folds; copy the uploaded uniform's real colours/pattern (vest + check)."""
        worn = np.array(worn_pil.convert("RGB"))
        person = np.array(person_pil.convert("RGB"))
        garment = np.array(garment_pil.convert("RGB"))
        mask = np.array(mask_pil.convert("L"))
        h, w = worn.shape[:2]
        if garment.shape[:2] != (h, w):
            garment = cv2.resize(garment, (w, h), interpolation=cv2.INTER_LANCZOS4)
        g_keep = self._clothing_keep(garment)
        p_pts = self._shoulder_points(mask)
        g_pts = self._shoulder_points(g_keep)
        if p_pts is None or g_pts is None:
            logger.warning("[CatVTON] Template chroma lock skipped — no shoulder anchors")
            return worn_pil

        p_left, p_right, p_neck = p_pts
        try:
            app = self.compositor._get_insightface()
            if app is not None:
                faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR))
                if faces:
                    f = max(faces, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
                    p_neck = (int((p_left[0] + p_right[0]) * 0.5), int(f.bbox[3]))
        except Exception:
            pass

        src = np.float32([g_pts[0], g_pts[1], g_pts[2]])
        dst = np.float32([p_left, p_right, p_neck])
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            M = cv2.getAffineTransform(src, dst)

        warped = cv2.warpAffine(garment, M, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))
        alpha = cv2.warpAffine(g_keep, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        schp = self.compositor._get_schp()
        if schp is not None:
            try:
                labels = schp.parse(person)
                alpha[np.isin(labels["labels"], [2, 4, 13])] = 0
            except Exception:
                pass
        body = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
        alpha = np.minimum(alpha, body)
        a = cv2.GaussianBlur(alpha, (9, 9), 0).astype(np.float32) / 255.0
        a = np.clip(a, 0, 1)[:, :, None]

        worn_lab = cv2.cvtColor(worn, cv2.COLOR_RGB2LAB).astype(np.float32)
        warp_lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
        mixed = worn_lab.copy()
        mixed[:, :, 0] = worn_lab[:, :, 0] * 0.55 + warp_lab[:, :, 0] * 0.45
        mixed[:, :, 1] = warp_lab[:, :, 1]
        mixed[:, :, 2] = warp_lab[:, :, 2]
        mixed_rgb = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        out = (mixed_rgb * a + worn.astype(np.float32) * (1.0 - a)).clip(0, 255).astype(np.uint8)
        logger.info("[CatVTON] Locked uploaded uniform colours/pattern onto worn folds")
        return Image.fromarray(out, mode="RGB")

    @staticmethod
    def _shoulder_points(mask_u8: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        m = mask_u8 > 127
        ys, xs = np.where(m)
        if len(xs) < 80:
            return None
        h, w = mask_u8.shape[:2]
        y0, y1 = int(ys.min()), int(ys.max())
        y_sh = int(np.clip(y0 + 0.10 * max(1, y1 - y0), 0, h - 1))
        xr = np.where(m[y_sh])[0]
        if len(xr) < 8:
            return None
        left = (int(xr.min()), y_sh)
        right = (int(xr.max()), y_sh)
        cx = (left[0] + right[0]) // 2
        x0, x1 = max(0, cx - w // 12), min(w, cx + w // 12)
        col = m[:, x0:x1]
        if col.any():
            neck_y = int(np.where(col.any(axis=1))[0][0])
        else:
            neck_y = y0
        return left, right, (cx, neck_y)

    def _person_alpha(self, person_rgb: np.ndarray) -> np.ndarray:
        h, w = person_rgb.shape[:2]
        try:
            from pipelines.birefnet_service import background_removal
            im = Image.fromarray(person_rgb)
            with BytesIO() as bio:
                im.save(bio, format="PNG")
                fg = Image.open(BytesIO(background_removal.remove_background(bio.getvalue()))).convert("RGBA")
            a = np.array(fg.split()[-1])
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            return a
        except Exception as e:
            logger.warning("[CatVTON] Person silhouette fallback: %s", e)
            g = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2GRAY)
            return ((g < 245) & (g > 12)).astype(np.uint8) * 255

    @staticmethod
    def _pose_shoulders(person_rgb: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return MediaPipe shoulder landmarks when a torso is visible."""
        try:
            import mediapipe as mp

            h, w = person_rgb.shape[:2]
            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            ) as pose:
                result = pose.process(person_rgb)
            if not result.pose_landmarks:
                return None
            lm = result.pose_landmarks.landmark
            left = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            if min(left.visibility, right.visibility) < 0.45:
                return None
            return (
                (int(np.clip(left.x * w, 0, w - 1)), int(np.clip(left.y * h, 0, h - 1))),
                (int(np.clip(right.x * w, 0, w - 1)), int(np.clip(right.y * h, 0, h - 1))),
            )
        except Exception:
            return None

    def _place_uniform_on_torso(
        self,
        person_pil: Image.Image,
        garment_pil: Image.Image,
        mask_pil: Image.Image,
        lighting_pil: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Warp the uploaded vest/shirt onto the torso silhouette (collar at chin)."""
        person = np.array(person_pil.convert("RGB"))
        garment = np.array(garment_pil.convert("RGB"))
        h, w = person.shape[:2]
        if garment.shape[:2] != (h, w):
            garment = cv2.resize(garment, (w, h), interpolation=cv2.INTER_LANCZOS4)

        keep = self._clothing_keep(garment)
        keep[self._checker_pixels(garment)] = 0
        ys, xs = np.where(keep > 127)
        if len(xs) < 80:
            logger.warning("[CatVTON] Uniform placement skipped — empty garment")
            return person_pil
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        g_rgb = garment[y0:y1, x0:x1]
        g_a = keep[y0:y1, x0:x1]
        g_a[self._checker_pixels(g_rgb)] = 0
        gh, gw = g_a.shape[:2]
        ff = (g_a <= 40).astype(np.uint8)
        flood = np.zeros((gh + 2, gw + 2), np.uint8)
        for sx in (0, gw // 2, gw - 1):
            if ff[0, min(max(sx, 0), gw - 1)] > 0:
                cv2.floodFill(ff, flood, (int(min(max(sx, 0), gw - 1)), 0), 2)
        g_a[ff == 2] = 0

        face_cx, chin_y, face_w, face_h = w // 2, int(h * 0.42), int(w * 0.28), int(h * 0.28)
        try:
            app = self.compositor._get_insightface()
            if app is not None:
                faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR))
                if faces:
                    f = max(faces, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
                    x1f, y1f, x2f, y2f = [int(v) for v in f.bbox]
                    face_w = max(24, x2f - x1f)
                    face_h = max(24, y2f - y1f)
                    face_cx = (x1f + x2f) // 2
                    chin_y = y2f
        except Exception as e:
            logger.warning("[CatVTON] Face box for uniform place failed: %s", e)

        sil = self._person_alpha(person)
        body = sil > 40
        body_rows = np.where(body.any(axis=1))[0]
        if len(body_rows) == 0:
            return person_pil
        y_end = int(body_rows[-1])
        sh_y = int(np.clip(chin_y + 0.35 * face_h, 0, h - 1))
        sh_cols = np.where(body[sh_y])[0]
        if len(sh_cols) > 8:
            shoulder_w = int(sh_cols.max() - sh_cols.min())
        else:
            shoulder_w = int(face_w * 2.2)
        shoulder_w = int(np.clip(shoulder_w, face_w * 2.15, min(w * 0.90, face_w * 3.10)))

        # Transform the complete reference garment once, using the actual
        # shoulder span and the base of the detected neck.  The old row-by-row
        # warp stretched each collar row independently, producing a second
        # shoulder and an adult-sized collar on small children.
        # Seat the collar just below the detected chin.  This keeps the neck
        # proportionate to the uniform instead of leaving an elongated gap.
        neck_y = int(np.clip(chin_y + 0.04 * face_h, 0, h - 2))
        target_bottom = max(neck_y + int(1.15 * face_h), y_end)
        target_bottom = int(np.clip(target_bottom, neck_y + 24, h))
        target_h = target_bottom - neck_y
        warped = np.zeros_like(person)
        alpha = np.zeros((h, w), dtype=np.float32)
        source_anchors = self._shoulder_points(g_a)
        pose_shoulders = self._pose_shoulders(person)

        if source_anchors is not None and pose_shoulders is not None:
            src_left, src_right, src_neck = source_anchors
            dst_left, dst_right = pose_shoulders
            # Template shoulder/collar anchors become the person's measured
            # shoulder line and lower-neck point. This follows portrait tilt
            # and shoulder asymmetry instead of using a fixed rectangular fit.
            src = np.float32([src_left, src_right, src_neck])
            dst = np.float32([dst_left, dst_right, (face_cx, neck_y)])
            matrix = cv2.getAffineTransform(src, dst)
            warped = cv2.warpAffine(
                g_rgb, matrix, (w, h), flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            alpha = cv2.warpAffine(
                g_a, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.float32)
            coverage = float(np.mean((alpha > 32) & body))
            if coverage >= 0.12:
                logger.info("[CatVTON] Applied pose-guided collar/shoulder affine warp (%.1f%%)", coverage * 100)
            else:
                logger.warning("[CatVTON] Rejected sparse pose warp (%.1f%%); using silhouette fit", coverage * 100)
                warped.fill(0)
                alpha.fill(0)

        if not np.any(alpha > 32):
            target_w = int(np.clip(shoulder_w, 32, w))
            scaled_rgb = cv2.resize(g_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_alpha = cv2.resize(g_a, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            x_start = int(np.clip(face_cx - target_w // 2, 0, max(0, w - target_w)))
            x_end = x_start + target_w
            warped[neck_y:target_bottom, x_start:x_end] = scaled_rgb
            alpha[neck_y:target_bottom, x_start:x_end] = scaled_alpha
            logger.info("[CatVTON] Pose landmarks unavailable; using silhouette fit")
        alpha *= body.astype(np.float32)
        collar_y0, out_bot = neck_y, target_bottom

        # Templates arrive in two forms: a garment with a real neck opening, or
        # a closed collar photographed on a mannequin/person.  A closed collar
        # has fabric through the centre of its top band.  It cannot be pasted
        # over a bare target neck, so reveal the target neck only in that case.
        top_band_h = max(8, int(gh * 0.22))
        centre_l, centre_r = int(gw * 0.40), int(gw * 0.60)
        source_centre = g_a[:top_band_h, centre_l:centre_r] > 80
        closed_collar = bool(source_centre.size and source_centre.mean() > 0.55)
        neck_window = np.zeros((h, w), dtype=np.uint8)
        neck_reveal = np.zeros((h, w), dtype=np.float32)
        if closed_collar:
            neck_cy = int(np.clip(chin_y + 0.10 * face_h, 0, h - 1))
            cv2.ellipse(
                neck_window,
                (face_cx, neck_cy),
                (max(8, int(face_w * 0.24)), max(7, int(face_h * 0.13))),
                0,
                0,
                360,
                255,
                -1,
            )
            # The opening should stop before the shirt placket; collar wings
            # and the vest remain in front of the person.
            neck_window[: max(0, chin_y - int(face_h * 0.04)), :] = 0
            neck_window[int(neck_y + target_h * 0.18) :, :] = 0
            neck_reveal = cv2.GaussianBlur(neck_window, (9, 9), 2.0).astype(np.float32) / 255.0
            logger.info("[CatVTON] Closed collar detected; retained target neck through collar opening")
        hair = np.zeros((h, w), dtype=bool)
        protect = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            protect,
            (face_cx, chin_y - face_h // 2),
            (int(face_w * 0.68), int(face_h * 0.90)),
            0, 0, 360, 255, -1,
        )
        protect[chin_y:, :] = 0
        alpha[protect > 0] = 0
        alpha *= (1.0 - neck_reveal)
        # Neutral grey canvas remnants are not garment fabric.  Reject them
        # before compositing so they cannot become rectangular shoulder blocks.
        warped_chroma = (
            np.maximum(
                np.abs(warped[:, :, 0].astype(np.int16) - warped[:, :, 1].astype(np.int16)),
                np.maximum(
                    np.abs(warped[:, :, 1].astype(np.int16) - warped[:, :, 2].astype(np.int16)),
                    np.abs(warped[:, :, 0].astype(np.int16) - warped[:, :, 2].astype(np.int16)),
                ),
            )
        )
        warped_mean = warped.mean(axis=2)
        alpha[(warped_chroma < 8) & (warped_mean > 118)] = 0
        schp = self.compositor._get_schp()
        if schp is not None:
            try:
                labels = schp.parse(person)["labels"]
                hair = labels == 2
                hair_above_neck = hair & (np.arange(h)[:, None] < chin_y)
                alpha[hair_above_neck] *= 0.12
            except Exception:
                pass

        a = cv2.GaussianBlur(alpha, (3, 3), 0)
        a = np.clip(a / 255.0, 0.0, 1.0)
        worn = warped
        a3 = a[:, :, None]
        # The person photo's original dress may extend beyond the template
        # sleeves.  Clear only that lower clothing field before blending so it
        # cannot remain as a white double shoulder in the finished portrait.
        base = person.copy()
        tryon_mask = np.array(mask_pil.convert("L")) > 127
        # Preserve the neck and collar opening.  Clear only lower source
        # clothing that falls outside the fitted template.
        cleanup_y = neck_y + int(0.26 * target_h)
        below_clothing_line = np.arange(h)[:, None] >= cleanup_y
        leftover_clothes = tryon_mask & body & below_clothing_line & ~hair & (a < 0.08) & (neck_window == 0)
        base[leftover_clothes] = (205, 230, 248)
        out = (worn.astype(np.float32) * a3 + base.astype(np.float32) * (1.0 - a3)).clip(0, 255).astype(np.uint8)
        # The person alpha was already calculated for fit measurement.  Use it
        # here instead of asking a second matting pass to segment a synthetic
        # garment image, which previously left source-background fragments at
        # the hair and sleeve boundaries.
        subject_alpha = alpha.clip(0, 255) / 255.0
        # Below the collar, only the fitted template is foreground.  Retaining
        # the original person matte there brought outdoor background objects
        # through as false shoulders beside the vest.
        subject_alpha[:chin_y, :] = sil[:chin_y, :] / 255.0
        neck_subject = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            neck_subject,
            (face_cx, int(chin_y + 0.16 * face_h)),
            (max(10, int(face_w * 0.31)), max(10, int(face_h * 0.30))),
            0, 0, 360, 255, -1,
        )
        neck_subject[: max(0, chin_y - int(face_h * 0.05)), :] = 0
        subject_alpha[neck_subject > 0] = np.maximum(
            subject_alpha[neck_subject > 0], sil[neck_subject > 0] / 255.0
        )
        # Reject common outdoor-background intrusions that BiRefNet can retain
        # inside loose hair silhouettes.  This is restricted to saturated green
        # pixels and outside a conservative head/torso envelope, so uniform
        # blues, skin, hair, and badges are never keyed out.
        head_envelope = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            head_envelope,
            (face_cx, int(chin_y - face_h * 0.45)),
            (max(16, int(face_w * 0.92)), max(20, int(face_h * 1.18))),
            0, 0, 360, 255, -1,
        )
        torso_envelope = np.zeros((h, w), dtype=np.uint8)
        half_shoulder = max(18, int(shoulder_w * 0.54))
        cv2.fillConvexPoly(
            torso_envelope,
            np.array([
                (max(0, face_cx - half_shoulder), sh_y),
                (min(w - 1, face_cx + half_shoulder), sh_y),
                (min(w - 1, face_cx + int(half_shoulder * 1.10)), h - 1),
                (max(0, face_cx - int(half_shoulder * 1.10)), h - 1),
            ], dtype=np.int32),
            255,
        )
        envelope = (head_envelope > 0) | (torso_envelope > 0)
        green_spill = (
            (out[:, :, 1].astype(np.int16) > out[:, :, 0].astype(np.int16) + 18)
            & (out[:, :, 1].astype(np.int16) > out[:, :, 2].astype(np.int16) + 12)
        )
        subject_alpha[green_spill & ~envelope] *= 0.03
        subject_alpha[~envelope] *= 0.18
        subject_alpha = cv2.GaussianBlur(subject_alpha, (5, 5), 1.2)[:, :, None]
        studio = np.full_like(out, (205, 230, 248))
        out = (out.astype(np.float32) * subject_alpha + studio.astype(np.float32) * (1.0 - subject_alpha)).clip(0, 255).astype(np.uint8)
        logger.info(
            "[CatVTON] Wore uploaded uniform on torso (alpha %.1f%%, shoulders %d, y %d-%d)",
            100.0 * float(a.mean()), shoulder_w, collar_y0, out_bot,
        )
        return Image.fromarray(out, mode="RGB")

    def _natural_uniform_finish(
        self,
        person_pil: Image.Image,
        dressed_pil: Image.Image,
        tryon_mask: Image.Image,
    ) -> Image.Image:
        """Repair only high-contrast garment seam defects; never repaint identity."""
        person = np.array(person_pil.convert("RGB"))
        dressed = np.array(dressed_pil.convert("RGB"))
        h, w = dressed.shape[:2]
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LANCZOS4)

        diff = cv2.cvtColor(cv2.absdiff(dressed, person), cv2.COLOR_RGB2GRAY)
        changed = ((diff > 22).astype(np.uint8) * 255)
        torso = np.array(tryon_mask.convert("L").resize((w, h), Image.NEAREST))
        changed = cv2.bitwise_and(changed, torso)
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )

        # Limit repair to the thin seam ring and reject all face/head pixels.
        outer = cv2.dilate(changed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        inner = cv2.erode(changed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        seam = cv2.subtract(outer, inner)
        try:
            app = self.compositor._get_insightface()
            if app is not None:
                faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR))
                if faces:
                    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    cv2.ellipse(
                        seam,
                        ((x1 + x2) // 2, (y1 + y2) // 2),
                        (max(8, int((x2 - x1) * 0.85)), max(8, int((y2 - y1) * 1.05))),
                        0, 0, 360, 0, -1,
                    )
                    seam[: min(h, int(y2 + 0.08 * (y2 - y1))), :] = 0
        except Exception:
            seam[: int(h * 0.42), :] = 0

        # Repair only conspicuous local discontinuities, retaining the actual
        # garment pixels everywhere else.
        edge = cv2.Canny(cv2.cvtColor(dressed, cv2.COLOR_RGB2GRAY), 55, 140)
        repair = cv2.bitwise_and(seam, edge)
        if cv2.countNonZero(repair) < 40:
            return dressed_pil
        repaired = cv2.inpaint(dressed, repair, 2, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(repair, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
        finished = (repaired.astype(np.float32) * alpha + dressed.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        logger.info("[CatVTON] Applied natural seam finish to %d pixels", cv2.countNonZero(repair))
        return Image.fromarray(finished, mode="RGB")

    def _lock_source_identity(
        self,
        person_pil: Image.Image,
        dressed_pil: Image.Image,
    ) -> Image.Image:
        """Restore the original face and hair after deterministic garment fitting."""
        person = np.array(person_pil.convert("RGB"))
        dressed = np.array(dressed_pil.convert("RGB"))
        h, w = dressed.shape[:2]
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LANCZOS4)

        try:
            app = self.compositor._get_insightface()
            if app is None:
                return dressed_pil
            faces = app.get(cv2.cvtColor(person, cv2.COLOR_RGB2BGR))
            if not faces:
                return dressed_pil
            face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
        except Exception as e:
            logger.warning("[CatVTON] Source identity lock skipped: %s", e)
            return dressed_pil

        face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
        chin_y = min(h - 1, y2)
        face_identity = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            face_identity,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            (max(8, int(face_w * 0.60)), max(8, int(face_h * 0.72))),
            0,
            0,
            360,
            255,
            -1,
        )

        # Keep real hair around the face, but stop above the garment collar.
        hair_identity = np.zeros((h, w), dtype=bool)
        try:
            schp = self.compositor._get_schp()
            if schp is not None:
                hair = schp.parse(person)["labels"] == 2
                collar_centre = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(
                    collar_centre,
                    ((x1 + x2) // 2, chin_y + int(face_h * 0.28)),
                    (max(8, int(face_w * 0.38)), max(8, int(face_h * 0.34))),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )
                hair_identity = hair & (collar_centre == 0)
        except Exception:
            pass

        # Face pixels must never depend on matting confidence. Only loose hair
        # is silhouette-gated so the original photo background cannot bleed in.
        silhouette = self._person_alpha(person) > 128
        identity = face_identity.copy()
        identity[hair_identity & silhouette] = 255
        if cv2.countNonZero(identity) < 100:
            return dressed_pil
        alpha = cv2.GaussianBlur(identity, (5, 5), 1.0).astype(np.float32)[:, :, None] / 255.0
        locked = (person.astype(np.float32) * alpha + dressed.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        logger.info("[CatVTON] Restored original face/hair identity (%d px)", cv2.countNonZero(identity))
        return Image.fromarray(locked, mode="RGB")

    def _drape_template_exact(
        self,
        person_pil: Image.Image,
        garment_pil: Image.Image,
        mask_pil: Image.Image,
        lighting_pil: Optional[Image.Image] = None,
    ) -> Image.Image:
        return self._place_uniform_on_torso(person_pil, garment_pil, mask_pil, lighting_pil)

    def tryon(
        self,
        person_image: Union[Image.Image, str, Path],
        uniform_image: Union[Image.Image, str, Path],
        mask_image: Optional[Image.Image] = None,
        steps: int = 25,
        guidance_scale: float = 2.5,
        seed: int = 42,
        bg_color: Tuple[int, int, int] = (4, 126, 246),
        job_id: str = "catvton_job",
        apply_compositing: bool = True,
        exact_drape: bool = False,
    ) -> Path:
        """
        Execute CatVTON Virtual Try-On followed by VTON Image Compositing:
        1. Preprocess person and uniform images.
        2. Generate human-agnostic clothing mask (using SCHP / landmark parsing).
        3. Concatenate garment + masked person in latent space.
        4. Denoise with CatVTON UNet.
        5. Composite with authentic facial biometric identity lock and studio background.
        """
        # Load images. Uniform alpha/checkerboard must become white, never black.
        if isinstance(person_image, (str, Path)):
            person_raw = Image.open(person_image)
        else:
            person_raw = person_image
        person_pil = person_raw.convert("RGB") if person_raw.mode == "RGB" else self._rgb_on_white(person_raw)

        if isinstance(uniform_image, (str, Path)):
            uniform_raw = Image.open(uniform_image)
        else:
            uniform_raw = uniform_image
        uniform_pil = self._rgb_on_white(uniform_raw)

        # ── 1. Prepare Inputs & Agnostic Mask ────────────────────────────────
        # Official CatVTON canvas is 768x1024. Pad (don't center-crop) so odd poses
        # and full-body shots keep the head and shoulders.
        person_cropped = self._resize_and_padding(person_pil, (768, 1024))
        person_prepped, auto_mask, _masked_person = self.compositor.generate_tryon_mask(
            person_cropped, target_size=(768, 1024)
        )
        mask_pil = mask_image.resize((768, 1024), Image.NEAREST) if mask_image is not None else auto_mask
        uniform_prepped = self._prepare_garment(uniform_pil, (768, 1024))
        try:
            uniform_prepped.save(str(OUTPUTS_DIR / f"{job_id}_garment_cond.png"), "PNG")
            mask_pil.save(str(OUTPUTS_DIR / f"{job_id}_tryon_mask.png"), "PNG")
        except Exception:
            pass

        # ── 2. Wear the uploaded uniform ──────────────────────────────────────
        tryon_result_pil = person_prepped
        if exact_drape:
            logger.info("[CatVTON] Placing uploaded uniform on the torso")
            try:
                draped = self._place_uniform_on_torso(
                    person_prepped, uniform_prepped, mask_pil, None
                )
                tryon_result_pil = self._natural_uniform_finish(person_prepped, draped, mask_pil)
                tryon_result_pil = self._lock_source_identity(person_prepped, tryon_result_pil)
                if os.getenv("UNIFORM_QWEN_SEAM", "0") == "1":
                    try:
                        from pipelines.studio_uniform_pipeline import studio_uniform_pipeline
                        tryon_result_pil = studio_uniform_pipeline.finish_uniform_seams_qwen(
                            person_prepped, tryon_result_pil, mask_pil
                        )
                    except Exception as e:
                        logger.warning("[CatVTON] Qwen seam finish unavailable: %s", e)
                else:
                    logger.info("[CatVTON] Skipping optional Qwen seam repaint; deterministic collar geometry is active")
                draped.save(str(OUTPUTS_DIR / f"{job_id}_template_drape.png"), "PNG")
            except Exception as e:
                logger.warning("[CatVTON] Uniform placement failed: %s", e)
        else:
            try:
                self._load_model()
                logger.info("[CatVTON] Running diffusion denoising (%d steps, official H-concat)...", steps)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                concat_dim = -2
                person_np = np.array(person_prepped.convert("RGB"))
                garment_np = np.array(uniform_prepped.convert("RGB"))
                mask_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
                mask_np = (mask_np >= 0.5).astype(np.float32)
                person_tensor = torch.from_numpy(person_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                garment_tensor = torch.from_numpy(garment_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                mask_tensor = torch.from_numpy(mask_np)[None, None]
                person_tensor = person_tensor.to(self.device, dtype=self.dtype)
                garment_tensor = garment_tensor.to(self.device, dtype=self.dtype)
                mask_tensor = mask_tensor.to(self.device, dtype=self.dtype)
                masked_image = person_tensor * (mask_tensor < 0.5)
                with torch.inference_mode():
                    masked_latent = self.vae.encode(masked_image).latent_dist.sample() * self.vae.config.scaling_factor
                    condition_latent = self.vae.encode(garment_tensor).latent_dist.sample() * self.vae.config.scaling_factor
                    mask_latent = torch.nn.functional.interpolate(
                        mask_tensor, size=masked_latent.shape[-2:], mode="nearest"
                    )
                    masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
                    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
                    do_cfg = guidance_scale > 1.0
                    unet_condition = masked_latent_concat
                    unet_mask = mask_latent_concat
                    if do_cfg:
                        uncond = torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim)
                        unet_condition = torch.cat([uncond, masked_latent_concat], dim=0)
                        unet_mask = torch.cat([mask_latent_concat, mask_latent_concat], dim=0)
                    noise = torch.randn(
                        masked_latent_concat.shape,
                        generator=generator,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    self.scheduler.set_timesteps(steps, device=self.device)
                    latents = noise * self.scheduler.init_noise_sigma
                    cross_dim = getattr(self.unet.config, "cross_attention_dim", 768) or 768
                    dummy_embeds = torch.zeros(
                        (2 if do_cfg else 1, 77, cross_dim),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    extra_step_kwargs = {}
                    step_params = inspect.signature(self.scheduler.step).parameters
                    if "eta" in step_params:
                        extra_step_kwargs["eta"] = 1.0
                    if "generator" in step_params:
                        extra_step_kwargs["generator"] = generator
                    for t in self.scheduler.timesteps:
                        latent_in = torch.cat([latents] * 2, dim=0) if do_cfg else latents
                        latent_in = self.scheduler.scale_model_input(latent_in, t)
                        model_input = torch.cat([latent_in, unet_mask, unet_condition], dim=1)
                        noise_pred = self.unet(
                            model_input,
                            t,
                            encoder_hidden_states=dummy_embeds[: latent_in.shape[0]],
                        ).sample
                        if do_cfg:
                            noise_uncond, noise_cond = noise_pred.chunk(2)
                            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    person_gen_latent = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                    decoded = self.vae.decode(person_gen_latent / self.vae.config.scaling_factor).sample
                    decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
                    decoded_arr = (decoded.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
                    m3 = mask_np[:, :, None]
                    repaint = (decoded_arr.astype(np.float32) * m3 + person_np.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
                    tryon_result_pil = Image.fromarray(repaint, mode="RGB")
                    logger.info("[CatVTON] Diffusion try-on synthesis successful")
                    tryon_result_pil = self._tint_worn_garment(
                        tryon_result_pil, uniform_prepped, mask_pil
                    )
                    tryon_result_pil = self._blend_template_chroma(
                        tryon_result_pil, uniform_prepped, mask_pil, person_prepped
                    )
            except Exception as e:
                logger.warning("[CatVTON] Diffusion inference fallback triggered (%s)", e)
                u_arr = np.array(uniform_prepped)
                m_arr = (np.array(mask_pil.convert("L")).astype(np.float32) / 255.0)[:, :, None]
                p_arr = np.array(person_prepped)
                fallback_arr = (u_arr.astype(np.float32) * m_arr + p_arr.astype(np.float32) * (1.0 - m_arr)).astype(np.uint8)
                tryon_result_pil = Image.fromarray(fallback_arr, mode="RGB")

        # ── 3. High-Fidelity Image Compositing ────────────────────────────────
        if apply_compositing:
            logger.info("[CatVTON] Running Image Compositor (Identity Lock + Studio Polish)...")
            final_out_path = self.compositor.composite(
                person_input=person_pil,
                tryon_input=tryon_result_pil,
                uniform_input=uniform_pil,
                bg_color=bg_color,
                job_id=job_id,
            )
            return final_out_path
        else:
            out_path = OUTPUTS_DIR / f"{job_id}_catvton_raw.png"
            tryon_result_pil.save(str(out_path), "PNG")
            return out_path
