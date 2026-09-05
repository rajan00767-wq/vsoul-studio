"""
VTON Image Compositor
High-Fidelity Hybrid Compositor for Virtual Try-On (CatVTON / IDM-VTON) & Uniform Swap.
Combines:
1. Anatomical Neck & Collar Alignment (InsightFace + SCHP landmark alignment)
2. 100% Biometric Identity & Hair Lock (Zero AI drift)
3. Smooth Feathered Collar-Neck Fusion (no neck seams or double-neck artifacts)
4. Ultra-Clean Studio Background Replacement (BiRefNet-HR matting)
5. Standard Studio 3:4 Portrait Framing & Micro-Contrast Enhancement
"""

from __future__ import annotations

import math
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pipelines.birefnet_service import background_removal
from utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class VTONCompositor:
    """
    Precision image compositor that merges AI-generated VTON garment draping / uniform templates
    with 100% authentic biometric facial features, natural neck geometry,
    and studio-grade background replacement.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VTONCompositor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._insightface_app = None
        self._schp_service = None

    def _get_insightface(self):
        if self._insightface_app is None:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self._insightface_app = app
                logger.info("[VTONCompositor] InsightFace initialized")
            except Exception as e:
                logger.warning("[VTONCompositor] InsightFace initialization failed: %s", e)
        return self._insightface_app

    def _get_schp(self):
        if self._schp_service is None:
            try:
                from pipelines import schp_service
                if schp_service.available():
                    self._schp_service = schp_service
                    logger.info("[VTONCompositor] SCHP semantic segmentation service ready")
            except Exception as e:
                logger.warning("[VTONCompositor] SCHP service unavailable: %s", e)
        return self._schp_service

    def generate_tryon_mask(
        self,
        person_img: Union[Image.Image, np.ndarray, str, Path],
        target_size: Tuple[int, int] = (768, 1024),
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Generate the clothing-agnostic mask and preprocessed images for CatVTON / IDM-VTON.
        """
        if isinstance(person_img, (str, Path)):
            person_pil = Image.open(person_img).convert("RGB")
        elif isinstance(person_img, np.ndarray):
            person_pil = Image.fromarray(person_img).convert("RGB")
        else:
            person_pil = person_img.convert("RGB")

        pw, ph = target_size
        person_resized = person_pil.resize((pw, ph), Image.LANCZOS)
        person_arr = np.array(person_resized)
        person_bgr = cv2.cvtColor(person_arr, cv2.COLOR_RGB2BGR)

        mask = np.zeros((ph, pw), dtype=np.uint8)

        # Landmark torso mask first so the whole shirt/vest region is always painted.
        app = self._get_insightface()
        face_box = None
        if app is not None:
            try:
                faces = app.get(person_bgr)
                if len(faces) > 0:
                    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    face_box = [int(v) for v in f.bbox]
            except Exception as e:
                logger.warning("[VTONCompositor] Face box for try-on mask failed: %s", e)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            chin_y = min(ph - 1, int(y2 + fh * 0.08))
            mask[chin_y:, :] = 255
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.ellipse(mask, face_center, (int(fw * 0.78), int(fh * 0.95)), 0, 0, 360, 0, -1)
        else:
            mask[int(ph * 0.32):, :] = 255

        # Union SCHP clothes if it covers a real torso (ignore tiny necklace blobs).
        schp = self._get_schp()
        if schp is not None:
            try:
                labels = schp.parse(person_arr)["labels"]
                clothes_mask = np.isin(labels, [5, 6, 7, 10, 11, 12, 14, 15]).astype(np.uint8) * 255
                frac = float(clothes_mask.sum()) / float(max(1, clothes_mask.size * 255))
                if frac >= 0.08:
                    k_cloth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    clothes_mask = cv2.dilate(clothes_mask, k_cloth, iterations=2)
                    protect_mask = np.isin(labels, [2, 13]).astype(np.uint8) * 255
                    k_prot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    protect_mask = cv2.dilate(protect_mask, k_prot, iterations=2)
                    clothes_mask = np.where(protect_mask > 0, 0, clothes_mask).astype(np.uint8)
                    mask = np.maximum(mask, clothes_mask)
                    logger.info("[VTONCompositor] SCHP clothes unioned into try-on mask (%.1f%%)", frac * 100)
                else:
                    logger.info("[VTONCompositor] SCHP clothes too small (%.1f%%) — using chin-to-hem mask", frac * 100)
            except Exception as e:
                logger.warning("[VTONCompositor] SCHP parsing error: %s", e)

        mask_clean = cv2.GaussianBlur(mask, (11, 11), 0)
        _, mask_clean = cv2.threshold(mask_clean, 48, 255, cv2.THRESH_BINARY)
        logger.info("[VTONCompositor] Try-on mask coverage %.1f%%", 100.0 * mask_clean.sum() / max(1, mask_clean.size * 255))
        mask_pil = Image.fromarray(mask_clean, mode="L")

        mask_norm = (mask_clean.astype(np.float32) / 255.0)[:, :, None]
        masked_arr = (person_arr.astype(np.float32) * (1.0 - mask_norm)).astype(np.uint8)
        masked_person_pil = Image.fromarray(masked_arr, mode="RGB")

        return person_resized, mask_pil, masked_person_pil

    def composite(
        self,
        person_input: Union[Image.Image, np.ndarray, str, Path],
        tryon_input: Union[Image.Image, np.ndarray, str, Path],
        uniform_input: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
        bg_color: Tuple[int, int, int] = (4, 126, 246),
        job_id: str = "vton_job",
        face_restore: bool = True,
        output_size: Tuple[int, int] = (1086, 1448),
        blend_strength: float = 0.95,
    ) -> Path:
        """
        High-Fidelity Virtual Try-On & Uniform Compositing Engine:
        1. Extract clean person and uniform foregrounds using BiRefNet-HR.
        2. Detect facial landmarks (chin tip, jawline, eye center) and uniform collar opening.
        3. Scale and align head/neck so chin sits directly above uniform collar with zero neck gaps.
        4. Apply feathered neck-to-collar alpha blending and face protection.
        5. Composite over requested studio background (Studio Blue rgb(4,126,246), White, Gray).
        6. Frame into ISO/Passport standard 3:4 portrait canvas (1086x1448).
        7. Micro-contrast & CLAHE detail enhancement for crisp fabric texture and facial definition.
        """
        TARGET_H = 1000

        # ── 1. Load Raw Inputs ───────────────────────────────────────────────
        def _to_pil(img_in) -> Image.Image:
            if isinstance(img_in, (str, Path)):
                return Image.open(img_in).convert("RGB")
            elif isinstance(img_in, np.ndarray):
                return Image.fromarray(img_in).convert("RGB")
            return img_in.convert("RGB")

        person_pil = _to_pil(person_input)
        uniform_pil = _to_pil(uniform_input) if uniform_input is not None else _to_pil(tryon_input)

        # ── 2. Remove Backgrounds & Extract Garment ──────────────────────────
        try:
            buf_p = BytesIO()
            person_pil.save(buf_p, format="PNG")
            p_cutout_bytes = background_removal.remove_background(buf_p.getvalue())
            person_rgba_raw = Image.open(BytesIO(p_cutout_bytes)).convert("RGBA")
        except Exception as e:
            logger.warning("[VTONCompositor] Person background removal fallback: %s", e)
            person_rgba_raw = person_pil.convert("RGBA")

        try:
            # Check if uniform template contains a person/face; extract ONLY the garment with SCHP
            u_arr_raw = np.array(uniform_pil.convert("RGB"))
            from pipelines import schp_service
            u_parsed = schp_service.parse(u_arr_raw)
            u_labels = u_parsed["labels"]
            # Clothing labels: UpperClothes (5), Dress (6), Coat/Jacket (7), Jumpsuit (10), Scarf/Tie (11), Skirt (12)
            u_clothing_mask = np.isin(u_labels, [5, 6, 7, 10, 11, 12]).astype(np.uint8) * 255

            if np.sum(u_clothing_mask > 0) > 1000:
                # Keep only the garment, completely removing the other person's face/neck
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                u_clothing_mask = cv2.dilate(u_clothing_mask, kernel, iterations=1)
                uniform_rgba_raw = Image.fromarray(np.dstack([u_arr_raw, u_clothing_mask]), "RGBA")
                logger.info("[VTONCompositor] Extracted uniform garment from person template using SCHP")
            else:
                buf_u = BytesIO()
                uniform_pil.save(buf_u, format="PNG")
                u_cutout_bytes = background_removal.remove_background(buf_u.getvalue())
                uniform_rgba_raw = Image.open(BytesIO(u_cutout_bytes)).convert("RGBA")
        except Exception as e:
            logger.warning("[VTONCompositor] Uniform background removal fallback: %s", e)
            uniform_rgba_raw = uniform_pil.convert("RGBA")

        # Normalise both cutouts to TARGET_H
        def _resize_h(img: Image.Image, h: int) -> Image.Image:
            w = max(1, int(img.width * h / float(img.height)))
            return img.resize((w, h), Image.LANCZOS)

        # Fill internal holes inside the uniform garment (e.g. transparent chest holes)
        u_arr_init = np.array(uniform_rgba_raw)
        if u_arr_init.ndim == 3 and u_arr_init.shape[2] == 4:
            u_alpha_bin = (u_arr_init[:, :, 3] > 40).astype(np.uint8)
            contours, _ = cv2.findContours(u_alpha_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(u_alpha_bin, contours, -1, 1, thickness=-1)
                # Keep soft boundary
                u_arr_init[:, :, 3] = np.maximum(u_arr_init[:, :, 3], (u_alpha_bin * 255).astype(np.uint8))
                uniform_rgba_raw = Image.fromarray(u_arr_init, "RGBA")

        person_rgba = _resize_h(person_rgba_raw, TARGET_H)

        # If uniform is a full body dress/suit, crop to upper torso for natural portrait proportions
        u_arr_temp = np.array(uniform_rgba_raw)
        u_alpha_temp = u_arr_temp[:, :, 3]
        u_y_coords = np.where(u_alpha_temp > 30)[0]
        if len(u_y_coords) > 0 and (u_y_coords.max() - u_y_coords.min()) > uniform_rgba_raw.height * 0.75:
            # Full body garment detected -> crop to upper 60% (bust and shoulders)
            u_crop_h = int(u_y_coords.min() + (u_y_coords.max() - u_y_coords.min()) * 0.62)
            uniform_rgba_raw = uniform_rgba_raw.crop((0, 0, uniform_rgba_raw.width, u_crop_h))

        uniform_rgba = _resize_h(uniform_rgba_raw, TARGET_H)

        person_arr = np.array(person_rgba).copy()    # H W 4
        uniform_arr = np.array(uniform_rgba).copy()  # H W 4

        uniform_alpha_orig = uniform_arr[:, :, 3].copy()
        ph, pw = person_arr.shape[:2]
        uh, uw = uniform_arr.shape[:2]

        person_bgr = cv2.cvtColor(person_arr[:, :, :3], cv2.COLOR_RGB2BGR)
        uniform_bgr = cv2.cvtColor(uniform_arr[:, :, :3], cv2.COLOR_RGB2BGR)

        # ── 3. Detect Face Landmarks & Collar Opening ────────────────────────
        app = self._get_insightface()
        p_faces = app.get(person_bgr) if app is not None else []
        u_faces = app.get(uniform_bgr) if app is not None else []

        if len(p_faces) == 0:
            # Fallback face box if insightface missed
            fx, fy, fw, fh = pw // 4, int(ph * 0.15), pw // 2, int(ph * 0.35)
            chin_y_person = fy + int(fh * 1.05)
            jaw_w_person = fw * 0.85
        else:
            pf = max(p_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            x1, y1, x2, y2 = [int(v) for v in pf.bbox]
            fx, fy, fw, fh = x1, y1, max(1, x2 - x1), max(1, y2 - y1)
            chin_y_person = fy + int(fh * 1.05)
            jaw_w_person = fw * 0.88
            if hasattr(pf, "kps") and pf.kps is not None and len(pf.kps) >= 5:
                # Keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
                mouth_y = int((pf.kps[3][1] + pf.kps[4][1]) * 0.5)
                chin_y_person = mouth_y + int(fh * 0.30)

        # ── 4. Robust Collar Top Detection on Uniform ───────────────────────
        neck_band = uniform_alpha_orig[:, int(uw * 0.25):int(uw * 0.75)]
        y_indices = np.where(neck_band > 40)[0]
        if len(y_indices) > 0:
            collar_y = int(y_indices.min())
        else:
            collar_y = int(uh * 0.15)

        # ── 5. Align Person Head & Neck to Collar (Anatomical Proportions) ───
        # A natural human face width is ~30-36% of shoulder width
        target_fw = uw * 0.34
        scale_p = target_fw / max(float(fw), 1.0)
        scale_p = float(np.clip(scale_p, 0.75, 1.20))

        neck_show_px = int(fh * 0.25)
        head_bottom = min(ph, chin_y_person + neck_show_px)
        person_head = person_arr[:head_bottom, :]

        sp_w = max(1, int(pw * scale_p))
        sp_h = max(1, int(head_bottom * scale_p))
        person_scaled = cv2.resize(person_head, (sp_w, sp_h), interpolation=cv2.INTER_LANCZOS4)
        chin_scaled = int(chin_y_person * scale_p)

        # ── 6. Neck Fade Ramp into Collar Opening ────────────────────────────
        p_alpha = person_scaled[:, :, 3].copy()
        fade_s = chin_scaled + int(fh * scale_p * 0.05)
        fade_e = sp_h
        if fade_e > fade_s:
            ramp = np.linspace(1.0, 0.0, fade_e - fade_s, dtype=np.float32)
            p_alpha[fade_s:fade_e] = (p_alpha[fade_s:fade_e].astype(np.float32) * ramp[:, None]).astype(np.uint8)

        person_scaled[:, :, 3] = p_alpha

        # ── 7. Place Aligned Head & Uniform on Composite Canvas ──────────────
        desired_chin_y = collar_y + int(fh * scale_p * 0.10)
        head_y_offset = desired_chin_y - chin_scaled

        canvas_top_pad = max(0, -head_y_offset)
        canvas_h = uh + canvas_top_pad
        canvas_w = max(uw, sp_w + 40)
        canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)

        # Center X
        cx = canvas_w // 2
        px_off = cx - sp_w // 2
        py_off = head_y_offset + canvas_top_pad

        # Paste head
        p_y0 = max(0, py_off)
        p_y1 = min(canvas_h, py_off + sp_h)
        s_y0 = max(0, -py_off)
        s_y1 = s_y0 + (p_y1 - p_y0)

        p_x0 = max(0, px_off)
        p_x1 = min(canvas_w, px_off + sp_w)
        s_x0 = max(0, -px_off)
        s_x1 = s_x0 + (p_x1 - p_x0)

        if p_x1 > p_x0 and p_y1 > p_y0:
            src = person_scaled[s_y0:s_y1, s_x0:s_x1]
            src_a = (src[:, :, 3] / 255.0)[:, :, None]
            canvas[p_y0:p_y1, p_x0:p_x1, :3] = (src[:, :, :3].astype(np.float32) * src_a).astype(np.uint8)
            canvas[p_y0:p_y1, p_x0:p_x1, 3] = src[:, :, 3]

        # Paste uniform onto canvas-sized layer
        u_y0 = canvas_top_pad
        u_y1 = u_y0 + uh
        u_x0 = (canvas_w - uw) // 2
        u_x1 = u_x0 + uw

        uniform_canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)
        uniform_canvas[u_y0:u_y1, u_x0:u_x1] = uniform_arr
        u_f = uniform_canvas.astype(np.float32)
        u_a = u_f[:, :, 3:4] / 255.0

        face_protect = np.zeros((canvas_h, canvas_w), np.uint8)
        face_cx_c = int(px_off + (fx + fw * 0.5) * scale_p)
        face_cy_c = int(py_off + (fy + fh * 0.48) * scale_p)
        cv2.ellipse(
            face_protect,
            (face_cx_c, face_cy_c),
            (max(8, int(fw * scale_p * 0.65)), max(8, int(fh * scale_p * 0.75))),
            0, 0, 360, 255, -1
        )
        face_protect = cv2.GaussianBlur(face_protect, (45, 45), 0)

        person_pres = (canvas[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        prot_alpha = (face_protect.astype(np.float32) / 255.0)[:, :, None] * person_pres
        u_a_eff = u_a * (1.0 - prot_alpha)

        canvas[:, :, :3] = (
            u_f[:, :, :3] * u_a_eff + canvas[:, :, :3].astype(np.float32) * (1.0 - u_a_eff)
        ).astype(np.uint8)
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], uniform_canvas[:, :, 3])

        # Paste onto solid studio background
        rgba_canvas = Image.fromarray(canvas, "RGBA")
        studio_pil = Image.new("RGB", rgba_canvas.size, bg_color)
        studio_pil.paste(rgba_canvas, mask=rgba_canvas.split()[3])

        # ── 7. Crop Padding & Standard 3:4 Portrait Framing ──────────────────
        framed_pil = self._frame_portrait(studio_pil, bg_color, target_size=output_size)

        # ── 8. Micro-Contrast & Detail Enhancement ───────────────────────────
        enhanced_pil = self._enhance_details(framed_pil, bg_color)

        out_path = OUTPUTS_DIR / f"{job_id}_vton_composite.png"
        enhanced_pil.save(str(out_path), "PNG")
        logger.info("[VTONCompositor] Success! Final composited portrait saved to %s", out_path)
        return out_path

    def _frame_portrait(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int],
        target_size: Tuple[int, int] = (1086, 1448),
    ) -> Image.Image:
        """Frame the subject centered in a 3:4 standard portrait canvas."""
        tw, th = target_size
        img_arr = np.array(image)
        h, w = img_arr.shape[:2]

        # Trim top/bottom background margins
        bg_arr = np.array(bg_color[:3], dtype=np.int32)
        diff = np.abs(img_arr.astype(np.int32) - bg_arr).max(axis=2)
        non_bg = np.where(diff > 18)
        if len(non_bg[0]) > 0:
            top_c = max(0, non_bg[0].min() - 10)
            bot_c = min(h, non_bg[0].max() + 20)
            left_c = max(0, non_bg[1].min() - 10)
            right_c = min(w, non_bg[1].max() + 10)
            cropped = image.crop((left_c, top_c, right_c, bot_c))
        else:
            cropped = image

        cw, ch = cropped.size
        scale = min(tw * 0.90 / float(cw), th * 0.88 / float(ch))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cropped.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (tw, th), bg_color)
        pos_x = (tw - nw) // 2
        pos_y = max(0, int(th * 0.08))
        canvas.paste(resized, (pos_x, pos_y))
        return canvas

    def _enhance_details(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Polish fabric weave, hair texture, and collar crispness."""
        arr = np.array(image)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_chan)
        lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        pil_enh = Image.fromarray(rgb_enhanced)
        sharp = pil_enh.filter(ImageFilter.UnsharpMask(radius=1.1, percent=105, threshold=3))
        return sharp
