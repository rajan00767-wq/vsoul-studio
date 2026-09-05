"""
Pipeline Orchestrator
Central coordinator for all AI enhancement pipelines.
Detects features, selects appropriate models, and runs the full enhancement chain.
"""

import asyncio
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from utils.logger import get_logger
from pipelines.birefnet_service import background_removal

logger = get_logger(__name__)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


class PipelineOrchestrator:
    """
    Orchestrates the full AI pipeline:
    1. Image Analysis (OpenCV + YOLOv8 + face detection)
    2. Face Restoration (GFPGAN / CodeFormer)
    3. Background Processing (SAM + Stable Diffusion)
    4. Enhancement (denoise, sharpen, relight)
    5. Upscaling (Real-ESRGAN)
    6. Export
    """

    def __init__(self):
        # Patch missing torchvision module before any basicsr import
        self._install_torchvision_compat()
        self._models_loaded: Dict[str, bool] = {
            "realesrgan": False,
            "gfpgan": False,
            "sdxl": False,
            "controlnet": False,
            "ic_light": False,
            "sam": False,
            "opencv": True,  # always available
        }
        self._lazy_models: Dict[str, Any] = {}
        # GFPGAN stores FaceRestoreHelper as an instance var — not thread-safe.
        # Serialize all calls to enhance() so concurrent jobs don't overwrite
        # each other's face landmarks (the root cause of image mismatch).
        self._gfpgan_lock = threading.Lock()
        # Serialize model lazy-loading so two threads don't both try to init.
        self._model_init_lock = threading.Lock()
        # Dedicated 2-worker pool for face restoration.
        # Concurrent jobs queue here instead of blocking the default executor,
        # preventing the "frozen at 45%" stall when two uploads arrive together.
        self._face_restore_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="face_restore"
        )

    # ─── Device / provider helpers ────────────────────────────────────────────

    @staticmethod
    def _torch_device():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except ImportError:
            pass
        return torch.device("cpu")

    @staticmethod
    def _onnx_providers():
        """Return ONNX Runtime execution providers ordered by speed."""
        try:
            import torch
            if torch.cuda.is_available():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            pass
        return ["CPUExecutionProvider"]

    # ─── Cached model loaders ─────────────────────────────────────────────────

    def _get_rembg_session(self):
        # Deprecated: rembg replaced by BiRefNet-lite. Kept to avoid AttributeError
        # on any call sites that haven't been updated yet.
        background_removal.warm_up()

    def _get_codeformer(self):
        """Returns (net, device) with net cached across jobs."""
        if "codeformer" not in self._lazy_models:
            with self._model_init_lock:
                if "codeformer" not in self._lazy_models:
                    self._install_torchvision_compat()
                    import torch
                    from basicsr.archs.codeformer_arch import CodeFormer

                    model_path = Path("models/codeformer/codeformer.pth")
                    if not model_path.exists():
                        raise FileNotFoundError("CodeFormer weights not found")

                    device = self._torch_device()
                    net = CodeFormer(
                        dim_embd=512, codebook_size=1024, n_head=8,
                        n_layers=9, connect_list=["32", "64", "128", "256"],
                    ).to(device)
                    checkpoint = torch.load(str(model_path), map_location=device)
                    net.load_state_dict(checkpoint["params_ema"])
                    net.eval()
                    self._lazy_models["codeformer"] = (net, device)
                    self._models_loaded["gfpgan"] = True
                    logger.info("CodeFormer loaded on %s", device)
        return self._lazy_models["codeformer"]

    def _get_controlnet_pipeline(self):
        """Lazily construct the canny-ControlNet SDXL wrapper (shared for
        background regeneration and uniform-swap seam polish)."""
        if "controlnet" not in self._lazy_models:
            with self._model_init_lock:
                if "controlnet" not in self._lazy_models:
                    from pipelines.controlnet_pipeline import ControlNetPipeline
                    self._lazy_models["controlnet"] = ControlNetPipeline()
                    self._models_loaded["controlnet"] = True
        return self._lazy_models["controlnet"]

    def _get_ic_light_pipeline(self):
        """Lazily construct the IC-Light generative-relight wrapper."""
        if "ic_light" not in self._lazy_models:
            with self._model_init_lock:
                if "ic_light" not in self._lazy_models:
                    from pipelines.ic_light_pipeline import ICLightPipeline
                    self._lazy_models["ic_light"] = ICLightPipeline()
                    self._models_loaded["ic_light"] = True
        return self._lazy_models["ic_light"]

    def _get_face_analysis(self):
        """Returns a cached InsightFace FaceAnalysis app (buffalo_l), used
        for the identity-similarity quality gate. Deliberately forced onto
        CPUExecutionProvider (not self._onnx_providers()'s CoreML/MPS
        preference) — on Apple Silicon, onnxruntime's CoreMLExecutionProvider
        does a first-run ANE compile of this model graph that took 10+
        minutes in testing here, long enough to blow through ai_worker.py's
        300s job timeout. buffalo_l is small enough that plain CPU inference
        is fast (well under a second per face) and has no such stall risk."""
        if "face_analysis" not in self._lazy_models:
            with self._model_init_lock:
                if "face_analysis" not in self._lazy_models:
                    from insightface.app import FaceAnalysis

                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(320, 320))
                    self._lazy_models["face_analysis"] = app
                    logger.info("InsightFace FaceAnalysis (buffalo_l, CPU) loaded")
        return self._lazy_models["face_analysis"]

    def _detect_all_faces(self, bgr_img):
        """Returns list of dicts: {'bbox': (x,y,w,h), 'kps': landmarks, 'embedding': emb, 'det_score': score}"""
        try:
            app = self._get_face_analysis()
            faces = app.get(bgr_img)
            results = []
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                results.append({
                    "bbox": (x1, y1, w, h),
                    "kps": f.kps,
                    "embedding": f.embedding,
                    "det_score": float(f.det_score) if hasattr(f, "det_score") else 1.0,
                })
            return sorted(results, key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True)
        except Exception as e:
            logger.debug("InsightFace detection fallback: %s", e)
            return []

    def _detect_primary_face(self, bgr_img):
        faces = self._detect_all_faces(bgr_img)
        return faces[0] if faces else None

    def _identity_similarity(self, original_path: str, final_path: str) -> Optional[dict]:
        """Cosine similarity between ArcFace embeddings of the largest face
        in the original vs. final image. Returns None (non-fatal) if
        InsightFace is unavailable or no face is found in either image."""
        try:
            import cv2
            import numpy as np

            orig = cv2.imread(original_path)
            final = cv2.imread(final_path)
            if orig is None or final is None:
                return None

            app = self._get_face_analysis()
            orig_faces = app.get(orig)
            final_faces = app.get(final)
            if not orig_faces or not final_faces:
                return None

            def largest(faces):
                return max(
                    faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )

            o_emb = largest(orig_faces).embedding
            f_emb = largest(final_faces).embedding
            sim = float(np.dot(o_emb, f_emb) / (np.linalg.norm(o_emb) * np.linalg.norm(f_emb)))
            return {"identity_similarity": round(sim, 3), "passed": sim >= 0.4}
        except Exception as e:
            logger.info("Identity similarity gate skipped: %s", e)
            return None

    def _get_gfpgan(self):
        """Returns GFPGANer cached across jobs."""
        if "gfpgan" not in self._lazy_models:
            with self._model_init_lock:
                if "gfpgan" not in self._lazy_models:
                    self._install_torchvision_compat()
                    from gfpgan import GFPGANer

                    model_path = Path("models/gfpgan/GFPGANv1.4.pth")
                    if not model_path.exists():
                        raise FileNotFoundError("GFPGAN model not found")

                    restorer = GFPGANer(
                        model_path=str(model_path),
                        upscale=1,
                        arch="clean",
                        channel_multiplier=2,
                    )
                    self._lazy_models["gfpgan"] = restorer
                    self._models_loaded["gfpgan"] = True
                    logger.info("GFPGAN loaded")
        return self._lazy_models["gfpgan"]

    def _get_spandrel(self, model_file: Path):
        """Returns Spandrel model cached by file path."""
        key = f"spandrel_{model_file.name}"
        if key in self._lazy_models:
            return self._lazy_models[key]

        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        model = ModelLoader().load_from_file(str(model_file))
        if not isinstance(model, ImageModelDescriptor):
            raise TypeError("Loaded model is not an image SR model")

        device = self._torch_device()
        model = model.model.eval().to(device)
        self._lazy_models[key] = (model, device)
        logger.info("Spandrel %s loaded on %s", model_file.name, device)
        return self._lazy_models[key]

    def _get_realesrgan(self):
        """Returns RealESRGANer cached across jobs."""
        if "realesrgan" in self._lazy_models:
            return self._lazy_models["realesrgan"]

        self._install_torchvision_compat()
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch

        model_file = Path("models/realesrgan/RealESRGAN_x4plus.pth")
        if not model_file.exists():
            raise FileNotFoundError("RealESRGAN model not found")

        device = self._torch_device()
        half = device.type == "cuda"
        rrdb = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_file),
            model=rrdb,
            tile=512,
            tile_pad=16,
            pre_pad=0,
            half=half,
            device=device,
        )
        self._lazy_models["realesrgan"] = upsampler
        self._models_loaded["realesrgan"] = True
        logger.info("Real-ESRGAN loaded on %s (half=%s)", device, half)
        return upsampler

    # ─── Startup pre-warm ────────────────────────────────────────────────────

    def prewarm(self) -> None:
        """Load all available models into memory. Called once at server startup
        in a background thread so the first real job is fast."""
        # Must run before any basicsr import — newer torchvision removed this module
        self._install_torchvision_compat()
        logger.info("Pre-warming models...")
        try:
            background_removal.warm_up()
        except Exception as e:
            logger.info("BiRefNet-lite pre-warm skipped: %s", e)

        try:
            self._get_codeformer()
        except Exception as e:
            logger.info("CodeFormer pre-warm skipped: %s", e)

        if "codeformer" not in self._lazy_models:
            try:
                self._get_gfpgan()
            except Exception as e:
                logger.info("GFPGAN pre-warm skipped: %s", e)

        spandrel_candidates = [
            Path("models/hat/HAT-L_SRx4_ImageNet-pretrain.pth"),
            Path("models/hat/HAT_SRx4.pth"),
            Path("models/swinir/SwinIR_x4.pth"),
        ]
        spandrel_file = next((p for p in spandrel_candidates if p.exists()), None)
        if spandrel_file:
            try:
                self._get_spandrel(spandrel_file)
            except Exception as e:
                logger.info("Spandrel pre-warm skipped: %s", e)
        else:
            try:
                self._get_realesrgan()
            except Exception as e:
                logger.info("Real-ESRGAN pre-warm skipped: %s", e)

        try:
            self._get_face_analysis()
        except Exception as e:
            logger.info("InsightFace FaceAnalysis pre-warm skipped: %s", e)

        logger.info("Pre-warm complete")

    # ─── Model Status ─────────────────────────────────────────────────────────

    def get_model_status(self) -> dict:
        status = {}
        for name, loaded in self._models_loaded.items():
            model_dir = Path("models") / name
            status[name] = {
                "loaded": loaded,
                "available": self._check_model_files(name),
                "path": str(model_dir),
            }
        return status

    def _check_model_files(self, name: str) -> bool:
        model_dir = Path("models") / name
        if not model_dir.exists():
            return False
        files = list(model_dir.iterdir())
        return len(files) > 0

    # ─── Image Analysis ───────────────────────────────────────────────────────

    async def analyze_image(self, image_path: str) -> dict:
        """Analyze image properties without running full pipeline"""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._analyze_sync, image_path
        )

    def _analyze_sync(self, image_path: str) -> dict:
        try:
            import cv2
            import numpy as np

            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not read image"}

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = blur_score < 100

            # Brightness
            brightness = gray.mean()

            # Face detection using InsightFace (RetinaFace deep neural network)
            detected_faces = self._detect_all_faces(img)
            face_count = len(detected_faces)

            tilt_info = {}
            if face_count > 0:
                primary = detected_faces[0]
                tilt_info = self._estimate_tilt_insightface(primary)

            return {
                "width": w,
                "height": h,
                "megapixels": round((w * h) / 1_000_000, 2),
                "faces_detected": face_count,
                "is_blurry": bool(is_blurry),
                "blur_score": round(float(blur_score), 2),
                "brightness": round(float(brightness), 2),
                "is_dark": bool(brightness < 80),
                "is_overexposed": bool(brightness > 200),
                "aspect_ratio": round(w / h, 3),
                "recommendations": self._get_recommendations(
                    face_count, is_blurry, brightness, w, h
                ),
                **tilt_info,
            }
        except Exception as e:
            logger.exception("Analysis failed: %s", e)
            return {"error": str(e)}

    def _estimate_tilt_insightface(self, face_dict: dict) -> dict:
        """Estimate exact roll angle from InsightFace 5-point keypoints."""
        try:
            import math
            kps = face_dict.get("kps")
            if kps is None or len(kps) < 2:
                return {"face_landmarks_detected": False}

            # kps[0] = left eye, kps[1] = right eye
            l_eye = kps[0]
            r_eye = kps[1]
            dx = r_eye[0] - l_eye[0]
            dy = r_eye[1] - l_eye[1]
            if dx == 0:
                return {"face_landmarks_detected": False}

            angle = math.degrees(math.atan2(dy, dx))
            category = (
                "aligned" if abs(angle) < 5 else
                "small_tilt" if abs(angle) < 15 else
                "large_tilt"
            )
            return {
                "face_landmarks_detected": True,
                "tilt_angle_deg": round(angle, 1),
                "tilt_category": category,
            }
        except Exception as e:
            logger.info("InsightFace tilt estimation skipped: %s", e)
            return {}

    def _get_recommendations(
        self, faces: int, blurry: bool, brightness: float, w: int, h: int
    ) -> list:
        recs = []
        if faces > 0:
            recs.append("face_restore")
        if blurry:
            recs.append("sharpen")
        if brightness < 80:
            recs.append("brighten")
        if w < 1024 or h < 1024:
            recs.append("upscale_2x")
        else:
            recs.append("upscale_2x")
        recs.append("denoise")
        return recs

    # ─── Full Pipeline ────────────────────────────────────────────────────────

    async def run(
        self,
        job_id: str,
        input_path: str,
        config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """
        Run the full AI enhancement pipeline.
        Falls back gracefully if GPU models aren't installed yet.
        """
        def cb(p: int, msg: str, s: str = "processing", extra: Optional[dict] = None):
            if progress_callback:
                progress_callback(p, msg, s, extra=extra)

        loop = asyncio.get_running_loop()

        # Step 1: Analyze
        cb(10, "Analyzing image...", "analysis")
        analysis = await self.analyze_image(input_path)
        logger.info("📊 Analysis: %s", analysis)
        cb(15, "Analysis complete", "analysis", extra={"analysis": analysis})

        # Step 2: Basic OpenCV enhancements (always available)
        cb(25, "Applying core enhancements...", "enhance")
        enhanced_path = await loop.run_in_executor(
            None, self._opencv_enhance, input_path, job_id, config, analysis
        )

        # Step 2b: Geometric tilt correction (only if explicitly enabled in config)
        if config.get("auto_tilt_correct", False) and analysis.get("tilt_category") == "small_tilt":
            cb(30, "Leveling tilted face...", "tilt_correct")
            enhanced_path = await loop.run_in_executor(
                None, self._correct_tilt, str(enhanced_path), job_id, analysis["tilt_angle_deg"]
            )

        # Step 3: Face Restoration (skip if disabled, no faces, or — when
        # face_restore_if_needed is set — the face is already sharp)
        _face_restore_gate = (
            config.get("face_restore") and analysis.get("faces_detected", 0) > 0
        )
        if _face_restore_gate and config.get("face_restore_if_needed"):
            _face_restore_gate = bool(analysis.get("is_blurry", True))
        if _face_restore_gate:
            cb(46, "Restoring faces...", "face_restore")
            _t_restore = time.time()
            _tick_msgs = [
                "Restoring faces...",
                "Improving skin clarity...",
                "Sharpening facial features...",
                "Finalising face quality...",
            ]
            try:
                # Run in dedicated executor so concurrent jobs queue, not race.
                # asyncio.shield keeps the future alive if we break from the loop early.
                _restore_future = loop.run_in_executor(
                    self._face_restore_executor,
                    self._face_restore, str(enhanced_path), job_id,
                )
                _tick = 0
                _pct  = 50
                while not _restore_future.done():
                    try:
                        enhanced_path = await asyncio.wait_for(
                            asyncio.shield(_restore_future), timeout=2.0
                        )
                        break
                    except asyncio.TimeoutError:
                        # Send a progress tick so the frontend bar keeps moving.
                        _msg = _tick_msgs[min(_tick, len(_tick_msgs) - 1)]
                        cb(min(_pct, 68), _msg, "face_restore")
                        _pct  += 6
                        _tick += 1
                        if time.time() - _t_restore > 45.0:
                            logger.warning(
                                "[FACE_RESTORE_TIMEOUT] job=%s exceeded 45s — "
                                "using bilateral fallback", job_id
                            )
                            enhanced_path = Path(str(enhanced_path))
                            break
                logger.info(
                    "[FACE_RESTORE_COMPLETE] job=%s elapsed=%.1fs",
                    job_id, time.time() - _t_restore,
                )
            except Exception as _exc:
                logger.warning(
                    "[FACE_RESTORE_ERROR] job=%s error=%s — bilateral fallback",
                    job_id, _exc,
                )
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            cb(45, "Skipping face restoration (disabled, no faces, or already sharp)", "face_restore")

        # Step 4: Background Replace (only if explicitly enabled)
        if config.get("background_replace"):
            cb(60, "Processing background...", "background")
            enhanced_path = await loop.run_in_executor(
                None, self._background_replace, str(enhanced_path), job_id, config
            )
            # Clean up GPU memory
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

        # Step 5: Upscaling (skip if factor is 1, or — when upscale_if_needed
        # is set — the image is already above the low-res threshold)
        upscale = config.get("upscale_factor", 2)
        _upscale_gate = upscale > 1
        if _upscale_gate and config.get("upscale_if_needed"):
            _upscale_gate = analysis.get("megapixels", 0) < 1.0
        if _upscale_gate:
            cb(75, f"Upscaling {upscale}x...", "upscale")
            _pre_upscale = str(enhanced_path)
            try:
                enhanced_path = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._upscale, _pre_upscale, job_id, upscale
                    ),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[UPSCALE_TIMEOUT] job=%s exceeded 60s — Lanczos fallback", job_id
                )
                enhanced_path = await loop.run_in_executor(
                    None, self._upscale_fallback, _pre_upscale, job_id, upscale
                )
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            cb(75, "Skipping upscaling (factor = 1, or already high-res)", "upscale")

        # Step 6: Final polish
        cb(90, "Final cinematic polish...", "polish")
        final_path = await loop.run_in_executor(
            None, self._final_polish, str(enhanced_path), job_id, config
        )

        # Every stage degrades by passing its input path through, so a
        # mid-pipeline loss (e.g. an intermediate file deleted externally)
        # can silently propagate a dead path all the way here. Fail loudly
        # rather than mark the job done with an output that doesn't exist.
        if not Path(final_path).is_file():
            raise RuntimeError(
                f"pipeline produced no output file (last path: {final_path})"
            )

        # Step 7: Identity-similarity quality gate (non-fatal)
        quality_gate = None
        if not config.get("skip_quality_gate", False):
            try:
                quality_gate = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self._identity_similarity, input_path, str(final_path)
                    ),
                    timeout=3.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug("[QUALITY_GATE] skipped: %s", e)
                quality_gate = None

        cb(100, "Complete!", "done", extra={"quality_gate": quality_gate} if quality_gate else None)
        return Path(final_path)

    # ─── OpenCV Core Enhancement ──────────────────────────────────────────────

    def _opencv_enhance(
        self, input_path: str, job_id: str, config: dict, analysis: dict
    ) -> Path:
        """Core OpenCV-based enhancement pipeline"""
        import cv2
        import numpy as np

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        # Denoise lightly; passport photos should keep natural skin texture.
        if config.get("denoise", True):
            img = cv2.bilateralFilter(img, d=5, sigmaColor=10, sigmaSpace=10)

        # Auto exposure correction, kept intentionally subtle.
        if analysis.get("is_dark", False):
            img = self._auto_brightness(img, 18)
        elif analysis.get("is_overexposed", False):
            img = self._auto_brightness(img, -12)

        # Relight (config: relight): flatten the low-frequency lighting
        # gradient (retinex-style — evens out side-light/shadow across the
        # face), then suppress specular sun-glare hotspots. The glare pass
        # covers the whole frame (skin, hair AND clothing) when the
        # background is being replaced anyway — the wide-wash detector can
        # dim background pixels near the subject, which would show as a
        # halo if the original background were kept.
        # Runs before style grading so the LUT works on evened-out skin.
        if config.get("relight"):
            if config.get("relight_engine") == "ic_light":
                img = self._ic_light_relight(img, job_id, config)
            else:
                img = self._even_illumination(img)
                img = self._reduce_glare(
                    img, skin_only=not config.get("background_replace")
                )

        # Style-specific color grading
        style = config.get("style_mode", "luxury")
        img = self._apply_style_lut(img, style)

        # Save intermediate
        out_path = OUTPUTS_DIR / f"{job_id}_enhanced.png"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return out_path

    def _ic_light_relight(self, img_bgr, job_id: str, config: dict):
        """Generative relight via IC-Light (config: relight_engine=ic_light,
        relight_prompt, relight_strength). Falls back to the classical
        retinex + glare-suppression pass if the model can't load (missing
        rembg, OOM, first-download failure, etc.) so a job never hard-fails
        just because the generative engine wasn't picked up cleanly."""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from pipelines.ic_light_pipeline import DEFAULT_PROMPT, DEFAULT_NEG

            ic = self._get_ic_light_pipeline()
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = ic.relight(
                Image.fromarray(rgb),
                prompt=config.get("relight_prompt") or DEFAULT_PROMPT,
                negative_prompt=DEFAULT_NEG,
                strength=float(config.get("relight_strength", 0.6)),
                job_id=job_id,
            )
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(
                "[IC_LIGHT] job=%s failed (%s) — falling back to classical relight",
                job_id, e,
            )
            img = self._even_illumination(img_bgr)
            return self._reduce_glare(
                img, skin_only=not config.get("background_replace")
            )

    def _even_illumination(self, img):
        """Retinex-style lighting normalization on skin: estimate the
        low-frequency illumination field (large-sigma blur of luminance) and
        pull skin regions toward their mean lighting level. Evens out
        side-light and shadow gradients across face/neck/chest without
        touching detail — the illumination estimate is too blurry to contain
        any. Deterministic, identity-safe, no model."""
        import cv2
        import numpy as np

        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)

        # Very low frequency = the lighting field (face-scale gradients)
        illum = cv2.GaussianBlur(L, (0, 0), min(h, w) / 6.0)

        # Same skin gate as _reduce_glare, dilated wide to cover face+neck+chest
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 133)).astype(np.uint8)
        if int(skin.sum()) < 500:
            return img
        skin_wide = cv2.dilate(skin, np.ones((25, 25), np.uint8))
        mask = cv2.GaussianBlur(
            skin_wide.astype(np.float32), (0, 0), max(5.0, min(h, w) / 40.0)
        )
        mask = np.clip(mask, 0.0, 1.0)

        # Flatten 60% of the deviation from the skin's mean lighting level —
        # full flattening looks waxy; partial keeps natural dimensionality.
        target = float((illum * skin).sum() / max(1, int(skin.sum())))
        L = L - (illum - target) * 0.6 * mask

        out = cv2.merge((np.clip(L, 0, 255), A, B)).astype(np.uint8)
        logger.info(
            "[EVEN_ILLUM] flattened lighting over %.1f%% of frame (target L=%.0f)",
            mask.mean() * 100, target,
        )
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    def _reduce_glare(self, img, skin_only: bool = True):
        """Suppress specular sun glare: pull patches that are much brighter
        than their local surroundings back toward the local illumination
        baseline, and re-tint the chroma the glare washed out.
        Limits: pixels blown to pure white carry no detail — they get toned
        down and re-tinted (reads as matte lighting), not reconstructed.
        skin_only=True restricts to skin-tone regions; False also treats
        clothing/body — safe because the detector is relative (brighter than
        local surroundings), so uniformly bright fabric is unaffected."""
        import cv2
        import numpy as np

        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)

        # Local illumination baseline — blur radius at patch scale so glare
        # spots stand out but broad lighting differences don't.
        sigma = max(8.0, min(h, w) / 12.0)
        # Two context scales: the small one catches specular spots, the large
        # one catches wide sun washes (e.g. the whole top of the head) that
        # are big enough to be their own "surroundings" at the small scale.
        base_small = cv2.GaussianBlur(L, (0, 0), sigma)
        base_large = cv2.GaussianBlur(L, (0, 0), min(h, w) / 4.0)
        excess = np.maximum(L - base_small, L - base_large)

        # Specular candidates: notably brighter than surroundings AND bright.
        glare = ((excess > 10.0) & (L > 165.0)).astype(np.float32)

        if skin_only:
            # Skin gate (YCrCb ranges); dilated to catch glare cores that
            # clipped to white (pure white falls outside the skin chroma range).
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            skin = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 133)).astype(np.uint8)
            skin = cv2.dilate(skin, np.ones((15, 15), np.uint8)).astype(np.float32)
            mask = glare * skin
        else:
            # Whole-frame: the relative detector already excludes uniformly
            # bright areas (flat white background/shirt has excess ≈ 0), so
            # only true specular streaks — skin OR fabric — are selected.
            mask = glare
        if mask.sum() < 50:
            return img
        mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, sigma / 4.0))
        mask = np.clip(mask, 0.0, 1.0)

        # Pull luminance toward the baseline; restore washed-out chroma.
        a_base = cv2.GaussianBlur(A, (0, 0), sigma)
        b_base = cv2.GaussianBlur(B, (0, 0), sigma)
        L = L - excess * 0.65 * mask
        # Soft-knee on what remains near-blown inside glare zones: the
        # relative pull-down alone leaves 250-ish cores still glowing.
        L = L - np.clip(L - 205.0, 0.0, None) * 0.45 * mask
        A = A + (a_base - A) * 0.5 * mask
        B = B + (b_base - B) * 0.5 * mask

        out = cv2.merge((
            np.clip(L, 0, 255), np.clip(A, 0, 255), np.clip(B, 0, 255)
        )).astype(np.uint8)
        logger.info("[GLARE_REDUCE] corrected %.1f%% of frame", mask.mean() * 100)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    def _auto_brightness(self, img, delta: int):
        import numpy as np
        img = img.astype(np.int16)
        img = np.clip(img + delta, 0, 255).astype(np.uint8)
        return img

    def _apply_style_lut(self, img, style: str):
        """Apply cinematic color grading based on style mode"""
        import numpy as np
        import cv2

        img_float = img.astype(np.float32) / 255.0

        if style == "luxury":
            # Warm gold tones, raised shadows
            img_float[:, :, 0] *= 0.95   # B
            img_float[:, :, 1] *= 0.98   # G
            img_float[:, :, 2] *= 1.08   # R (warm)
            img_float = np.clip(img_float, 0, 1)
            # S-curve for contrast
            img_float = self._scurve(img_float, strength=0.3)

        elif style == "beach":
            # Warm + slight cyan in highlights
            img_float[:, :, 0] *= 1.05
            img_float[:, :, 2] *= 1.1
            img_float = np.clip(img_float, 0, 1)
            img_float = self._scurve(img_float, strength=0.2)

        elif style == "corporate":
            # Cooler, neutral tones
            img_float[:, :, 0] *= 1.05
            img_float[:, :, 2] *= 0.97
            img_float = np.clip(img_float, 0, 1)

        elif style == "anime":
            # Boost saturation
            hsv = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 40, 0, 255).astype(np.uint8)
            img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        elif style == "temple":
            # Dramatic, warm amber glow
            img_float[:, :, 0] *= 0.85
            img_float[:, :, 2] *= 1.15
            img_float = np.clip(img_float, 0, 1)
            img_float = self._scurve(img_float, strength=0.4)

        elif style == "whatsapp_dp":
            # Natural ID/passport look: neutral white balance, corrected saturation,
            # and a tiny brightness lift only.
            img_float = self._gray_world_balance(img_float)
            img_float = self._normalize_saturation(img_float)
            img_float = np.clip(img_float * 1.015 + 0.006, 0, 1)

        elif style == "professional":
            # Studio professional headshot: neutral balance, slight warmth for skin,
            # and a clean S-curve that lifts shadows without blowing highlights.
            img_float = self._gray_world_balance(img_float)
            img_float = self._normalize_saturation(img_float)
            # Kept subtle: stronger pushes (1.05R/0.97B) read as an orange
            # cast on medium/dark skin tones.
            img_float[:, :, 2] = np.clip(img_float[:, :, 2] * 1.02, 0, 1)  # R warm
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] * 0.99, 0, 1)  # B cool
            img_float = self._scurve(img_float, strength=0.28)

        return (img_float * 255).astype(np.uint8)

    def _gray_world_balance(self, img):
        """Gentle neutral color balance without a stylized cast."""
        import numpy as np

        means = img.reshape(-1, 3).mean(axis=0)
        gray = means.mean()
        scale = gray / np.maximum(means, 1e-6)
        scale = np.clip(scale, 0.94, 1.06)
        return np.clip(img * scale, 0, 1)

    def _normalize_saturation(self, img):
        """Bring saturation toward a natural portrait range."""
        import cv2
        import numpy as np

        rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        mean_sat = float(sat.mean())

        if mean_sat > 98:
            factor = 0.88
        elif mean_sat < 42:
            factor = 1.12
        else:
            factor = 1.0

        hsv[:, :, 1] = np.clip(sat * factor, 0, 138)
        corrected_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
        return corrected_bgr.astype(np.float32) / 255.0

    def _scurve(self, img, strength: float = 0.3):
        """Apply S-curve for cinematic contrast"""
        import numpy as np
        # Midpoint lift/drop
        shadow = img * (1 - strength * 0.5)
        highlight = 1 - (1 - img) * (1 - strength * 0.5)
        # Simple blend based on luminance
        lum = img.mean(axis=2, keepdims=True)
        weight = lum
        result = shadow * (1 - weight) + highlight * weight
        return np.clip(result, 0, 1)

    # ─── Geometric Tilt Correction ────────────────────────────────────────────

    def _correct_tilt(self, input_path: str, job_id: str, angle_deg: float) -> Path:
        """Level a small face tilt by rotating the whole image around its
        center by the measured roll angle, then crop to the largest
        axis-aligned rectangle inside the rotated frame — otherwise the
        smeared border corners survive and show up as slanted edges when a
        later stage (passport compose) zooms out. Only for 'small_tilt' —
        larger tilts need generative correction, out of scope here."""
        import cv2
        import math

        out_path = OUTPUTS_DIR / f"{job_id}_leveled.png"
        img = cv2.imread(input_path)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Largest axis-aligned rectangle inside a w×h rect rotated by angle
        a = abs(math.radians(angle_deg))
        sin_a, cos_a = math.sin(a), math.cos(a)
        long_side, short_side = max(w, h), min(w, h)
        if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * short_side
            crop_w, crop_h = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            crop_w = (w * cos_a - h * sin_a) / cos_2a
            crop_h = (h * cos_a - w * sin_a) / cos_2a

        crop_w, crop_h = int(crop_w), int(crop_h)
        if 0 < crop_w <= w and 0 < crop_h <= h:
            x0 = (w - crop_w) // 2
            y0 = (h - crop_h) // 2
            rotated = rotated[y0:y0 + crop_h, x0:x0 + crop_w]

        cv2.imwrite(str(out_path), rotated)
        return out_path

    # ─── Face Restoration ─────────────────────────────────────────────────────

    def _face_restore(self, input_path: str, job_id: str) -> Path:
        """Face restoration chain: CodeFormer → GFPGAN → bilateral fallback.
        Models are cached after first load; only FaceRestoreHelper is recreated per image."""
        import time as _t
        _t0 = _t.time()
        out_path = OUTPUTS_DIR / f"{job_id}_face.png"
        logger.info(
            "[FACE_RESTORE_START] job=%s input=%s", job_id, input_path
        )

        # ── 1. CodeFormer (best quality, preserves identity) ──────────────
        # Fast settings: mobile0.25 detector (4× faster than resnet50),
        # no BiSeNet parser (saves 8s), single center face only.
        try:
            import torch
            import cv2
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            net, device = self._get_codeformer()
            logger.info("[CODEFORMER_START] job=%s device=%s", job_id, device)

            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_mobile0.25",  # 4× faster than resnet50 on CPU
                save_ext="png",
                use_parse=False,                    # skip BiSeNet parser — saves 8s
                device=device,
            )

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            face_helper.read_image(img)
            face_helper.get_face_landmarks_5(only_center_face=True, resize=480)
            face_helper.align_warp_face()

            with torch.inference_mode():
                for cropped in face_helper.cropped_faces:
                    t = (
                        torch.from_numpy(cropped.transpose(2, 0, 1))
                        .unsqueeze(0).float() / 255.0
                    ).to(device)
                    t = t * 2 - 1
                    output = net(t, w=0.7, adain=True)[0]
                    restored_face = ((output.squeeze(0).permute(1, 2, 0) + 1) / 2)
                    restored_face = restored_face.clamp(0, 1).cpu().numpy()
                    restored_face = (restored_face * 255).astype("uint8")
                    face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine()
            restored = face_helper.paste_faces_to_input_image()
            cv2.imwrite(str(out_path), restored)
            logger.info(
                "[CODEFORMER_COMPLETE] job=%s elapsed=%.1fs", job_id, _t.time() - _t0
            )
            return out_path

        except (ImportError, FileNotFoundError) as e:
            logger.info("CodeFormer unavailable (%s), trying GFPGAN", e)
        except Exception as e:
            logger.warning("CodeFormer failed: %s — trying GFPGAN", e)

        # ── 2. GFPGAN fallback ────────────────────────────────────────────
        # Use lock.acquire(timeout=30) instead of "with lock:" so a crash in
        # another thread cannot permanently deadlock this job at 45%.
        try:
            import cv2

            restorer = self._get_gfpgan()
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            _h, _w = img.shape[:2]
            # Cap large images at 1280px to keep face detection fast on CPU.
            if max(_h, _w) > 1280:
                _sc = 1280.0 / max(_h, _w)
                img = cv2.resize(
                    img, (int(_w * _sc), int(_h * _sc)), interpolation=cv2.INTER_AREA
                )
                logger.info(
                    "[GFPGAN_RESIZE] job=%s %dx%d → %dx%d",
                    job_id, _w, _h, img.shape[1], img.shape[0],
                )
            logger.info(
                "[GFPGAN_START] job=%s img=%dx%d", job_id, img.shape[1], img.shape[0]
            )
            _t_gfp = _t.time()
            # Timeout-aware lock: if another thread holds for >30s, skip rather than hang.
            _got_lock = self._gfpgan_lock.acquire(timeout=30)
            if not _got_lock:
                logger.warning(
                    "[GFPGAN_LOCK_TIMEOUT] job=%s could not acquire lock — "
                    "bilateral fallback", job_id
                )
                return self._face_restore_fallback(input_path, job_id)
            try:
                from pipelines.photo_restoration import photo_restorer
                restored, _ = photo_restorer.enhance_faces_optical(img, sharpen_eyes=True, smooth_skin=True)
            finally:
                self._gfpgan_lock.release()

            cv2.imwrite(str(out_path), restored)
            logger.info(
                "[FACE_RESTORE_COMPLETE] job=%s elapsed=%.1fs", job_id, _t.time() - _t_gfp
            )
            return out_path

        except (ImportError, FileNotFoundError) as e:
            logger.warning("GFPGAN unavailable (%s), using bilateral fallback", e)
        except Exception as e:
            logger.exception("GFPGAN failed: %s", e)

        return self._face_restore_fallback(input_path, job_id)

    def _face_restore_fallback(self, input_path: str, job_id: str) -> Path:
        """Pass-through fallback — image was already denoised in _opencv_enhance."""
        return Path(input_path)

    # ─── Upscaling ────────────────────────────────────────────────────────────

    def _upscale(self, input_path: str, job_id: str, factor: int) -> Path:
        """Upscaling chain: Real-ESRGAN (fast tiled) → Lanczos.
        Models are cached after first load."""
        out_path = OUTPUTS_DIR / f"{job_id}_upscaled.png"

        # ── 1. Real-ESRGAN (Fast GPU Tiled) ──────────────────────────────
        device = self._torch_device()
        if str(device) != "cpu":
            try:
                import cv2
                upsampler = self._get_realesrgan()
                img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                out_img, _ = upsampler.enhance(img, outscale=factor)
                cv2.imwrite(str(out_path), out_img)
                logger.info("[OK] Real-ESRGAN %dx upscale done in GPU tiled mode", factor)
                return out_path
            except Exception as e:
                logger.warning("Real-ESRGAN upscale failed: %s — trying Lanczos", e)

        # ── 2. Spandrel — HAT-L or SwinIR if weights present ─────────────
        spandrel_candidates = [
            Path("models/hat/HAT-L_SRx4_ImageNet-pretrain.pth"),
            Path("models/hat/HAT_SRx4.pth"),
            Path("models/swinir/SwinIR_x4.pth"),
        ]
        spandrel_model_file = next((p for p in spandrel_candidates if p.exists()), None)

        if spandrel_model_file:
            try:
                import torch
                import cv2
                import numpy as np

                model, device = self._get_spandrel(spandrel_model_file)

                img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

                with torch.inference_mode():
                    out_t = model(t).squeeze(0).clamp(0, 1)

                out_rgb = (out_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

                if factor != 4:
                    new_w = int(img.shape[1] * factor)
                    new_h = int(img.shape[0] * factor)
                    out_bgr = cv2.resize(out_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(str(out_path), out_bgr)
                logger.info("[OK] Spandrel (%s) %dx upscale done", spandrel_model_file.name, factor)
                return out_path

            except Exception as e:
                logger.warning("Spandrel upscale failed: %s — trying Real-ESRGAN", e)

        # ── 2. Real-ESRGAN fallback (GPU only — too slow on CPU) ─────────
        device = self._torch_device()
        if str(device) != "cpu":
            try:
                import cv2

                upsampler = self._get_realesrgan()
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=factor)
                cv2.imwrite(str(out_path), output)
                logger.info("[OK] Real-ESRGAN %dx upscale done", factor)
                return out_path

            except (ImportError, FileNotFoundError):
                logger.warning("Real-ESRGAN not available, using Lanczos fallback")
            except Exception as e:
                logger.exception("Upscale error: %s", e)
        else:
            logger.info("CPU detected — skipping Real-ESRGAN, using fast Lanczos upscale")

        return self._upscale_fallback(input_path, job_id, factor)

    @staticmethod
    def _install_torchvision_compat() -> None:
        """Provide the old torchvision functional_tensor import expected by BasicSR."""
        import sys
        import types

        module_name = "torchvision.transforms.functional_tensor"
        if module_name in sys.modules:
            return

        try:
            import torchvision.transforms.functional as functional
        except Exception:
            return

        shim = types.ModuleType(module_name)
        shim.rgb_to_grayscale = functional.rgb_to_grayscale
        sys.modules[module_name] = shim

    def _upscale_fallback(self, input_path: str, job_id: str, factor: int) -> Path:
        """OpenCV Lanczos upscaling fallback"""
        import cv2

        out_path = OUTPUTS_DIR / f"{job_id}_upscaled.png"
        img = cv2.imread(input_path)
        if img is None:
            return Path(input_path)

        h, w = img.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)
        # Cap at 8K
        max_dim = 7680
        if new_w > max_dim or new_h > max_dim:
            scale = max_dim / max(new_w, new_h)
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)

        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_path), upscaled)
        logger.info("[OK] Lanczos fallback %dx upscale: %dx%d", factor, new_w, new_h)
        return out_path

    # ─── Background Replace ───────────────────────────────────────────────────

    def _background_replace(self, input_path: str, job_id: str, config: dict) -> Path:
        """Remove background using rembg, replace with generated scene or solid color"""
        out_path = OUTPUTS_DIR / f"{job_id}_bg.png"

        # Generative background (config: background_engine=controlnet + background_prompt)
        # — canny-ControlNet keeps the subject's silhouette locked while the
        # scene behind them is regenerated from the prompt. Falls back to the
        # normal solid-color studio composite below on any failure.
        if config.get("background_engine") == "controlnet" and config.get("background_prompt"):
            try:
                return self._background_replace_controlnet(input_path, job_id, config)
            except Exception as e:
                logger.exception(
                    "[CONTROLNET_BG] job=%s failed (%s) — falling back to studio color",
                    job_id, e,
                )

        try:
            from PIL import Image
            import numpy as np
            import cv2

            with open(input_path, "rb") as f:
                raw_bytes = f.read()
            # BiRefNet-lite sigmoid masks are continuous [0,1] — no binary cuts,
            # no alpha matting needed. Hair/braids/ribbons are preserved natively.
            result = background_removal.remove_background(raw_bytes, job_id=job_id)

            # Get background color - check config first, then fall back to style
            bg_color_name = config.get("background_color")
            if bg_color_name:
                bg_color = self._get_color_by_name(bg_color_name)
            else:
                style = config.get("style_mode", "luxury")
                bg_color = self._get_style_bg_color(style)

            fg_no_bg = Image.open(io.BytesIO(result)).convert("RGBA")

            # Studio Portrait Auto-Crop: Generous headroom (14%) and balanced horizontal framing
            import numpy as _npc
            _arr = _npc.array(fg_no_bg)
            _alpha = _arr[:, :, 3]
            _rows = _npc.any(_alpha > 5, axis=1)
            _cols = _npc.any(_alpha > 5, axis=0)
            if _rows.any() and _cols.any():
                _rmin = int(_npc.where(_rows)[0][0])
                _rmax = int(_npc.where(_rows)[0][-1])
                _cmin = int(_npc.where(_cols)[0][0])
                _cmax = int(_npc.where(_cols)[0][-1])
                _h2, _w2 = _arr.shape[:2]

                # Generous top headroom (14%) so head and hair are never clipped
                _top_pad = max(24, int(_h2 * 0.14))
                _bot_pad = max(8, int(_h2 * 0.04))
                _px = max(16, int(_w2 * 0.08))

                _rmin = max(0, _rmin - _top_pad)
                _rmax = min(_h2 - 1, _rmax + _bot_pad)
                _cmin = max(0, _cmin - _px)
                _cmax = min(_w2 - 1, _cmax + _px)
                _new_h = _rmax - _rmin + 1
                _new_w = _cmax - _cmin + 1
                if _new_h < _h2 or _new_w < _w2:
                    fg_no_bg = Image.fromarray(_arr[_rmin:_rmax + 1, _cmin:_cmax + 1], "RGBA")
                    logger.info("[STUDIO_CROP] job=%s %dx%d→%dx%d with 14%% headroom",
                                job_id, _w2, _h2, _new_w, _new_h)

            bg = self._make_studio_bg(fg_no_bg.size, bg_color)
            composite = Image.alpha_composite(bg, fg_no_bg)
            composite.convert("RGB").save(str(out_path), "PNG")
            if os.environ.get("BIREFNET_DEBUG"):
                try:
                    composite.convert("RGB").save(str(OUTPUTS_DIR / "debug_final.png"))
                    if job_id:
                        composite.convert("RGB").save(
                            str(OUTPUTS_DIR / f"{job_id}_final_output.png")
                        )
                    logger.info("[BIREFNET_DEBUG] job=%s saved debug_final", job_id or "?")
                except Exception as _dbg_e:
                    logger.debug("[BIREFNET_DEBUG] final_composite save failed: %s", _dbg_e)
            logger.info("\u2705 Background removed & replaced with color: %s", bg_color_name or "style-default")
            return out_path

        except Exception as e:
            logger.exception("Background replace error: %s; using GrabCut fallback", e)
            return self._background_replace_fallback(input_path, job_id, config)

    def _background_replace_controlnet(self, input_path: str, job_id: str, config: dict) -> Path:
        """Generative background via canny-ControlNet SDXL inpaint. The
        subject is cut out with the same BiRefNet step used for the solid-
        color path, its alpha becomes the paint mask (background = white,
        subject = black), and the canny edge map of the *whole* frame is fed
        to the ControlNet so the regenerated background can't bleed into or
        deform the subject's outline. `config['background_prompt']` drives
        the scene (e.g. one of GET /api/styles' prompt_suffix values)."""
        import cv2
        import numpy as np
        from PIL import Image

        with open(input_path, "rb") as f:
            raw_bytes = f.read()
        result = background_removal.remove_background(raw_bytes, job_id=job_id)
        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        alpha = np.array(fg.split()[-1])

        # Paint mask: white = repaint (background), black = keep (subject).
        mask = Image.fromarray(255 - alpha)

        src_rgb = Image.open(input_path).convert("RGB").resize(fg.size)
        tmp_src  = OUTPUTS_DIR / f"{job_id}_cn_src.png"
        tmp_mask = OUTPUTS_DIR / f"{job_id}_cn_mask.png"
        src_rgb.save(str(tmp_src))
        mask.save(str(tmp_mask))

        cn = self._get_controlnet_pipeline()
        generated = cn.background_replace(
            str(tmp_src), str(tmp_mask),
            prompt=config["background_prompt"],
            controlnet_scale=float(config.get("controlnet_scale", 0.6)),
            job_id=job_id,
        )

        # Composite the ORIGINAL subject back over the generated background —
        # ControlNet locks edges but still resamples pixels inside the mask
        # border; re-pasting the exact foreground guarantees zero identity
        # drift, matching the identity guarantee of the solid-color path.
        gen_img = Image.open(generated).convert("RGB").resize(fg.size)
        gen_img = Image.composite(src_rgb, gen_img, mask=Image.fromarray(alpha))

        out_path = OUTPUTS_DIR / f"{job_id}_bg.png"
        gen_img.save(str(out_path), "PNG")
        logger.info("[OK] Background regenerated via ControlNet: %s", job_id)
        return out_path

    def _background_replace_fallback(self, input_path: str, job_id: str, config: dict) -> Path:
        """Local portrait-friendly fallback when rembg is unavailable."""
        import cv2
        import numpy as np

        out_path = OUTPUTS_DIR / f"{job_id}_bg.png"
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return Path(input_path)

        h, w = img.shape[:2]
        bg_color = self._get_color_by_name(
            config.get("background_color") or self._get_style_bg_color(config.get("style_mode", "luxury"))
        )[:3]
        bg_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.uint8)

        mask = np.zeros((h, w), np.uint8)
        rect = (
            max(1, int(w * 0.08)),
            max(1, int(h * 0.04)),
            max(2, int(w * 0.84)),
            max(2, int(h * 0.92)),
        )
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype("uint8")
        except Exception as e:
            logger.warning("GrabCut fallback failed: %s", e)
            fg_mask = np.zeros((h, w), np.uint8)
            fg_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha = cv2.GaussianBlur(fg_mask, (7, 7), 0).astype(np.float32) / 255.0
        alpha = alpha[:, :, None]

        bg = np.full_like(img, bg_bgr)
        composite = (img.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        logger.info("[OK] Background fallback applied with color: %s", config.get("background_color"))
        return out_path

    def _get_color_by_name(self, color_name: str) -> tuple:
        """Convert color name to RGBA tuple"""
        colors = {
            "red": (212, 98, 90, 255),
            "green": (109, 184, 138, 255),
            "blue": (175, 210, 245, 255),
            "light_blue": (175, 210, 245, 255),
            "white": (255, 255, 255, 255),
            "light_grey": (235, 235, 238, 255),
            "grey": (235, 235, 238, 255),
            "gray": (235, 235, 238, 255),
            "offwhite": (245, 245, 245, 255),
            "studio_dark": (35, 35, 42, 255),
            "warm_studio": (245, 238, 230, 255),
            "corporate_navy": (25, 42, 75, 255),
            "crimson_red": (180, 30, 40, 255),
            "studio_gradient": (240, 242, 245, 255),
        }
        if isinstance(color_name, tuple):
            return color_name
        color = str(color_name).strip().lower()
        if color.startswith("#") and len(color) == 7:
            try:
                return (
                    int(color[1:3], 16),
                    int(color[3:5], 16),
                    int(color[5:7], 16),
                    255,
                )
            except ValueError:
                pass

        if "light" in color and "blue" in color:
            return (175, 210, 245, 255)
        if "white" in color or "passport" in color:
            return (255, 255, 255, 255)
        if "grey" in color or "gray" in color:
            return (235, 235, 238, 255)
        if "dark" in color:
            return (35, 35, 42, 255)
        if "warm" in color or "beige" in color:
            return (245, 238, 230, 255)
        if "navy" in color:
            return (25, 42, 75, 255)
        if "red" in color or "crimson" in color:
            return (180, 30, 40, 255)
        if "blue" in color:
            return (175, 210, 245, 255)

        return colors.get(color, (255, 255, 255, 255))

    def _get_style_bg_color(self, style: str) -> tuple:
        colors = {
            "luxury": (20, 18, 15, 255),         # Dark charcoal
            "beach": (70, 130, 180, 255),          # Steel blue
            "temple": (45, 30, 15, 255),           # Dark amber
            "corporate": (240, 240, 245, 255),     # Light gray
            "anime": (135, 206, 235, 255),         # Sky blue
            "whatsapp_dp": (255, 255, 255, 255),   # White
            "professional": (245, 245, 248, 255),  # Studio off-white
        }
        return colors.get(style, (30, 30, 30, 255))

    # ─── Final Polish ─────────────────────────────────────────────────────────

    def _pro_portrait_polish(self, img):
        """Face-aware professional headshot polish: even exposure + skin smooth + sharp eyes."""
        import cv2
        import numpy as np

        h, w = img.shape[:2]

        # CLAHE in LAB for even, natural-looking exposure across the whole image
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b_ch)), cv2.COLOR_LAB2BGR)

        primary = self._detect_primary_face(img)
        if not primary:
            return img

        fx, fy, fw, fh = primary["bbox"]

        # Soft face-oval mask for skin smoothing
        skin_mask = np.zeros((h, w), np.uint8)
        cx, cy = fx + fw // 2, fy + int(fh * 0.50)
        cv2.ellipse(skin_mask, (cx, cy), (int(fw * 0.55), int(fh * 0.65)), 0, 0, 360, 255, -1)
        skin_mask = cv2.GaussianBlur(skin_mask, (31, 31), 0)
        skin_alpha = skin_mask.astype(np.float32) / 255.0

        # Bilateral smoothing blended at 25 % over the face: reduces blemishes
        # without the plastic look of heavy filtering.
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=28, sigmaSpace=28)
        blend = 0.25
        img = (
            smooth.astype(np.float32) * skin_alpha[:, :, None] * blend
            + img.astype(np.float32) * (1.0 - skin_alpha[:, :, None] * blend)
        ).astype(np.uint8)

        return img

    def _smart_crop_img(self, img, config: dict, job_id: str):
        """Remove empty solid-background margins via corner-colour sampling.
        Used for non-rembg paths (no bg_replace) where we still want tight framing.
        """
        import cv2
        import numpy as np
        h, w = img.shape[:2]
        if w < 50 or h < 50:
            return img
        s = max(6, min(w, h) // 20)
        corners = np.vstack([
            img[:s, :s].reshape(-1, 3),
            img[:s, max(0, w - s):].reshape(-1, 3),
            img[max(0, h - s):, :s].reshape(-1, 3),
            img[max(0, h - s):, max(0, w - s):].reshape(-1, 3),
        ])
        bg_bgr = np.median(corners, axis=0).astype(np.float32)
        diff = np.abs(img.astype(np.float32) - bg_bgr)
        mask = (np.max(diff, axis=2) > 35).astype(np.uint8)
        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return img
        x, y, cw, ch = cv2.boundingRect(coords)
        pad_x = max(8, int(w * 0.03))
        pad_y = max(8, int(h * 0.03))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        if (x2 - x1) >= w * 0.92 and (y2 - y1) >= h * 0.92:
            logger.info("[SMART_CROP_IMG] job=%s fills %.0f%%x%.0f%% — skip",
                        job_id, (x2 - x1) / w * 100, (y2 - y1) / h * 100)
            return img
        logger.info("[SMART_CROP_IMG] job=%s %dx%d→%dx%d (fg %dx%d @ %d,%d)",
                    job_id, w, h, x2 - x1, y2 - y1, cw, ch, x, y)
        return img[y1:y2, x1:x2]

    def _final_polish(self, input_path: str, job_id: str, config: dict) -> Path:
        """Final sharpening pass and metadata"""
        import cv2
        import numpy as np
        from PIL import Image

        out_path = OUTPUTS_DIR / f"{job_id}_final.png"
        img = cv2.imread(input_path)
        if img is None:
            return Path(input_path)

        style = config.get("style_mode", "luxury")

        # Professional/corporate: face-aware even lighting, skin polish, eye boost
        if style in ("professional", "corporate"):
            img = self._pro_portrait_polish(img)

        if config.get("sharpen", True):
            if style in ("professional", "corporate"):
                # Stronger unsharp mask for professional crispness, but face region is
                # softened by the bilateral above so this lands on edges/hair only.
                blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
                sharpened = cv2.addWeighted(img, 1.22, blurred, -0.22, 0)
            else:
                # Gentle clarity pass; avoid the crunchy/filtered look on faces.
                blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
                sharpened = cv2.addWeighted(img, 1.08, blurred, -0.08, 0)
        else:
            sharpened = img

        # Slight vignette for cinematic feel
        if style in ("luxury", "temple", "beach") and not config.get("background_replace"):
            sharpened = self._add_vignette(sharpened, strength=0.15)

        # Auto-crop for non-bg-replace ID styles: corner sampling removes plain margins
        if not config.get("background_replace") and style in ("passport", "whatsapp_dp"):
            sharpened = self._smart_crop_img(sharpened, config, job_id)

        target_w = int(config.get("output_width_px") or 413)
        target_h = int(config.get("output_height_px") or 531)
        dpi = int(config.get("output_dpi") or 300)
        margins = {
            "top": self._clamp_margin(config.get("margin_top_px"), target_h),
            "bottom": self._clamp_margin(config.get("margin_bottom_px"), target_h),
            "left": self._clamp_margin(config.get("margin_left_px"), target_w),
            "right": self._clamp_margin(config.get("margin_right_px"), target_w),
        }
        # passport_compose takes precedence over margins: it is the only path
        # that produces a spec-compliant head crop; _fit_with_margins merely
        # shrinks the whole frame into the canvas, which is not a passport crop.
        if config.get("passport_compose"):
            if config.get("background_replace"):
                # Background was recolored to the config color upstream —
                # pad and subject-threshold against that same color.
                bg_color = self._get_color_by_name(config.get("background_color") or "white")
                bg_bgr   = (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))
            else:
                # Original background kept: the config color has nothing to do
                # with what's in the image. Sample the image's own border so
                # padding blends in and the subject threshold actually works.
                bg_bgr = self._sample_border_bgr(sharpened)
            try:
                head_ratio = float(config.get("head_ratio") or 0.50)
            except (TypeError, ValueError):
                head_ratio = 0.50
            sharpened = self._passport_compose(
                sharpened, target_w, target_h, bg_bgr,
                head_ratio=head_ratio,
                # Padding is invisible on a replaced (uniform) background;
                # with the original background kept it shows as fill bands,
                # so the crop stays full-bleed instead.
                allow_padding=bool(config.get("background_replace")),
            )
        elif any(margins.values()):
            bg_color = self._get_color_by_name(config.get("background_color") or "white")
            sharpened = self._fit_with_margins(sharpened, target_w, target_h, margins, bg_color)
        else:
            sharpened = self._cover_resize(sharpened, target_w, target_h)

        rgb     = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb)
        pil_out.save(str(out_path), "PNG", dpi=(dpi, dpi), compress_level=0)
        jpg_path = out_path.with_suffix(".jpg")
        pil_out.save(str(jpg_path), "JPEG", quality=95, subsampling=0, dpi=(dpi, dpi))
        logger.info("[OK] Final output: %s  (+JPG %s)", out_path.name, jpg_path.name)
        return out_path

    def _clamp_margin(self, value, limit: int) -> int:
        try:
            margin = int(value or 0)
        except (TypeError, ValueError):
            margin = 0
        return max(0, min(limit // 3, margin))

    def _cover_resize(self, img, target_w: int, target_h: int):
        """Resize and center-crop to exact output dimensions."""
        import cv2

        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img

        scale = max(target_w / w, target_h / h)
        resized_w = max(target_w, int(round(w * scale)))
        resized_h = max(target_h, int(round(h * scale)))
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)

        x = max(0, (resized_w - target_w) // 2)
        y = max(0, (resized_h - target_h) // 2)
        return resized[y:y + target_h, x:x + target_w]

    def _fit_with_margins(self, img, target_w: int, target_h: int, margins: dict, bg_color: tuple):
        """Fit the image inside a passport canvas with explicit background margins."""
        import cv2
        import numpy as np

        content_w = max(1, target_w - margins["left"] - margins["right"])
        content_h = max(1, target_h - margins["top"] - margins["bottom"])
        h, w = img.shape[:2]
        scale = min(content_w / w, content_h / h)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)

        bg_bgr = np.array([bg_color[2], bg_color[1], bg_color[0]], dtype=np.uint8)
        canvas = np.full((target_h, target_w, 3), bg_bgr, dtype=np.uint8)
        x = margins["left"] + max(0, (content_w - resized_w) // 2)
        y = margins["top"] + max(0, (content_h - resized_h) // 2)
        canvas[y:y + resized_h, x:x + resized_w] = resized
        return canvas

    @staticmethod
    def _make_studio_bg(size, rgba_color):
        """Studio-backdrop version of a solid color: soft radial falloff —
        slightly brighter behind the head (upper center), gently darker
        toward the corners — like a lit studio wall instead of flat paint.
        Amplitude stays within ±22 of the base color so _passport_compose's
        subject threshold (diff > 22 vs the flat color) still treats every
        background pixel as background."""
        import numpy as np
        from PIL import Image

        w, h = size
        r, g, b = rgba_color[0], rgba_color[1], rgba_color[2]

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        # Light source centered behind the head (upper-middle of frame)
        dx = (xs / max(1, w - 1)) - 0.5
        dy = (ys / max(1, h - 1)) - 0.35
        dist = np.sqrt(dx * dx + dy * dy) / 0.85          # 0 center → ~1 corners
        gain = 1.05 - 0.13 * np.clip(dist, 0.0, 1.0)      # +5% center, -8% corners

        arr = np.empty((h, w, 4), dtype=np.uint8)
        for i, c in enumerate((r, g, b)):
            arr[:, :, i] = np.clip(c * gain, 0, 255).astype(np.uint8)
        arr[:, :, 3] = 255
        return Image.fromarray(arr, "RGBA")

    @staticmethod
    def _sample_border_bgr(img) -> tuple:
        """Median BGR of the top-left and top-right corner patches — in a
        portrait these are almost always pure background, unlike the side
        borders (hair) or bottom (shoulders/clothing), which drag the
        median toward the subject."""
        import numpy as np

        h, w = img.shape[:2]
        patch = max(4, min(h, w) // 12)
        corners = np.vstack([
            img[:patch, :patch].reshape(-1, 3),
            img[:patch, w - patch:].reshape(-1, 3),
        ])
        med = np.median(corners, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    def _passport_compose(
        self,
        img,
        target_w: int,
        target_h: int,
        bg_bgr: tuple,
        head_ratio: float = 0.50,
        allow_padding: bool = False,
    ):
        """
        Passport-photo framing using actual subject bounding box.

        Step 1 — Subject bbox from background threshold:
          The image already has a solid bg_bgr background (applied by
          _background_replace before this point). Thresholding against that
          colour gives the exact visible extent of the person — hair, ears,
          shoulders, clothing — with no estimation needed.

        Step 2 — Face detection for chin and horizontal centre:
          Haar face cascade locates the chin bottom and refines the horizontal
          centre more precisely than the subject bbox midpoint alone.

        Step 3 — Scale so head (crown→chin) = head_ratio of canvas height.
          Default 0.50 gives chest-pocket framing (head + shoulders + upper
          chest visible); strict passport specs want ~0.55–0.80 with 0.68 a
          sensible middle. Overridable per-job via the `head_ratio` config key.

        Step 4 — Place crown at 8% from top.
          Chin safety clamp prevents bottom overflow.

        Fallback chain:
          no subject + no face → _cover_resize
          no subject → face-estimation only (pre-v2 approach)
          no face    → subject bbox used, chin estimated at 65% of subject height
        """
        import cv2
        import numpy as np

        h, w = img.shape[:2]

        # ── Step 1: Subject bounding box from background colour ───────────────
        bg = np.array(bg_bgr, dtype=np.float32)
        diff = np.abs(img.astype(np.float32) - bg).max(axis=2)
        raw_mask = (diff > 22).astype(np.uint8)

        # Morphological close fills holes (white collar on white bg, tooth gaps, etc.)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        rows_hit = np.where(np.any(closed, axis=1))[0]
        cols_hit = np.where(np.any(closed, axis=0))[0]
        have_subj = rows_hit.size > 0 and cols_hit.size > 0

        if have_subj:
            sy1 = int(rows_hit[0])
            sy2 = int(rows_hit[-1])
            sx1 = int(cols_hit[0])
            sx2 = int(cols_hit[-1])
            bbox_cx = (sx1 + sx2) // 2

        # ── Step 2: Face detection ────────────────────────────────────────────
        # InsightFace (RetinaFace) first: on full-body shots with patterned
        # clothing, Haar detects false "faces" in the fabric and — picked by
        # area — they beat the real, smaller face, so the crop frames the
        # chest as if it were the head. Haar remains the fallback.
        faces = []
        try:
            app = self._get_face_analysis()
            dets = app.get(img)
            if dets:
                d = max(dets, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                faces = [(x1, y1, max(1, x2 - x1), max(1, y2 - y1))]
                logger.info("[PASSPORT_COMPOSE] face via InsightFace")
        except Exception as e:
            logger.info("InsightFace unavailable for compose (%s)", e)
        if not faces:
            detected = self._detect_all_faces(img)
            if detected:
                faces = [detected[0]["bbox"]]
        have_face = len(faces) > 0

        if not have_subj and not have_face:
            logger.info("[PASSPORT_COMPOSE] no subject or face — cover resize fallback")
            return self._cover_resize(img, target_w, target_h)

        if have_face:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            # Sanity: a real head sits in the upper part of the subject. A
            # "face" whose center lies below 55% of the subject bbox is a
            # false positive (fabric pattern) — discard it.
            if have_subj:
                face_cy = fy + fh / 2
                subj_h = max(1, sy2 - sy1)
                if face_cy > sy1 + subj_h * 0.55:
                    logger.info(
                        "[PASSPORT_COMPOSE] face at %.0f%% of subject height — "
                        "rejected as false positive", (face_cy - sy1) / subj_h * 100,
                    )
                    have_face = False
        if have_face:
            face_cx   = fx + fw // 2
            haar_crown = max(0, fy - int(fh * 0.35))   # hairline → crown
            haar_chin  = min(h - 1, fy + fh + int(fh * 0.05))  # box ≈ includes chin

        # ── Step 3: Determine crown_y, chin_y, subj_cx ───────────────────────
        if have_subj and have_face:
            # Crown: take the higher of BiRefNet subject top and Haar estimate
            crown_y = min(sy1, haar_crown)
            chin_y  = haar_chin
            # Horizontal: face centre if it falls inside subject bbox, else bbox mid
            subj_cx = face_cx if sx1 <= face_cx <= sx2 else bbox_cx

        elif have_subj and not have_face:
            # No face: crown = subject top; estimate chin at 65% of subject height
            crown_y  = sy1
            subj_cx  = bbox_cx
            subj_h   = sy2 - sy1 + 1
            chin_y   = sy1 + int(subj_h * 0.65)

        else:  # have_face, not have_subj
            crown_y = haar_crown
            chin_y  = haar_chin
            subj_cx = face_cx

        head_h = max(1, chin_y - crown_y)

        # ── Step 4: Scale so head = head_ratio of canvas height ─────────────
        head_ratio = float(np.clip(head_ratio, 0.35, 0.8))
        TARGET_TOP = int(target_h * 0.08)   # crown headroom, reused in Step 5
        scale = (target_h * head_ratio) / head_h
        # Never zoom out so far that the subject floats inside the canvas.
        # Without padding allowed, the source must cover the whole canvas.
        # With padding allowed (replaced background), padding may appear
        # ONLY above the crown as natural headroom — the width and
        # everything below the crown must still fill the frame, otherwise
        # the result is a small photo floating in empty background.
        if allow_padding:
            below_crown = max(1, h - crown_y)
            min_scale = max(target_w / w, (target_h - TARGET_TOP) / below_crown)
        else:
            min_scale = max(target_w / w, target_h / h)
        if scale < min_scale:
            logger.info(
                "[PASSPORT_COMPOSE] head_ratio %.2f needs scale %.2f but frame "
                "fill needs %.2f — clamping",
                head_ratio, scale, min_scale,
            )
            scale = min_scale
        scale = float(np.clip(scale, 0.25, 4.0))

        new_w  = max(1, int(round(w * scale)))
        new_h  = max(1, int(round(h * scale)))
        interp = cv2.INTER_LANCZOS4 if scale <= 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Actual scale factors after resize (may differ from scale if clamped).
        # Use these for ALL coordinate transforms so positions in resized-image
        # space are correct regardless of aspect-ratio adjustments.
        asx = new_w / w if w > 0 else scale
        asy = new_h / h if h > 0 else scale

        # ── Step 5: Canvas window position ───────────────────────────────────
        crown_s   = int(crown_y  * asy)
        chin_s    = int(chin_y   * asy)
        subj_cx_s = int(subj_cx  * asx)

        # Crown at 8% from top (TARGET_TOP computed in Step 4)
        src_y = crown_s - TARGET_TOP
        src_x = subj_cx_s - (target_w // 2)

        # Safety: chin must remain visible (4% min bottom margin)
        MIN_BOT = int(target_h * 0.04)
        if chin_s - src_y > target_h - MIN_BOT:
            src_y = chin_s - (target_h - MIN_BOT)

        # Keep the crop window inside the source (min_scale above guarantees
        # coverage). With padding allowed, only the top may fall outside the
        # source — invisible headroom on a replaced background; width and
        # bottom always stay covered so the subject fills the frame.
        src_x = int(np.clip(src_x, 0, max(0, new_w - target_w)))
        if allow_padding:
            src_y = min(src_y, new_h - target_h)
        else:
            src_y = int(np.clip(src_y, 0, max(0, new_h - target_h)))

        # ── Step 6: Composite onto background canvas ──────────────────────────
        canvas = np.full((target_h, target_w, 3), bg_bgr, dtype=np.uint8)
        sr_x1 = max(0, src_x);         sr_y1 = max(0, src_y)
        sr_x2 = min(new_w, src_x + target_w); sr_y2 = min(new_h, src_y + target_h)
        ds_x1 = sr_x1 - src_x;         ds_y1 = sr_y1 - src_y
        ds_x2 = ds_x1 + (sr_x2 - sr_x1); ds_y2 = ds_y1 + (sr_y2 - sr_y1)
        if sr_x2 > sr_x1 and sr_y2 > sr_y1:
            if ds_x1 > 0 or ds_y1 > 0 or ds_x2 < target_w or ds_y2 < target_h:
                # Source doesn't cover the canvas — feather the boundary so
                # the photo fades into the fill instead of showing a hard
                # rectangle seam (fill tone never matches exactly once the
                # image has been through enhancement).
                src_layer = canvas.copy()
                src_layer[ds_y1:ds_y2, ds_x1:ds_x2] = resized[sr_y1:sr_y2, sr_x1:sr_x2]
                mask = np.zeros((target_h, target_w), np.float32)
                mask[ds_y1:ds_y2, ds_x1:ds_x2] = 1.0
                mask = cv2.GaussianBlur(mask, (0, 0), max(5, target_w // 20))
                canvas = (
                    src_layer.astype(np.float32) * mask[..., None]
                    + canvas.astype(np.float32) * (1.0 - mask[..., None])
                ).astype(np.uint8)
            else:
                canvas[ds_y1:ds_y2, ds_x1:ds_x2] = resized[sr_y1:sr_y2, sr_x1:sr_x2]

        canvas_crown = crown_s - src_y
        canvas_chin  = chin_s  - src_y
        logger.info(
            "[PASSPORT_COMPOSE] subj=%s bbox=(%s,%d→%s,%d) "
            "crown=%d chin=%d head_h=%d scale=%.2f "
            "canvas_crown=%dpx(%.0f%%) canvas_chin=%dpx(%.0f%%) src=(%d,%d)",
            "Y" if have_subj else "N",
            sx1 if have_subj else "?", sy1 if have_subj else -1,
            sx2 if have_subj else "?", sy2 if have_subj else -1,
            crown_y, chin_y, head_h, scale,
            canvas_crown, canvas_crown / target_h * 100,
            canvas_chin,  canvas_chin  / target_h * 100,
            src_x, src_y,
        )
        return canvas

    # ─── Uniform Swap ─────────────────────────────────────────────────────────
    # ─── Uniform Swap ─────────────────────────────────────────────────────────
    # ─── Uniform Swap ─────────────────────────────────────────────────────────

    def shirt_replacement(
        self,
        person_path: str,
        uniform_path: str,
        job_id: str,
        bg_color: tuple = (4, 126, 246),   # studio ID-photo blue
        engine: str = "catvton",           # "catvton" | "idm_vton" | "vton" | "classic" | "controlnet" | "qwen_edit"
    ) -> Path:
        """
        Merge a person photo with a uniform image:
        - engine="catvton" / "vton": Runs CatVTON virtual try-on diffusion followed by high-fidelity Image Compositing
        - engine="idm_vton": Runs IDM-VTON virtual try-on diffusion followed by high-fidelity Image Compositing
        - engine="qwen_edit": Runs Qwen-Image-Edit AI try-on with biometric face recovery
        - engine="controlnet": Classic composite + Canny-ControlNet img2img restyle
        - engine="classic": BiRefNet + MediaPipe/InsightFace alignment + collar blend
        """
        if engine in ("catvton", "vton"):
            try:
                logger.info("[UNIFORM_CATVTON] Starting CatVTON try-on + Image Compositing for job=%s", job_id)
                from pipelines.catvton_pipeline import CatVTONPipeline
                catvton = CatVTONPipeline()
                return catvton.tryon(
                    person_image=person_path,
                    uniform_image=uniform_path,
                    bg_color=bg_color,
                    job_id=job_id,
                    apply_compositing=True,
                )
            except Exception as e:
                logger.warning("[UNIFORM_CATVTON] job=%s failed (%s) — falling back to classic pipeline", job_id, e)

        elif engine == "idm_vton":
            try:
                logger.info("[UNIFORM_IDMVTON] Starting IDM-VTON try-on + Image Compositing for job=%s", job_id)
                from pipelines.idm_vton_pipeline import IDMVTONPipeline
                idm = IDMVTONPipeline()
                return idm.tryon(
                    person_image=person_path,
                    uniform_image=uniform_path,
                    bg_color=bg_color,
                    job_id=job_id,
                    apply_compositing=True,
                )
            except Exception as e:
                logger.warning("[UNIFORM_IDMVTON] job=%s failed (%s) — falling back to classic pipeline", job_id, e)

        import cv2
        import numpy as np
        import os
        import tempfile
        from PIL import Image
        from io import BytesIO

        TARGET_H = 1000   # normalise both inputs to this height

        def _remove_solid_bg(img: Image.Image) -> Image.Image:
            arr = np.array(img.convert("RGBA"))
            rgb_u8 = arr[:, :, :3]
            rgb = rgb_u8.astype(np.int16)
            h0, w0 = rgb.shape[:2]
            pad = max(8, min(h0, w0) // 18)
            corners = np.concatenate([
                rgb[:pad, :pad].reshape(-1, 3),
                rgb[:pad, -pad:].reshape(-1, 3),
                rgb[-pad:, :pad].reshape(-1, 3),
                rgb[-pad:, -pad:].reshape(-1, 3),
            ])
            bg = np.median(corners, axis=0)
            hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
            corner_hsv = cv2.cvtColor(
                np.uint8([[bg.clip(0, 255)]]), cv2.COLOR_RGB2HSV
            )[0, 0].astype(np.int16)

            if corner_hsv[1] > 40:
                hue_delta = np.abs(hsv[:, :, 0].astype(np.int16) - corner_hsv[0])
                hue_delta = np.minimum(hue_delta, 180 - hue_delta)
                bg_hint = np.array(bg_color[:3], dtype=np.int32)
                hint_delta = rgb.astype(np.int32) - bg_hint
                hint_dist = np.sqrt((hint_delta ** 2).sum(axis=2))
                bg_candidates = (
                    (hint_dist < 65)
                    | (
                        (hue_delta < 13)
                        & (hsv[:, :, 1] > 120)
                        & (hsv[:, :, 2] > 175)
                    )
                )
            else:
                diff = np.abs(rgb - bg).max(axis=2)
                bg_candidates = diff < 46

            candidate_u8 = cv2.morphologyEx(
                bg_candidates.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
                iterations=2,
            )
            count, labels = cv2.connectedComponents(candidate_u8, connectivity=8)
            border_labels = set(np.unique(labels[0, :]))
            border_labels.update(np.unique(labels[-1, :]))
            border_labels.update(np.unique(labels[:, 0]))
            border_labels.update(np.unique(labels[:, -1]))
            border_labels.discard(0)
            bg_mask = np.isin(labels, list(border_labels)) if count > 1 else candidate_u8 > 0

            alpha = arr[:, :, 3].copy()
            alpha[bg_mask] = 0
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            arr[:, :, 3] = alpha
            return Image.fromarray(arr, "RGBA")

        # ── 1. Remove backgrounds ─────────────────────────────────────────
        try:
            runtime_tmp_dir = Path("outputs") / "runtime_tmp"
            numba_cache_dir = Path("outputs") / "numba_cache"
            runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
            numba_cache_dir.mkdir(parents=True, exist_ok=True)

            runtime_tmp = str(runtime_tmp_dir.resolve())
            os.environ["TMPDIR"] = runtime_tmp
            os.environ["TEMP"] = runtime_tmp
            os.environ["TMP"] = runtime_tmp
            os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir.resolve()))
            tempfile.tempdir = runtime_tmp

            with open(person_path, "rb") as f:
                person_rgba_raw = Image.open(
                    BytesIO(background_removal.remove_background(f.read()))
                ).convert("RGBA")

            with open(uniform_path, "rb") as f:
                u_bytes = f.read()
                try:
                    uniform_rgba_raw = Image.open(
                        BytesIO(background_removal.remove_background(u_bytes))
                    ).convert("RGBA")
                except Exception:
                    uniform_rgba_raw = _remove_solid_bg(Image.open(BytesIO(u_bytes)).convert("RGBA"))

            logger.info("[OK] Background removal complete (BiRefNet-HR for person & uniform)")
        except Exception as e:
            logger.warning("BiRefNet removal failed, using original images: %s", e)
            person_rgba_raw  = Image.open(person_path).convert("RGBA")
            uniform_rgba_raw = Image.open(uniform_path).convert("RGBA")

        # ── 2. Normalise both to TARGET_H ─────────────────────────────────
        def _resize_h(img: Image.Image, h: int) -> Image.Image:
            w = max(1, int(img.width * h / img.height))
            return img.resize((w, h), Image.LANCZOS)

        person_rgba  = _resize_h(person_rgba_raw,  TARGET_H)
        uniform_rgba = _resize_h(uniform_rgba_raw, TARGET_H)

        person_arr  = np.array(person_rgba)          # H W 4  (RGBA)
        uniform_arr = np.array(uniform_rgba).copy()  # H W 4

        # Save original alpha for accurate chin detection (before erosion/feathering)
        uniform_alpha_orig = uniform_arr[:, :, 3].copy()

        # Erode uniform edges to remove rembg colour fringe (blue/teal on collar)
        _ku = np.ones((3, 3), np.uint8)
        _ua = cv2.erode(uniform_arr[:, :, 3].copy(), _ku, iterations=1)
        _ua = cv2.GaussianBlur(_ua, (7, 7), 0)
        uniform_arr[:, :, 3] = _ua

        # Feather collar top edge per-column so collar wraps into neck naturally
        _u_a_f = _ua.astype(np.float32)
        _first_opaque = np.argmax(_u_a_f > 40, axis=0)
        _has_content  = _u_a_f[_first_opaque, np.arange(_ua.shape[1])] > 40
        _is_collar    = (_first_opaque < _ua.shape[0] * 0.65) & _has_content
        _rows_grid    = np.arange(_ua.shape[0])[:, None]
        _dist_grid    = _rows_grid - _first_opaque[None, :]
        _ramp         = np.clip(_dist_grid / 25.0, 0.0, 1.0)
        _ua_ramped    = np.where(_is_collar[None, :], _u_a_f * _ramp, _u_a_f)
        uniform_arr[:, :, 3] = np.clip(_ua_ramped, 0, 255).astype(np.uint8)

        ph, pw = person_arr.shape[:2]
        uh, uw = uniform_arr.shape[:2]

        # BGR for OpenCV ops
        person_bgr  = cv2.cvtColor(person_arr[:, :, :3],  cv2.COLOR_RGB2BGR)
        uniform_bgr = cv2.cvtColor(uniform_arr[:, :, :3], cv2.COLOR_RGB2BGR)

        # ── 3. Detect faces using InsightFace ─────────────────────────────
        p_detected = self._detect_all_faces(person_bgr)
        u_detected = self._detect_all_faces(uniform_bgr)

        if len(p_detected) == 0:
            raise ValueError("No face detected in person image — use a clear front-facing photo")

        fx, fy, fw, fh = p_detected[0]["bbox"]

        # Only count uniform face if it's in the top 40% and large enough
        real_u_face = next(
            (f["bbox"] for f in u_detected
             if f["bbox"][1] < uh * 0.40 and f["bbox"][2] > uw * 0.08),
            None
        )

        # ── 4A. Uniform has a model face → enhanced face swap ─────────────
        if real_u_face:
            uFx, uFy, uFw, uFh = real_u_face

            # Build flat destination (uniform composited over bg_color)
            bg_bgr  = np.array([bg_color[2], bg_color[1], bg_color[0]], np.float32)
            alpha_u = uniform_arr[:, :, 3:4].astype(np.float32) / 255.0
            dst_bgr = (
                uniform_bgr.astype(np.float32) * alpha_u
                + bg_bgr * (1.0 - alpha_u)
            ).astype(np.uint8)

            result_bgr = self._do_seamless_face_swap(
                person_bgr, (fx, fy, fw, fh),
                dst_bgr,    (uFx, uFy, uFw, uFh),
            )
            result_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            logger.info("[OK] Face swap complete")

        # ── 4B. Head + neck only, AI-aligned to collar ───────────────────
        else:
            collar_cx = uw // 2

            # ── Collar y-position ─────────────────────────────────────────
            collar_y = int(uh * 0.50)
            for r in range(int(uh * 0.10), int(uh * 0.70)):
                if uniform_alpha_orig[r, collar_cx] > 40:
                    collar_y = r
                    break

            # ── AI: MediaPipe chin + jaw landmarks ────────────────────────
            _h_p, _w_p = person_bgr.shape[:2]
            chin_y_person = fy + int(fh * 1.05)   # Haar fallback
            jaw_w_person  = fw * 0.88              # Haar fallback
            try:
                import mediapipe as _mp
                _rgb_p = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
                with _mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    min_detection_confidence=0.4,
                ) as _fm:
                    _res = _fm.process(_rgb_p)
                    if _res.multi_face_landmarks:
                        _lms = _res.multi_face_landmarks[0].landmark
                        chin_y_person = int(_lms[152].y * _h_p)        # landmark 152 = chin tip
                        jaw_w_person  = abs(_lms[454].x - _lms[234].x) * _w_p  # jaw span
            except Exception:
                pass

            # ── Scale: align person's chin to collar_y ────────────────────
            scale_p = collar_y / max(chin_y_person, 1)
            scale_p = min(scale_p, uw * 0.85 / max(fw, 1))  # cap head at 85 % canvas width

            # ── Head+neck crop: from top of image to chin + 28 % face height ─
            neck_show_px  = int(fh * 0.28)
            head_bottom   = min(ph, chin_y_person + neck_show_px)
            person_head   = person_arr[:head_bottom, :]
            sp_w = max(1, int(pw        * scale_p))
            sp_h = max(1, int(head_bottom * scale_p))
            person_scaled = cv2.resize(person_head, (sp_w, sp_h),
                                       interpolation=cv2.INTER_LANCZOS4)
            chin_scaled   = int(chin_y_person * scale_p)

            # ── Background removal on the head crop ───────────────────────
            k3     = np.ones((3, 3), np.uint8)
            bg_fill = np.array(bg_color[:3], np.uint8)

            person_orig_crop = np.array(
                _resize_h(Image.open(person_path).convert("RGB"), TARGET_H)
            )[:head_bottom, :]
            _corners = np.concatenate([
                person_orig_crop[:15, :15].reshape(-1, 3),
                person_orig_crop[:15, -15:].reshape(-1, 3),
                person_orig_crop[-15:, :15].reshape(-1, 3),
                person_orig_crop[-15:, -15:].reshape(-1, 3),
            ])
            solid_bg = float(np.std(_corners, axis=0).mean()) < 30

            p_alpha = person_scaled[:, :, 3].copy()
            if solid_bg:
                p_alpha = cv2.erode(p_alpha, k3, iterations=3)
                person_scaled[p_alpha < 10, :3] = bg_fill
                p_alpha = cv2.GaussianBlur(p_alpha, (21, 21), 0)
                bg_rgb  = np.median(_corners, axis=0).astype(np.uint8)
                orig_s  = cv2.resize(person_orig_crop, (sp_w, sp_h),
                                     interpolation=cv2.INTER_LANCZOS4)
                diff    = np.abs(orig_s.astype(np.int32) -
                                 bg_rgb.astype(np.int32)).max(axis=2)
                person_scaled[diff < 35, :3] = bg_fill
                p_alpha = np.where(diff < 35, 0, p_alpha).astype(np.uint8)
            else:
                p_alpha = cv2.erode(p_alpha, k3, iterations=3)
                person_scaled[p_alpha == 0, :3] = bg_fill
                p_alpha = cv2.GaussianBlur(p_alpha, (7, 7), 0)

            # ── Neck fade: chin → bottom of crop (full fade into collar) ──
            fade_s = chin_scaled + int(fh * scale_p * 0.06)
            fade_e = sp_h
            if fade_e > fade_s:
                ramp = np.linspace(1.0, 0.0, fade_e - fade_s, dtype=np.float32)
                p_alpha[fade_s:fade_e] = (
                    p_alpha[fade_s:fade_e].astype(np.float32) * ramp[:, None]
                ).astype(np.uint8)

            # ── Neck sides: feathered column using MediaPipe jaw width ────
            neck_cx   = int((fx + fw * 0.5) * scale_p)
            neck_half = int(jaw_w_person * scale_p * 0.27)   # ~54 % of jaw width total
            neck_half = max(int(fw * scale_p * 0.24),
                            min(neck_half, int(fw * scale_p * 0.34)))
            side_left  = max(0, neck_cx - neck_half)
            side_right = min(sp_w, neck_cx + neck_half)
            feather_w  = 18
            for col in range(max(0, side_left - feather_w), side_left):
                t = (col - (side_left - feather_w)) / feather_w
                p_alpha[chin_scaled:, col] = (
                    p_alpha[chin_scaled:, col].astype(np.float32) * t
                ).astype(np.uint8)
            p_alpha[chin_scaled:, :max(0, side_left - feather_w)] = 0
            for col in range(side_right, min(sp_w, side_right + feather_w)):
                t = 1.0 - (col - side_right) / feather_w
                p_alpha[chin_scaled:, col] = (
                    p_alpha[chin_scaled:, col].astype(np.float32) * t
                ).astype(np.uint8)
            p_alpha[chin_scaled:, min(sp_w, side_right + feather_w):] = 0

            person_scaled[:, :, 3] = p_alpha

            # ── Place head on canvas (centred, from top) ──────────────────
            px_off   = collar_cx - sp_w // 2
            canvas_h = uh
            canvas_w = uw
            canvas   = np.zeros((canvas_h, canvas_w, 4), np.uint8)
            p_x0 = max(0, px_off);  p_x1 = min(canvas_w, px_off + sp_w)
            s_x0 = max(0, -px_off); s_x1 = s_x0 + (p_x1 - p_x0)
            p_y1 = min(canvas_h, sp_h)
            if p_x1 > p_x0 and p_y1 > 0:
                src   = person_scaled[0:p_y1, s_x0:s_x1]
                src_a = (src[:, :, 3] / 255.0)[:, :, None]
                canvas[0:p_y1, p_x0:p_x1, :3] = (
                    src[:, :, :3].astype(np.float32) * src_a
                ).astype(np.uint8)
                canvas[0:p_y1, p_x0:p_x1, 3] = src[:, :, 3]

            # ── Composite uniform: collar wraps over fading neck ──────────
            u_f = uniform_arr.astype(np.float32)
            u_a = u_f[:, :, 3:4] / 255.0

            # Face-only protect: prevent collar touching the face proper
            face_protect = np.zeros((canvas_h, canvas_w), np.uint8)
            face_cx_c = int(px_off + (fx + fw * 0.5) * scale_p)
            face_cy_c = int((fy + fh * 0.50) * scale_p)
            cv2.ellipse(face_protect,
                        (face_cx_c, face_cy_c),
                        (max(8, int(fw * scale_p * 0.62)),
                         max(8, int(fh * scale_p * 0.72))),
                        0, 0, 360, 255, -1)
            face_protect = cv2.GaussianBlur(face_protect, (51, 51), 0)

            person_pres  = (canvas[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
            prot_alpha   = (face_protect.astype(np.float32) / 255.0)[:, :, None] * person_pres
            u_a_eff      = u_a * (1.0 - prot_alpha)

            canvas[:, :, :3] = (
                u_f[:, :, :3] * u_a_eff
                + canvas[:, :, :3].astype(np.float32) * (1.0 - u_a_eff)
            ).astype(np.uint8)
            canvas[:, :, 3] = np.maximum(canvas[:, :, 3], uniform_arr[:, :, 3])
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            rgba_canvas = Image.fromarray(canvas, "RGBA")
            result_pil  = Image.new("RGB", rgba_canvas.size, bg_color)
            result_pil.paste(rgba_canvas, mask=rgba_canvas.split()[3])

        # ── 5. Trim background padding, keep content rows ────────────────
        arr    = np.array(result_pil)
        bg_arr = np.array(bg_color[:3], dtype=np.int32)
        diff   = np.abs(arr.astype(np.int32) - bg_arr).max(axis=(1, 2))
        non_bg = np.where(diff > 15)[0]
        if len(non_bg) > 0:
            top_content    = max(0, non_bg[0] - 10)
            bottom_content = min(arr.shape[0], non_bg[-1] + 20)
            result_pil = result_pil.crop((0, top_content,
                                          result_pil.width, bottom_content))

        out_path = OUTPUTS_DIR / f"{job_id}_uniform.png"
        result_pil = self._frame_uniform_portrait(result_pil, bg_color)
        result_pil = self._enhance_uniform_portrait(result_pil, bg_color)
        result_pil.save(str(out_path), "PNG")
        logger.info("[OK] Uniform swap done: %s", out_path)

        if engine == "qwen_edit":
            try:
                logger.info("[UNIFORM_QWEN_EDIT] Starting Qwen-Image-Edit AI try-on for job=%s", job_id)
                from pipelines.qwen_edit_pipeline import QwenEditPipeline
                qwen_pipe = QwenEditPipeline()
                qwen_out = qwen_pipe.uniform_swap(
                    person_path=person_path,
                    uniform_path=uniform_path,
                    steps=20,
                    guidance_scale=7.5,
                    job_id=job_id,
                )
                if Path(qwen_out).exists():
                    qwen_img = Image.open(qwen_out).convert("RGB")
                    qwen_img = self._frame_uniform_portrait(qwen_img, bg_color)
                    qwen_img = self._enhance_uniform_portrait(qwen_img, bg_color)

                    # Restore the verified biometric identity region
                    base_img = np.array(Image.open(out_path).convert("RGB"))
                    ai_img = np.array(qwen_img.resize((base_img.shape[1], base_img.shape[0]), Image.LANCZOS))
                    base_bgr = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
                    dets = self._get_face_analysis().get(base_bgr)
                    if dets:
                        d = max(dets, key=lambda z: float((z.bbox[2]-z.bbox[0])*(z.bbox[3]-z.bbox[1])))
                        x1, y1, x2, y2 = [int(v) for v in d.bbox]
                        fw2, fh2 = max(1, x2-x1), max(1, y2-y1)
                        protect = np.zeros(base_img.shape[:2], np.uint8)
                        cv2.ellipse(protect, (x1+fw2//2, y1+fh2//2),
                                    (int(fw2*0.82), int(fh2*1.18)), 0, 0, 360, 255, -1)
                        protect = cv2.GaussianBlur(protect, (41, 41), 0).astype(np.float32)/255.0
                        ai_img = (base_img.astype(np.float32)*protect[...,None]
                                  + ai_img.astype(np.float32)*(1.0-protect[...,None])).astype(np.uint8)
                    Image.fromarray(ai_img).save(str(out_path), "PNG")
                    logger.info("[OK] Uniform swap Qwen-Image-Edit applied: %s", job_id)
            except Exception as e:
                logger.warning(
                    "[UNIFORM_QWEN_EDIT] job=%s failed (%s) — keeping classic composite",
                    job_id, e,
                )

        elif engine == "controlnet":
            try:
                cn = self._get_controlnet_pipeline()
                polished = cn.guided_restyle(
                    str(out_path),
                    prompt=(
                        "professional uniform portrait photo, seamless collar, "
                        "natural fabric texture, consistent studio lighting, photorealistic"
                    ),
                    strength=0.3,
                    controlnet_scale=0.85,
                    job_id=job_id,
                )
                Image.open(polished).convert("RGB").save(str(out_path), "PNG")
                logger.info("[OK] Uniform swap ControlNet polish applied: %s", job_id)
            except Exception as e:
                logger.warning(
                    "[UNIFORM_CONTROLNET_POLISH] job=%s failed (%s) — keeping classic composite",
                    job_id, e,
                )

        return out_path

    def _frame_uniform_portrait(self, image, bg_color: tuple, size: tuple = (1086, 1448)):
        """Place the merged uniform image onto a 3:4 studio portrait canvas."""
        import cv2
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        src = np.array(image.convert("RGB"))
        target_w, target_h = size
        bg_rgb = np.array(bg_color[:3], dtype=np.uint8)
        bg_i16 = bg_rgb.astype(np.int16)

        bg_dist = np.abs(src.astype(np.int16) - bg_i16).max(axis=2)
        content = bg_dist > 18
        if not content.any():
            return image.resize(size, Image.LANCZOS)

        ys, xs = np.where(content)
        pad = 8
        x0 = max(0, xs.min() - pad)
        x1 = min(src.shape[1], xs.max() + pad + 1)
        y0 = max(0, ys.min() - pad)
        y1 = min(src.shape[0], ys.max() + pad + 1)
        crop = src[y0:y1, x0:x1]

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.ndim == 3 and crop.shape[2] == 3 else crop[:, :, :3]
        primary_crop = self._detect_primary_face(crop_bgr)

        if primary_crop:
            fx, fy, fw, fh = primary_crop["bbox"]
            desired_face_top = int(target_h * 0.19)
            face_scale = (target_w * 0.34) / max(fw, 1)
            bottom_scale = (target_h - desired_face_top) / max(crop.shape[0] - fy, 1)
            scale = max(face_scale, bottom_scale)
            paste_x = int(target_w * 0.5 - (fx + fw * 0.5) * scale)
            # Clamp paste_y so the top of the head is never clipped off-canvas
            paste_y = max(0, int(desired_face_top - fy * scale))
        else:
            scale = min(target_w / crop.shape[1], target_h / crop.shape[0])
            paste_x = int((target_w - crop.shape[1] * scale) * 0.5)
            paste_y = int(target_h * 0.08)

        scaled_w = max(1, int(crop.shape[1] * scale))
        scaled_h = max(1, int(crop.shape[0] * scale))
        crop_pil = Image.fromarray(crop).resize((scaled_w, scaled_h), Image.LANCZOS)

        yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
        cx = target_w * 0.5
        cy = target_h * 0.45
        dist = np.sqrt(((xx - cx) / target_w) ** 2 + ((yy - cy) / target_h) ** 2)
        lift = np.clip(1.0 - dist * 1.45, 0.0, 1.0)[:, :, None]
        base = bg_rgb.astype(np.float32)
        light = np.minimum(base + np.array([28, 24, 8], dtype=np.float32), 255)
        bg = (base * (1.0 - lift * 0.28) + light * (lift * 0.28)).astype(np.uint8)
        canvas = Image.fromarray(bg, "RGB")
        canvas.paste(crop_pil, (paste_x, paste_y))
        return canvas

    def _enhance_uniform_portrait(self, image, bg_color: tuple):
        """Clean and polish a merged ID-style uniform portrait."""
        import cv2
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        bg_rgb = tuple(int(c) for c in bg_color[:3])
        rgb = np.array(image.convert("RGB"))
        original_rgb = rgb.copy()
        h, w = rgb.shape[:2]
        bg_arr = np.array(bg_rgb, dtype=np.int16)

        # Preserve the studio background while polishing the subject.
        bg_dist = np.abs(rgb.astype(np.int16) - bg_arr).max(axis=2)
        bg_mask = bg_dist < 58

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Very light cleanup only; aggressive denoise makes faces look painted.
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 2, 2, 5, 11)

        # Local contrast in luminance only, so skin and uniform colors stay natural.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.08, tileGridSize=(8, 8))
        l = clahe.apply(l)
        bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        primary_u = self._detect_primary_face(bgr)
        face_mask = np.zeros((h, w), np.uint8)

        if primary_u:
            fx, fy, fw, fh = primary_u["bbox"]
            center = (fx + fw // 2, fy + int(fh * 0.52))
            axes = (int(fw * 0.62), int(fh * 0.74))
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
            face_mask = cv2.GaussianBlur(face_mask, (31, 31), 0)

            face_soft = cv2.bilateralFilter(bgr, 5, 24, 24)
            alpha = (face_mask.astype(np.float32) / 255.0)[:, :, None] * 0.12
            bgr = (face_soft.astype(np.float32) * alpha + bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        # Sharpen details gently, but reduce the strength on skin.
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
        sharp = cv2.addWeighted(bgr, 1.12, blur, -0.12, 0)
        if face_mask.any():
            face_alpha = (face_mask.astype(np.float32) / 255.0)[:, :, None] * 0.75
            bgr = (bgr.astype(np.float32) * face_alpha + sharp.astype(np.float32) * (1.0 - face_alpha)).astype(np.uint8)
        else:
            bgr = sharp

        # Subtle studio lift without changing the face shape or texture.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int16) + 2, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Soft chin/neck shadow where skin meets collar — adds natural depth.
        if primary_u:
            chin_y = fy + fh
            shadow_top = chin_y + int(fh * 0.08)
            shadow_bot = chin_y + int(fh * 0.30)
            shadow_cx  = fx + fw // 2
            shadow_rx  = int(fw * 0.36)
            if shadow_bot < h and shadow_top < shadow_bot:
                shadow_map = np.zeros((h, w), np.float32)
                cv2.ellipse(shadow_map, (shadow_cx, (shadow_top + shadow_bot) // 2),
                            (shadow_rx, (shadow_bot - shadow_top) // 2), 0, 0, 360, 1.0, -1)
                shadow_map = cv2.GaussianBlur(shadow_map, (0, 0), float(fh) * 0.08)
                shadow_map = np.clip(shadow_map * 0.22, 0.0, 0.22)[:, :, None]
                bgr = np.clip(bgr.astype(np.float32) * (1.0 - shadow_map), 0, 255).astype(np.uint8)

        out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_rgb[bg_mask] = original_rgb[bg_mask]
        return Image.fromarray(out_rgb)


    # ─── Face Swap Helpers ────────────────────────────────────────────────────

    def _do_seamless_face_swap(self, person_bgr, person_face, dst_bgr, dst_face):
        """
        Face swap with progressive fallback:
        1. InsightFace inswapper_128  (best quality, requires model)
        2. MediaPipe eye-aligned warp + skin-tone match + MIXED_CLONE
        3. Improved ellipse + skin-tone match + MIXED_CLONE
        """
        import cv2
        import numpy as np

        fx, fy, fw, fh     = person_face
        uFx, uFy, uFw, uFh = dst_face
        ph, pw = person_bgr.shape[:2]
        uh, uw = dst_bgr.shape[:2]

        # ── 1. InsightFace ────────────────────────────────────────────────
        try:
            result = self._insightface_swap(person_bgr, dst_bgr)
            if result is not None:
                logger.info("[OK] InsightFace inswapper done")
                return result
        except Exception as e:
            logger.info("InsightFace unavailable (%s), using OpenCV path", e)

        # ── Crop regions ──────────────────────────────────────────────────
        t_top  = max(0, uFy - int(uFh * 0.90))
        t_bot  = min(uh, uFy + uFh + int(uFh * 0.45))
        t_left = max(0, uFx - int(uFw * 0.55))
        t_righ = min(uw, uFx + uFw + int(uFw * 0.55))
        tw = max(1, t_righ - t_left)
        th = max(1, t_bot  - t_top)

        s_top  = max(0, fy - int(fh * 0.90))
        s_bot  = min(ph, fy + fh + int(fh * 0.45))
        s_left = max(0, fx - int(fw * 0.55))
        s_righ = min(pw, fx + fw + int(fw * 0.55))
        src_raw  = person_bgr[s_top:s_bot, s_left:s_righ]
        dst_crop = dst_bgr[t_top:t_bot, t_left:t_righ]

        # ── 2. MediaPipe alignment ────────────────────────────────────────
        src_eyes = self._get_mp_eye_centers(src_raw)
        dst_eyes = self._get_mp_eye_centers(dst_crop)

        if src_eyes and dst_eyes:
            src_crop = self._affine_align_face(src_raw, src_eyes, dst_eyes, (tw, th))
            mp_mask = self._get_mp_face_mask(src_crop)
            mask = mp_mask if mp_mask is not None else self._make_ellipse_mask(th, tw)
            logger.info("MediaPipe-aligned face swap (MIXED_CLONE)")
        else:
            src_crop = cv2.resize(src_raw, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            mask = self._make_ellipse_mask(th, tw)
            logger.info("Ellipse fallback face swap (MIXED_CLONE)")

        # ── 3. Skin-tone normalization ────────────────────────────────────
        src_crop = self._match_face_color(src_crop, dst_crop, mask)

        # ── 4. seamlessClone MIXED_CLONE ──────────────────────────────────
        clone_cx = (t_left + t_righ) // 2
        clone_cy = (t_top  + t_bot)  // 2
        half_w, half_h = tw // 2, th // 2
        clone_cx = int(np.clip(clone_cx, half_w + 1, uw - half_w - 1))
        clone_cy = int(np.clip(clone_cy, half_h + 1, uh - half_h - 1))

        return cv2.seamlessClone(
            src_crop, dst_bgr, mask,
            (clone_cx, clone_cy),
            cv2.MIXED_CLONE,
        )

    def _insightface_swap(self, person_bgr, dst_bgr):
        """InsightFace inswapper_128 face swap. Raises if model/library missing."""
        import insightface
        from insightface.app import FaceAnalysis

        # inswapper_128 requires InsightFace ≥ 0.6
        from packaging.version import Version
        if Version(insightface.__version__) < Version("0.6"):
            raise RuntimeError(f"InsightFace {insightface.__version__} too old; need ≥ 0.6")

        model_path = Path("models/insightface/inswapper_128.onnx")
        if not model_path.exists():
            raise FileNotFoundError("inswapper_128.onnx not at models/insightface/")

        app = FaceAnalysis(name="buffalo_l", providers=self._onnx_providers())
        app.prepare(ctx_id=0, det_size=(640, 640))

        src_faces = app.get(person_bgr)
        dst_faces = app.get(dst_bgr)
        if not src_faces or not dst_faces:
            raise ValueError("InsightFace: face not detected in one of the images")

        swapper = insightface.model_zoo.get_model(
            str(model_path), download=False, download_zip=False
        )
        src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        dst_face = max(dst_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        result = dst_bgr.copy()
        result = swapper.get(result, dst_face, src_face, paste_back=True)
        return result

    def _get_mp_eye_centers(self, bgr_img):
        """Return ((lx,ly),(rx,ry)) eye centers via MediaPipe, or None on failure."""
        try:
            import cv2
            import mediapipe as mp
            import numpy as np

            h, w = bgr_img.shape[:2]
            if h < 30 or w < 30:
                return None

            LEFT_EYE  = [33, 133, 159, 145, 153, 144, 160, 161]
            RIGHT_EYE = [362, 263, 386, 374, 380, 373, 387, 388]

            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3
            ) as fm:
                res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    return None
                lms = res.multi_face_landmarks[0].landmark

            def center(ids):
                return (
                    float(np.mean([lms[i].x * w for i in ids])),
                    float(np.mean([lms[i].y * h for i in ids])),
                )
            return center(LEFT_EYE), center(RIGHT_EYE)
        except Exception:
            return None

    def _affine_align_face(self, src, src_eyes, dst_eyes, dst_size):
        """Similarity-warp src so its eye positions match dst_eyes; output at dst_size."""
        import cv2
        import numpy as np

        src_pts = np.float32([src_eyes[0], src_eyes[1]])
        dst_pts = np.float32([dst_eyes[0], dst_eyes[1]])
        M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        if M is None:
            return cv2.resize(src, dst_size, interpolation=cv2.INTER_LANCZOS4)
        dw, dh = dst_size
        return cv2.warpAffine(
            src, M, (dw, dh),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _get_mp_face_mask(self, bgr_img):
        """Face-oval convex-hull mask from MediaPipe 468-point mesh, or None."""
        try:
            import cv2
            import mediapipe as mp
            import numpy as np

            FACE_OVAL = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172,  58, 132,  93, 234, 127, 162,  21,  54, 103,  67, 109,
            ]
            h, w = bgr_img.shape[:2]
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3
            ) as fm:
                res = fm.process(rgb)
                if not res.multi_face_landmarks:
                    return None
                lms = res.multi_face_landmarks[0].landmark

            pts = np.array(
                [[int(lms[i].x * w), int(lms[i].y * h)] for i in FACE_OVAL],
                dtype=np.int32,
            )
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)
            return mask
        except Exception:
            return None

    @staticmethod
    def _make_ellipse_mask(th, tw):
        """Ellipse mask covering face+hair; tighter inset than before."""
        import cv2
        import numpy as np

        mask = np.zeros((th, tw), np.uint8)
        cx_m = tw // 2
        cy_m = int(th * 0.42)
        rx   = max(4, tw // 2 - 6)
        ry   = max(4, int(th * 0.50) - 6)
        cv2.ellipse(mask, (cx_m, cy_m), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    @staticmethod
    def _match_face_color(src, dst, mask):
        """Shift src skin-tone distribution to match dst's; 60 % correction blend."""
        import cv2
        import numpy as np

        fp_src = src[mask > 127].astype(np.float32)
        fp_dst = dst[mask > 127].astype(np.float32)
        if len(fp_src) < 100 or len(fp_dst) < 100:
            return src

        src_mean = fp_src.mean(axis=0)
        src_std  = np.maximum(fp_src.std(axis=0), 1.0)
        dst_mean = fp_dst.mean(axis=0)
        dst_std  = np.maximum(fp_dst.std(axis=0), 1.0)

        corrected = (src.astype(np.float32) - src_mean) / src_std * dst_std + dst_mean
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return cv2.addWeighted(corrected, 0.6, src, 0.4, 0)

    def _add_vignette(self, img, strength: float = 0.2):
        """Add subtle cinematic vignette"""
        import numpy as np
        import cv2

        h, w = img.shape[:2]
        Y = np.linspace(-1, 1, h)[:, None]
        X = np.linspace(-1, 1, w)[None, :]
        mask = 1 - strength * (X**2 + Y**2)
        mask = np.clip(mask, 0, 1)[:, :, None]
        result = (img.astype(np.float32) * mask).astype(np.uint8)
        return result
