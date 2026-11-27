"""Authoritative V11 Uniform Swap Orchestrator.

Integrates and coordinates the modular uniform swap architecture:
1. Ingestion & Normalization (01_original_upload, 02_normalized_source)
2. Semantic Parsing (03_semantic_labels, 04_full_person_alpha)
3. Identity Layer Extraction (05_face_mask, 06_hair_mask, 07_accessory_mask, 08_neck_mask, 09_original_garment_mask)
4. Conditioning Building (10_qwen_image_1_actual, 11_qwen_image_2_actual, 12_edit_mask_actual, 13_hard_protection_mask_actual)
5. Latent Keep Mask Generation (14_latent_keep_mask)
6. Qwen Diffusion & Raw VAE Decoding (15_raw_vae_decoder)
7. Raw Quality Validation (16_raw_quality_validation.json)
8. Quality-Governed Localized Repair (17_localized_repair_mask, 18_final_output)
9. Final Quality Validation (19_final_quality_validation.json)
10. Configuration & Execution Trace Export (20_effective_generation_config.json, 21_pipeline_execution_trace.json)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from dataclasses import dataclass, field
from pipelines.uniform_swap.config import UniformSwapConfig
from pipelines.uniform_swap.semantic_parser import semantic_parser
from pipelines.uniform_swap.identity_extractor import identity_extractor
from pipelines.uniform_swap.conditioning_builder import conditioning_builder
from pipelines.uniform_swap.uniform_generator import uniform_generator
from pipelines.uniform_swap.compositor import uniform_compositor
from pipelines.uniform_swap.quality_validator import quality_validator

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v13.3"


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy types, Paths, and tensors for clean JSON export."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.intc, np.uint8, np.int16, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, Path):
        return str(obj)
    return obj



@dataclass
class DeliveryResult:
    """Structured production delivery outcome matching Task 7 requirements."""
    passed: bool
    status: str  # "completed" | "needs_review" | "rejected"
    delivered_path: Optional[Path] = None
    diagnostic_path: Optional[Path] = None
    defects: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    def __fspath__(self) -> str:
        if self.passed and self.delivered_path is not None:
            return str(self.delivered_path)
        if self.diagnostic_path is not None:
            return str(self.diagnostic_path)
        return ""

    def __str__(self) -> str:
        return self.__fspath__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "delivered_path": str(self.delivered_path) if self.delivered_path else None,
            "diagnostic_path": str(self.diagnostic_path) if self.diagnostic_path else None,
            "defects": self.defects,
            "warnings": self.warnings,
            "validation_report": self.validation_report,
            "trace": self.trace,
        }


def _colorize_schp_labels(labels: np.ndarray) -> np.ndarray:
    """Colorize 20-class SCHP labels for visual inspection."""
    cmap = np.array([
        [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
        [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85],
        [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [51, 170, 221],
        [0, 255, 255], [85, 255, 170], [170, 255, 85], [255, 255, 0], [255, 170, 0]
    ], dtype=np.uint8)
    return cmap[np.clip(labels, 0, len(cmap) - 1)]


class UniformSwapOrchestrator:
    """Authoritative production pipeline orchestrator for uniform swap."""

    def __init__(self) -> None:
        pass

    def run(
        self,
        person_image_input: Union[str, Path, Image.Image],
        uniform_template_input: Union[str, Path, Image.Image],
        config: Optional[UniformSwapConfig] = None,
        job_id: str = "uniform_swap",
        output_dir: Optional[Union[str, Path]] = None,
        debug_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cached_raw_candidate: Optional[Union[str, Path, Image.Image]] = None,
    ) -> Path:
        """Run the complete authoritative V11 uniform swap pipeline."""
        t_start = time.time()
        config = config or UniformSwapConfig()
        config.validate()

        out_path_dir = Path(output_dir or "outputs")
        deb_path_dir = Path(debug_dir or "outputs/debug_phases")
        out_path_dir.mkdir(parents=True, exist_ok=True)
        deb_path_dir.mkdir(parents=True, exist_ok=True)

        trace: Dict[str, Any] = {
            "job_id": job_id,
            "start_time": t_start,
            "stages": [],
            "config": config.to_dict(),
        }

        def record_stage(name: str, duration: float, details: Optional[Dict[str, Any]] = None):
            trace["stages"].append({
                "stage": name,
                "duration_seconds": round(duration, 3),
                "details": details or {},
            })

        logger.info("[ORCHESTRATOR] Starting Uniform Swap (job=%s, mode=%s, steps=%d)", job_id, config.execution_mode, config.steps)

        # ── 1. Ingestion & Normalization ──
        t0 = time.time()
        if isinstance(person_image_input, (str, Path)):
            raw_upload_pil = Image.open(str(person_image_input))
        else:
            raw_upload_pil = person_image_input

        # V12.6 Task 1: Check for transparent / alpha images (e.g. Student 2 Vecteezy PNG)
        # Explicitly composite transparent pixels over pure white (255, 255, 255) BEFORE RGB conversion
        # to completely prevent dirty uninitialized background pixels [32, 48, 48] from entering pipeline.
        if raw_upload_pil.mode in ("RGBA", "LA") or (raw_upload_pil.mode == "P" and "transparency" in raw_upload_pil.info):
            rgba = raw_upload_pil.convert("RGBA")
            bg_pil = Image.new("RGBA", rgba.size, config.background_color + (255,))
            raw_upload_pil = Image.alpha_composite(bg_pil, rgba).convert("RGB")
        else:
            raw_upload_pil = raw_upload_pil.convert("RGB")

        # Save 01_original_upload.png
        p_01 = deb_path_dir / f"{job_id}_01_original_upload.png"
        raw_upload_pil.save(p_01)
        raw_upload_pil.save(deb_path_dir / "01_original_upload.png")

        # Normalize / register to generation resolution with aspect-ratio preservation (Section 3 & 4)
        w, h = config.generation_resolution
        from pipelines.uniform_swap.image_normalizer import (
            normalize_portrait_aspect_ratio,
            create_landmarks_overlay,
            detect_scanned_photo,
            preprocess_scanned_photo,
            detect_harsh_sunlight,
            correct_harsh_sunlight_and_shadows,
            produce_passport_crop,
        )

        raw_upload_pil.save(deb_path_dir / f"{job_id}_01_original_source.png")
        raw_upload_pil.save(deb_path_dir / "01_original_source.png")

        preprocessed_upload_pil = raw_upload_pil
        is_scanned_print = detect_scanned_photo(raw_upload_pil)
        if is_scanned_print:
            logger.info("[ORCHESTRATOR] Low-quality scanned/photographed print detected -> applying conservative preprocessing (Section 4)")
            preprocessed_upload_pil, scan_meta = preprocess_scanned_photo(raw_upload_pil)
            record_stage("scanned_photo_preprocessing", 0.05, scan_meta)

        is_harsh_sunlight = detect_harsh_sunlight(preprocessed_upload_pil)
        if is_harsh_sunlight:
            logger.info("[ORCHESTRATOR] Harsh sunlight / deep shadow detected -> applying conservative illumination correction")
            preprocessed_upload_pil, sun_meta = correct_harsh_sunlight_and_shadows(preprocessed_upload_pil)
            record_stage("sunlight_illumination_correction", 0.05, sun_meta)

        normalized_pil, norm_transform, norm_meta = normalize_portrait_aspect_ratio(
            preprocessed_upload_pil,
            target_size=(w, h),
            background_rgb=config.background_color,
        )
        normalized_rgb = np.asarray(normalized_pil, dtype=np.uint8)

        # Save 02_normalized_source.png
        p_02 = deb_path_dir / f"{job_id}_02_normalized_source.png"
        normalized_pil.save(p_02)
        normalized_pil.save(deb_path_dir / "02_normalized_source.png")

        # Save 02_landmarks_overlay.png (Section 3 requirement)
        landmarks_overlay_pil = create_landmarks_overlay(normalized_pil, norm_meta.get("face_info"))
        landmarks_overlay_pil.save(deb_path_dir / f"{job_id}_02_landmarks_overlay.png")
        landmarks_overlay_pil.save(deb_path_dir / "02_landmarks_overlay.png")

        record_stage("ingestion_and_normalization", time.time() - t0, {
            "dimensions": (w, h),
            "distortion_delta": norm_meta.get("face_info", {}).get("distortion_delta", 0.0) if norm_meta.get("face_info") else 0.0,
        })

        # ── 2. Semantic Parsing ──
        t0 = time.time()
        if progress_callback:
            progress_callback(15, "Performing semantic parsing...")
        parse_data = semantic_parser.parse_source(normalized_rgb, job_id=job_id)

        # Save 03_semantic_labels.png
        schp_color = _colorize_schp_labels(parse_data["schp_labels"])
        Image.fromarray(schp_color).save(deb_path_dir / f"{job_id}_03_semantic_labels.png")
        Image.fromarray(schp_color).save(deb_path_dir / "03_semantic_labels.png")

        # Save 04_full_person_alpha.png
        birefnet_alpha = parse_data.get("birefnet_alpha")
        if birefnet_alpha is not None:
            Image.fromarray(birefnet_alpha).save(deb_path_dir / f"{job_id}_04_full_person_alpha.png")
            Image.fromarray(birefnet_alpha).save(deb_path_dir / "04_full_person_alpha.png")
        else:
            Image.fromarray(np.full((h, w), 255, dtype=np.uint8)).save(deb_path_dir / f"{job_id}_04_full_person_alpha.png")
            Image.fromarray(np.full((h, w), 255, dtype=np.uint8)).save(deb_path_dir / "04_full_person_alpha.png")

        record_stage("semantic_parsing", time.time() - t0)

        # ── 3. Identity Layer Extraction ──
        t0 = time.time()
        if progress_callback:
            progress_callback(25, "Extracting clean identity layer...")
        norm_face_bbox = None
        if norm_meta.get("face_info") and "bbox" in norm_meta["face_info"]:
            norm_face_bbox = tuple(int(round(v)) for v in norm_meta["face_info"]["bbox"])

        id_layer = identity_extractor.extract(
            normalized_rgb,
            face_bbox=norm_face_bbox,
            job_id=job_id,
            debug_dir=deb_path_dir,
            conditioning_strategy=config.conditioning_strategy,
            parse_data=parse_data,
        )

        # Save 05_face_mask.png
        Image.fromarray((id_layer["original_face_mask"].astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_05_face_mask.png")
        Image.fromarray((id_layer["original_face_mask"].astype(np.uint8) * 255)).save(deb_path_dir / "05_face_mask.png")

        # Save 06_hair_mask.png
        Image.fromarray((id_layer["original_hair_mask"].astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_06_hair_mask.png")
        Image.fromarray((id_layer["original_hair_mask"].astype(np.uint8) * 255)).save(deb_path_dir / "06_hair_mask.png")

        # Save 07_accessory_mask.png
        Image.fromarray((id_layer["original_accessory_mask"].astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_07_accessory_mask.png")
        Image.fromarray((id_layer["original_accessory_mask"].astype(np.uint8) * 255)).save(deb_path_dir / "07_accessory_mask.png")

        # Save 08_neck_mask.png
        Image.fromarray((id_layer["original_neck_mask"].astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_08_neck_mask.png")
        Image.fromarray((id_layer["original_neck_mask"].astype(np.uint8) * 255)).save(deb_path_dir / "08_neck_mask.png")

        # Save 09_original_garment_mask.png
        Image.fromarray((id_layer["original_garment_mask"].astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_09_original_garment_mask.png")
        Image.fromarray((id_layer["original_garment_mask"].astype(np.uint8) * 255)).save(deb_path_dir / "09_original_garment_mask.png")

        # Save 03_detected_accessory_objects.json (Section 10 Requirement)
        def _clean_json_val(val):
            if isinstance(val, dict):
                return {k: _clean_json_val(v) for k, v in val.items() if k != "mask"}
            elif isinstance(val, (list, tuple)):
                return [_clean_json_val(v) for v in val]
            elif isinstance(val, np.ndarray):
                return val.tolist() if val.size < 50 else f"<ndarray {val.shape}>"
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, (np.bool_, bool)):
                return bool(val)
            return val

        raw_acc_meta = {
            "has_confirmed_bows": id_layer.get("has_confirmed_bows", False),
            "has_accessories": id_layer.get("has_accessories", False),
            "accessory_objects": id_layer.get("accessory_objects", []),
            "individual_bows": id_layer.get("individual_bows", []),
        }
        acc_meta = _clean_json_val(raw_acc_meta)
        with open(deb_path_dir / f"{job_id}_03_detected_accessory_objects.json", "w", encoding="utf-8") as f:
            json.dump(acc_meta, f, indent=2)
        with open(deb_path_dir / "03_detected_accessory_objects.json", "w", encoding="utf-8") as f:
            json.dump(acc_meta, f, indent=2)

        # Save 04_protected_region_mask.png (Section 10 Requirement)
        prot_mask = (
            id_layer.get("original_face_mask", np.zeros((h, w), dtype=bool)) |
            id_layer.get("original_ears_mask", np.zeros((h, w), dtype=bool)) |
            id_layer.get("original_hair_mask", np.zeros((h, w), dtype=bool)) |
            id_layer.get("original_accessory_mask", np.zeros((h, w), dtype=bool))
        )
        Image.fromarray((prot_mask.astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_04_protected_region_mask.png")
        Image.fromarray((prot_mask.astype(np.uint8) * 255)).save(deb_path_dir / "04_protected_region_mask.png")

        record_stage("identity_extraction", time.time() - t0, {
            "neck_classification": id_layer.get("neck_classification"),
            "accessory_objects_count": len(id_layer.get("accessory_objects", [])),
        })

        # ── 4. Conditioning Building (Qwen Image 1 & Image 2) ──
        t0 = time.time()
        if progress_callback:
            progress_callback(35, "Preparing conditioning canvases...")

        # Qwen Image 2 (Uniform Reference)
        uniform_ref_pil, uniform_ref_arr = conditioning_builder.prepare_uniform_reference(
            uniform_template_input,
            target_size=(w, h),
            background_rgb=config.background_color,
        )
        from pipelines.uniform_swap.reference_analyzer import reference_analyzer
        uniform_cues = reference_analyzer.analyze(uniform_template_input)
        id_layer["cues"] = uniform_cues
        uniform_ref_pil.save(deb_path_dir / f"{job_id}_11_qwen_image_2_actual.png")
        uniform_ref_pil.save(deb_path_dir / "11_qwen_image_2_actual.png")

        # Qwen Image 1 & Edit/Protection Masks
        cond_res = conditioning_builder.build_conditioning(
            normalized_rgb,
            identity_layer=id_layer,
            config=config,
        )
        cond_pil = cond_res["conditioning_pil"]
        cond_pil.save(deb_path_dir / f"{job_id}_10_qwen_image_1_actual.png")
        cond_pil.save(deb_path_dir / "10_qwen_image_1_actual.png")

        edit_mask = cond_res["generation_edit_region"]
        hard_protect = cond_res["generation_hard_identity_protection"]
        Image.fromarray((edit_mask.astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_12_edit_mask_actual.png")
        Image.fromarray((edit_mask.astype(np.uint8) * 255)).save(deb_path_dir / "12_edit_mask_actual.png")

        Image.fromarray((hard_protect.astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_13_hard_protection_mask_actual.png")
        Image.fromarray((hard_protect.astype(np.uint8) * 255)).save(deb_path_dir / "13_hard_protection_mask_actual.png")

        # Save Section 10 Canonical Artifacts
        Image.fromarray((edit_mask.astype(np.uint8) * 255)).save(deb_path_dir / f"{job_id}_05_editable_garment_mask.png")
        Image.fromarray((edit_mask.astype(np.uint8) * 255)).save(deb_path_dir / "05_editable_garment_mask.png")

        cond_pil.save(deb_path_dir / f"{job_id}_08_qwen_image1.png")
        cond_pil.save(deb_path_dir / "08_qwen_image1.png")

        uniform_ref_pil.save(deb_path_dir / f"{job_id}_09_qwen_image2.png")
        uniform_ref_pil.save(deb_path_dir / "09_qwen_image2.png")

        record_stage("conditioning_building", time.time() - t0, {
            "strategy": cond_res["selected_strategy"],
            "edit_pixels": int(np.sum(edit_mask)),
            "protect_pixels": int(np.sum(hard_protect)),
        })

        # ── 5. Latent Keep Mask (14_latent_keep_mask.png) ──
        latent_h, latent_w = h // 16, w // 16
        latent_keep_visual = np.zeros((h, w), dtype=np.uint8)
        if config.latent_identity_lock != "off" and edit_mask is not None:
            keep_bool = ~edit_mask.astype(bool)
            if config.latent_identity_lock == "soft":
                keep_soft = cv2.GaussianBlur(keep_bool.astype(np.float32), (0, 0), sigmaX=16.0, sigmaY=16.0)
            else:
                keep_soft = keep_bool.astype(np.float32)
                if hard_protect is not None:
                    keep_soft = np.maximum(keep_soft, hard_protect.astype(np.float32))
            keep_latent = cv2.resize(keep_soft, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
            latent_keep_visual = (cv2.resize(keep_latent, (w, h), interpolation=cv2.INTER_NEAREST) * 255.0).clip(0, 255).astype(np.uint8)

        Image.fromarray(latent_keep_visual).save(deb_path_dir / f"{job_id}_14_latent_keep_mask.png")
        Image.fromarray(latent_keep_visual).save(deb_path_dir / "14_latent_keep_mask.png")

        # ── 6. Generative Diffusion & Raw VAE Decode ──
        # Section 2 Requirement: Generate uniform once. Re-use existing foreground
        # when only background color changes without calling Qwen again.
        t0 = time.time()
        raw_vae_img = None
        raw_native_img = None

        from pipelines.uniform_swap.cache_manager import cache_manager, compute_sha256

        ref_sha256 = compute_sha256(uniform_template_input)
        logger.info("[ORCHESTRATOR] Target uniform reference: path=%s, sha256=%s, type=%s", uniform_template_input, ref_sha256, uniform_cues.get("ref_type"))

        # Pre-build prompts for cache key construction
        pos_p, neg_p = uniform_generator.build_prompts(
            {},
            neck_cls=id_layer.get("neck_classification", "partially_visible_neck"),
            background_rgb=config.background_color,
            uniform_reference_image=uniform_ref_pil,
            identity_layer=id_layer,
            badge_policy=config.badge_policy,
            uniform_path=uniform_template_input,
        )
        effective_pos = config.positive_prompt or pos_p
        effective_neg = config.negative_prompt or neg_p

        cache_key, cache_meta = cache_manager.generate_cache_key(
            source_image=normalized_rgb,
            uniform_reference_image=uniform_template_input,
            positive_prompt=effective_pos,
            negative_prompt=effective_neg,
            inference_params={
                "steps": config.steps,
                "guidance_scale": getattr(config, "true_cfg_scale", 1.20),
                "resolution": (w, h),
                "latent_identity_lock": config.latent_identity_lock,
            },
        )

        cache_hit = False
        fresh_inference = False
        cache_lookup_t0 = time.time()

        # Enforce V13.2 cache policy: When policy is force_regenerate, strictly ignore
        # explicit cached_raw_candidate and authoritative cache, record cache_hit=False, fresh_inference=True
        if config.raw_diffusion_cache_policy == "force_regenerate":
            cached_raw_candidate = None
            raw_vae_img = None
            cache_hit = False
            fresh_inference = True
            logger.info(
                "[ORCHESTRATOR] Cache policy is 'force_regenerate': strictly ignoring explicit cached_raw_candidate and cache lookups. Forcing fresh generative inference."
            )
        else:
            # Check explicit parameter with strict provenance verification
            if cached_raw_candidate is not None:
                diagnostic_mode = getattr(config, "diagnostic_mode", False)
                is_valid = cache_manager.validate_external_candidate(
                    candidate_image=cached_raw_candidate,
                    expected_uniform_ref=uniform_template_input,
                    expected_ref_type=uniform_cues.get("ref_type", "generic_uniform"),
                    diagnostic_mode=diagnostic_mode,
                )
                if is_valid:
                    try:
                        if isinstance(cached_raw_candidate, (str, Path)):
                            raw_vae_img = Image.open(str(cached_raw_candidate)).convert("RGB")
                        else:
                            raw_vae_img = cached_raw_candidate.convert("RGB")
                        raw_native_img = raw_vae_img
                        cache_hit = True
                        logger.info(
                            "[ORCHESTRATOR] Using validated explicit raw candidate with verified provenance for reference %s",
                            ref_sha256[:12]
                        )
                    except Exception as e:
                        logger.warning("[ORCHESTRATOR] Could not load explicit cached candidate: %s", e)
                else:
                    logger.warning(
                        "[ORCHESTRATOR] Incompatible or unverified explicit raw candidate %s rejected in production mode. "
                        "Forcing fresh Qwen generation to guarantee reference and provenance integrity.",
                        cached_raw_candidate
                    )

            # Check authoritative cache manager
            if raw_vae_img is None:
                cached_entry = cache_manager.get(cache_key, expected_ref_sha256=ref_sha256)
                if cached_entry is not None:
                    raw_vae_img, meta = cached_entry
                    raw_native_img = raw_vae_img
                    cache_hit = True
                    logger.info("[ORCHESTRATOR] Authoritative cache HIT for key %s (ref_sha256=%s)", cache_key, ref_sha256[:12])

        cache_lookup_dur = time.time() - cache_lookup_t0
        record_stage("cache_lookup", cache_lookup_dur, {
            "cache_hit": cache_hit,
            "fresh_inference": (raw_vae_img is None),
            "cache_key": cache_key,
            "ref_sha256": ref_sha256,
        })

        qwen_fresh_dur = 0.0
        if raw_vae_img is None:
            if progress_callback:
                progress_callback(45, "Running Qwen generative diffusion...")

            t_qwen_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            from pipelines.qwen_edit_pipeline import qwen_service
            gen_res = uniform_generator.generate(
                qwen_pipeline_instance=qwen_service,
                conditioning_image=cond_pil,
                uniform_reference_image=uniform_ref_pil,
                config=config,
                edit_region=edit_mask,
                hard_protect=hard_protect,
                neck_cls=id_layer.get("neck_classification", "partially_visible_neck"),
                identity_layer=id_layer,
                job_id=job_id,
                debug_dir=deb_path_dir,
                progress_callback=progress_callback,
                uniform_path=uniform_template_input,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            qwen_fresh_dur = time.time() - t_qwen_start

            raw_vae_img = gen_res["raw_vae_decoder_img"]
            raw_native_img = gen_res.get("raw_vae_decoder_native_img") or raw_vae_img

            # Store in authoritative cache with full V13.2 cryptographic provenance manifest
            try:
                from pipelines.uniform_swap.cache_manager import create_provenance_manifest
                prov_manifest = create_provenance_manifest(
                    original_source_image=raw_upload_pil,
                    normalized_source_image=normalized_pil,
                    uniform_ref=uniform_ref_pil,
                    conditioning_image=cond_pil,
                    effective_prompt=effective_pos,
                    negative_prompt=effective_neg,
                    model_version=PIPELINE_VERSION,
                    lora_version="qwen_image_edit_lightning_4step",
                    inference_resolution=(w, h),
                    seed=getattr(config, "seed", 42),
                    steps=config.steps,
                    guidance_parameters={
                        "cfg": config.true_cfg_scale,
                        "latent_identity_lock": config.latent_identity_lock,
                    },
                    candidate_image=raw_vae_img,
                )
                cache_meta.update(prov_manifest)
                cache_manager.put(cache_key, raw_vae_img, cache_meta)
            except Exception as e:
                logger.warning("[ORCHESTRATOR] Could not store authoritative cache %s: %s", cache_key, e)

            record_stage("fresh_qwen_diffusion", qwen_fresh_dur, {
                "ref_sha256": ref_sha256,
                "cache_key": cache_key,
                "fresh_inference": True,
                "cache_hit": False,
            })

            # Save separate diagnostic files (Section 7)
            raw_native_img.save(deb_path_dir / f"{job_id}_raw_vae_decoder_native.png")
            raw_native_img.save(deb_path_dir / "raw_vae_decoder_native.png")

            raw_vae_img.save(deb_path_dir / f"{job_id}_raw_candidate_resized.png")
            raw_vae_img.save(deb_path_dir / "raw_candidate_resized.png")

            raw_vae_img.save(deb_path_dir / f"{job_id}_15_raw_vae_decoder.png")
            raw_vae_img.save(deb_path_dir / "15_raw_vae_decoder.png")

            record_stage("qwen_diffusion_and_raw_decode", time.time() - t0, {
                "steps": config.steps,
                "cfg": config.true_cfg_scale,
                "latent_lock": config.latent_identity_lock,
                "reused": False,
            })
        else:
            record_stage("qwen_diffusion_and_raw_decode", time.time() - t0, {
                "reused": True,
                "cache_key": cache_key,
            })

        # ── 6. Raw Quality Validation ──
        t0 = time.time()
        raw_quality = quality_validator.validate_candidate(
            raw_vae_img,
            normalized_rgb,
            identity_layer=id_layer,
            uniform_reference=uniform_ref_pil,
            job_id=job_id,
            stage="raw",
            uniform_path=uniform_template_input,
        )
        quality_validator.save_validation_json(raw_quality, deb_path_dir / f"{job_id}_16_raw_quality_validation.json")
        quality_validator.save_validation_json(raw_quality, deb_path_dir / "16_raw_quality_validation.json")
        record_stage("raw_quality_validation", time.time() - t0, raw_quality)

        # ── 7. Quality-Governed Localized Repair & Compositing ──
        t0 = time.time()
        if progress_callback:
            progress_callback(85, "Executing adaptive compositing...")

        if config.rgb_identity_restoration != "off":
            comp_res = uniform_compositor.composite(
                raw_vae_img=raw_vae_img,
                normalized_source_img=normalized_pil,
                identity_layer=id_layer,
                raw_quality=raw_quality,
                config=config,
                job_id=job_id,
                debug_dir=deb_path_dir,
            )
            repaired_img = comp_res["composite_img"]
            repair_actions = comp_res.get("actions", [])

            # V12.6 Task 2: Incompatible Collar Registration Handling
            # If generated candidate collar registration is incompatible (faces != 1 or rmse > 0.040),
            # never force unaligned source pixels onto the candidate. Keep candidate for gating & review.
            incompatible_geometry_rejected = "rejected_incompatible_collar_geometry" in repair_actions

            if config.rgb_identity_restoration in ("adaptive", "protected_source"):
                final_img = repaired_img
                # Explicit protected-region pixel preservation measurement (Section 6)
                protected_total = (
                    id_layer.get("original_face_mask", np.zeros((h, w), dtype=bool)) |
                    id_layer.get("original_ears_mask", np.zeros((h, w), dtype=bool)) |
                    id_layer.get("original_hair_mask", np.zeros((h, w), dtype=bool)) |
                    id_layer.get("original_accessory_mask", np.zeros((h, w), dtype=bool))
                )
                if np.any(protected_total) and not incompatible_geometry_rejected:
                    diff_arr = np.abs(np.asarray(repaired_img, dtype=float) - np.asarray(normalized_pil, dtype=float))
                    prot_delta = float(np.mean(diff_arr[protected_total]))
                    logger.info("[ORCHESTRATOR] Protected region pixel preservation delta: %.4f (threshold: <= 2.0)", prot_delta)
            else:
                # Legacy / fallback mode
                if quality_validator.compare_candidates(raw_quality, raw_quality):
                    final_img = repaired_img
                else:
                    final_img = raw_vae_img
                    repair_actions.append("discarded_repair_worse_than_raw")
        else:
            final_img = raw_vae_img
            repair_actions = ["bypassed_rgb_restoration_off"]
            incompatible_geometry_rejected = False

        # ── 7b. Coloured edge decontamination (V13.3.2 §5) ──
        # PA_11 was delivered with a blue fringe along the hair crown. Measured
        # on its artifacts: the raw Qwen output had 0 backdrop-coloured pixels
        # and the assembled foreground had 841, all within 25px of the subject
        # edge and exactly the source photo's backdrop blue (0,129,253). The
        # spill is source backdrop carried through matting, so it is removed
        # here, after compositing, using the measured source backdrop colour.
        edge_decon_info: Dict[str, Any] = {}
        try:
            from pipelines.uniform_swap.edge_decontamination import (
                decontaminate_edges, detect_colored_edge_halo, estimate_source_background_color,
            )

            _rgba_src = comp_res.get("composite_rgba") if 'comp_res' in locals() and comp_res else None
            if _rgba_src is None and final_img.mode == "RGBA":
                _rgba_src = final_img
            if _rgba_src is not None:
                _arr = np.asarray(_rgba_src.convert("RGBA"))
                _src_bg, _bg_conf = estimate_source_background_color(
                    np.asarray(Image.open(person_image_input).convert("RGB"))
                    if isinstance(person_image_input, (str, Path)) else normalized_rgb
                )
                _hair_m = id_layer.get("original_hair_mask") if 'id_layer' in locals() and id_layer else None
                _det = detect_colored_edge_halo(
                    _arr[:, :, :3], _arr[:, :, 3], _src_bg,
                    output_background_rgb=config.background_color,
                    hair_mask=_hair_m,
                )
                edge_decon_info = {"detection": {k: v for k, v in _det.items() if k != "mask"},
                                   "source_background_rgb": _src_bg,
                                   "source_background_confidence": _bg_conf}
                if _det["status"] == "FAIL" and _bg_conf >= 0.40:
                    _frgb, _falpha, _info = decontaminate_edges(
                        _arr[:, :, :3], _arr[:, :, 3], _src_bg, _det["mask"], hair_mask=_hair_m)
                    _fixed = Image.fromarray(np.dstack([_frgb, _falpha]), "RGBA")
                    if 'comp_res' in locals() and comp_res:
                        comp_res["composite_rgba"] = _fixed
                    _bg = np.array(config.background_color, dtype=np.float32)
                    _a = (_falpha.astype(np.float32) / 255.0)[:, :, None]
                    final_img = Image.fromarray(
                        np.clip(_frgb.astype(np.float32) * _a + _bg * (1 - _a), 0, 255).astype(np.uint8))
                    repair_actions.extend(_info.get("actions", []))
                    edge_decon_info["repair"] = {k: v for k, v in _info.items()}
                    logger.info("[ORCHESTRATOR] Coloured edge decontamination applied: %s",
                                _info.get("actions"))
        except Exception as e:  # never let a cosmetic repair break delivery
            logger.warning("[ORCHESTRATOR] Edge decontamination skipped: %s", e)
            edge_decon_info = {"error": str(e)}
        record_stage("edge_decontamination", 0.0, edge_decon_info)

        # Save separate diagnostic files (Section 7 & 9)
        final_candidate_path = deb_path_dir / f"{job_id}_final_candidate.png"
        final_img.save(final_candidate_path)
        final_img.save(deb_path_dir / "final_candidate.png")

        final_img.save(deb_path_dir / f"{job_id}_18_final_output.png")
        final_img.save(deb_path_dir / "18_final_output.png")
        final_img.save(deb_path_dir / f"{job_id}_17_protected_source_composite.png")
        final_img.save(deb_path_dir / "17_protected_source_composite.png")

        # Save 09_white_background_composite.png (V12.6 Task 1 & Deliverables requirement)
        final_img.save(deb_path_dir / f"{job_id}_09_white_background_composite.png")
        final_img.save(deb_path_dir / "09_white_background_composite.png")

        # ── 8. Two-Stage Validation: Stage A (Pre-Crop Validation) ──
        # V12.5 Task 2: Validate identity, protected-region integrity, uniform fidelity,
        # and accessories on uncropped candidate BEFORE cropping.
        t0 = time.time()
        pre_crop_quality = quality_validator.validate_pre_crop(
            candidate_image=final_img,
            source_image=normalized_rgb,
            identity_layer=id_layer,
            uniform_reference=uniform_ref_pil,
            job_id=job_id,
            background_rgb=config.background_color,
            uniform_path=uniform_template_input,
        )
        if incompatible_geometry_rejected:
            pre_crop_quality["passed"] = False
            pre_crop_quality["needs_review"] = True
            pre_crop_quality["defects"].append("incompatible_collar_registration_geometry")

        # V13.3 Intelligent Defect Recovery Controller
        if not pre_crop_quality.get("passed", False) and config.rgb_identity_restoration != "off" and 'comp_res' in locals():
            logger.info("[ORCHESTRATOR] Pre-crop quality failed with defects: %s. Initiating V13.3 RecoveryController...", pre_crop_quality.get("defects"))
            from pipelines.uniform_swap.recovery_controller import recovery_controller
            rec_res = recovery_controller.attempt_recovery(
                current_candidate_rgba=comp_res.get("composite_rgba", final_img.convert("RGBA")),
                normalized_source_img=normalized_pil,
                identity_layer=id_layer,
                validation_report=pre_crop_quality,
                reference_spec=uniform_cues,
                config=config,
                validator_fn=lambda c_rgb, s_rgb, id_l: quality_validator.validate_pre_crop(
                    candidate_image=c_rgb,
                    source_image=s_rgb,
                    identity_layer=id_l,
                    uniform_reference=uniform_ref_pil,
                    job_id=job_id,
                    background_rgb=config.background_color,
                    uniform_path=uniform_template_input,
                ),
                job_id=job_id,
                debug_dir=deb_path_dir,
            )
            if rec_res["defects_after"] < rec_res["defects_before"]:
                comp_res["composite_rgba"] = rec_res["recovered_rgba"]
                pre_crop_quality = rec_res["validation_report"]
                logger.info("[ORCHESTRATOR] RecoveryController improved candidate: defects %d -> %d (%s)", rec_res["defects_before"], rec_res["defects_after"], rec_res["actions_taken"])
                repair_actions.extend(rec_res["actions_taken"])

        quality_validator.save_validation_json(pre_crop_quality, deb_path_dir / f"{job_id}_16_pre_crop_quality_validation.json")
        quality_validator.save_validation_json(pre_crop_quality, deb_path_dir / "16_pre_crop_quality_validation.json")
        record_stage("pre_crop_quality_validation", time.time() - t0, pre_crop_quality)

        # ── 9. Standardized Passport Crop Framing (V12.5 Task 3 & V13.1 Authoritative RGBA) ──
        t0 = time.time()
        crop_input = comp_res.get("composite_rgba", final_img) if (config.rgb_identity_restoration != "off" and 'comp_res' in locals() and comp_res) else final_img
        passport_crop_pil, crop_meta = produce_passport_crop(
            crop_input,
            face_info=norm_meta.get("face_info"),
            identity_layer=id_layer,
            output_size=config.output_resolution,
            target_head_occupancy=0.58,
            target_headroom_ratio=0.10,
            background_rgb=config.background_color,
            debug_dir=deb_path_dir,
            job_id=job_id,
            return_meta=True,
        )
        crop_meta["uniform_type"] = uniform_cues.get("ref_type", "red_collar")
        crop_meta["uniform_cues"] = {k: v for k, v in uniform_cues.items() if isinstance(v, (str, int, float, bool))}
        passport_crop_path = deb_path_dir / f"{job_id}_19_passport_crop.png"
        passport_crop_pil.save(passport_crop_path)
        passport_crop_pil.save(deb_path_dir / "19_passport_crop.png")
        passport_crop_pil.save(deb_path_dir / f"{job_id}_20_final_output.png")
        passport_crop_pil.save(deb_path_dir / "20_final_output.png")

        # Save crop metadata JSON (filter out non-serializable ndarrays)
        clean_crop_meta = {k: v for k, v in crop_meta.items() if not isinstance(v, np.ndarray)}
        crop_meta_path = deb_path_dir / f"{job_id}_19_passport_crop_meta.json"
        with open(crop_meta_path, "w", encoding="utf-8") as f:
            json.dump(clean_crop_meta, f, indent=2)
        with open(deb_path_dir / "19_passport_crop_meta.json", "w", encoding="utf-8") as f:
            json.dump(clean_crop_meta, f, indent=2)
        record_stage("passport_framing", time.time() - t0, clean_crop_meta)

        # ── 10. Two-Stage Validation: Stage B (Post-Crop Validation) ──
        # V12.5 Task 2: Validate complete head/chin visibility, headroom, visible collar,
        # visible bows without clipping, exact dimensions, and non-distortion.
        t0 = time.time()
        post_crop_quality = quality_validator.validate_post_crop(
            cropped_image=passport_crop_pil,
            crop_meta=crop_meta,
            expected_size=config.output_resolution,
            has_confirmed_bows=bool(id_layer.get("has_confirmed_bows")),
            expected_bg_rgb=config.background_color,
            job_id=job_id,
        )
        quality_validator.save_validation_json(post_crop_quality, deb_path_dir / f"{job_id}_20_post_crop_quality_validation.json")
        quality_validator.save_validation_json(post_crop_quality, deb_path_dir / "20_post_crop_quality_validation.json")
        record_stage("post_crop_quality_validation", time.time() - t0, post_crop_quality)

        # Consolidated final quality report (backwards-compatible with 19_final_quality_validation.json)
        pre_passed = bool(pre_crop_quality.get("passed", False))
        post_passed = bool(post_crop_quality.get("passed", False))
        passed = pre_passed and post_passed

        combined_defects = list(pre_crop_quality.get("defects", [])) + list(post_crop_quality.get("defects", []))
        combined_warnings = list(pre_crop_quality.get("warnings", [])) + list(post_crop_quality.get("warnings", []))

        consolidated_quality = dict(pre_crop_quality)
        consolidated_quality["stage"] = "final_two_stage"
        consolidated_quality["passed"] = passed
        consolidated_quality["pre_crop_passed"] = pre_passed
        consolidated_quality["post_crop_passed"] = post_passed
        consolidated_quality["needs_review"] = not passed or bool(pre_crop_quality.get("needs_review")) or bool(post_crop_quality.get("needs_review"))
        consolidated_quality["defects"] = combined_defects
        consolidated_quality["warnings"] = combined_warnings
        consolidated_quality["post_crop_checks"] = post_crop_quality.get("checks", {})
        consolidated_quality["crop_metadata"] = clean_crop_meta

        quality_validator.save_validation_json(consolidated_quality, deb_path_dir / f"{job_id}_19_final_quality_validation.json")
        quality_validator.save_validation_json(consolidated_quality, deb_path_dir / "19_final_quality_validation.json")
        quality_validator.save_validation_json(consolidated_quality, deb_path_dir / f"{job_id}_20_final_quality_validation.json")
        quality_validator.save_validation_json(consolidated_quality, deb_path_dir / "20_final_quality_validation.json")

        defects = combined_defects
        warnings = combined_warnings

        delivered_output_path = out_path_dir / f"{job_id}_uniform.png"
        approved_output_path = deb_path_dir / f"{job_id}_approved_output.png"
        actual_delivered_path: Optional[Path] = None

        # V12.5 Task 9: Strict Diagnostic vs Production Output Separation
        # Failed diagnostic and raw-only candidates MUST NOT be written using the production delivery filename.
        if passed:
            passport_crop_pil.save(approved_output_path)
            passport_crop_pil.save(deb_path_dir / "approved_output.png")
            passport_crop_pil.save(delivered_output_path)
            actual_delivered_path = delivered_output_path
            job_status = "completed"
            logger.info("[ORCHESTRATOR] Candidate PASSED both validation stages -> delivered production output: %s", delivered_output_path)
        else:
            job_status = "needs_review"
            actual_delivered_path = None
            # Explicitly ensure no stale production output remains from earlier runs
            if delivered_output_path.exists():
                try:
                    delivered_output_path.unlink()
                except OSError:
                    pass
            # Save diagnostic file with explicit diagnostic naming in debug directory only
            diag_output_path = deb_path_dir / f"{job_id}_diagnostic_crop.png"
            passport_crop_pil.save(diag_output_path)
            passport_crop_pil.save(deb_path_dir / "diagnostic_crop.png")
            logger.warning("[ORCHESTRATOR] Candidate FAILED validation (defects=%s) -> production delivery gated (no approved output written)", defects)

        # ── 9. Export Config & Execution Trace ──
        # 20_effective_generation_config.json
        cfg_dict = config.to_dict()
        cfg_dict["input_person"] = str(person_image_input)
        cfg_dict["input_uniform"] = str(uniform_template_input)
        cfg_dict["delivery_status"] = job_status
        cfg_dict["passed"] = passed
        cfg_dict["defects"] = defects
        cfg_dict["warnings"] = warnings
        clean_cfg = _sanitize_for_json(cfg_dict)
        with open(deb_path_dir / f"{job_id}_20_effective_generation_config.json", "w", encoding="utf-8") as f:
            json.dump(clean_cfg, f, indent=2)
        with open(deb_path_dir / "20_effective_generation_config.json", "w", encoding="utf-8") as f:
            json.dump(clean_cfg, f, indent=2)

        # 21_pipeline_execution_trace.json
        trace["total_duration_seconds"] = round(time.time() - t_start, 3)
        trace["delivery_status"] = job_status
        trace["passed"] = passed
        trace["defects"] = defects
        if torch.cuda.is_available():
            trace["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        else:
            trace["peak_vram_mb"] = 0.0

        clean_trace = _sanitize_for_json(trace)
        with open(deb_path_dir / f"{job_id}_21_pipeline_execution_trace.json", "w", encoding="utf-8") as f:
            json.dump(clean_trace, f, indent=2)
        with open(deb_path_dir / "21_pipeline_execution_trace.json", "w", encoding="utf-8") as f:
            json.dump(clean_trace, f, indent=2)

        # 21_final_production_decision.json (Section 10 Requirement)
        decision_record = {
            "job_id": job_id,
            "passed": passed,
            "delivery_status": job_status,
            "delivered_path": str(actual_delivered_path) if actual_delivered_path else None,
            "diagnostic_path": str(final_candidate_path),
            "defects": defects,
            "warnings": warnings,
            "pre_crop_passed": pre_passed,
            "post_crop_passed": post_passed,
            "similarity": consolidated_quality.get("similarity", 0.0),
            "geometry_rmse": consolidated_quality.get("geometry_rmse", 0.0),
        }
        clean_decision = _sanitize_for_json(decision_record)
        with open(deb_path_dir / f"{job_id}_21_final_production_decision.json", "w", encoding="utf-8") as f:
            json.dump(clean_decision, f, indent=2)
        with open(deb_path_dir / "21_final_production_decision.json", "w", encoding="utf-8") as f:
            json.dump(clean_decision, f, indent=2)

        delivery_result = DeliveryResult(
            passed=passed,
            status=job_status,
            delivered_path=actual_delivered_path,
            diagnostic_path=final_candidate_path,
            defects=defects,
            warnings=warnings,
            validation_report=consolidated_quality,
            trace=trace,
        )

        logger.info(
            "[ORCHESTRATOR] Uniform Swap finished in %.2fs (job=%s, status=%s, final_sim=%.3f, delivered=%s)",
            trace["total_duration_seconds"], job_id, job_status, consolidated_quality.get("similarity", 0.0),
            str(actual_delivered_path) if actual_delivered_path else "None (flagged for review)"
        )
        return delivery_result

    # Alias for run()
    swap_uniform = run


# Global singleton orchestrator
uniform_swap_orchestrator = UniformSwapOrchestrator()
