"""Short-lived Qwen-VL analysis for constrained uniform fitting."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/qwen2.5-vl-3b-instruct")
# Bump whenever the visual-analysis prompts or normalization rules change.
# Cached observations are generation inputs, so retaining an old "vest" or
# "bow" interpretation after a prompt correction silently recreates the same
# incorrect Qwen edit for identical uploads.
VL_CACHE_VERSION = "content-v34-literal-plaits-and-multiple-hair-attachments"


def make_hair_inspection_crop(image: Image.Image) -> Image.Image:
    """Return a face-centred crop that still contains the complete visible hair.

    The old analysis cropped only by image height, leaving most of the backdrop
    and uniform in the supposedly focused hair pass.  Small clips then occupied
    too few visual tokens for reliable inspection.  Haar is used only to place
    the crop; Qwen-VL remains the authority for all semantic hair findings.
    """
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    height, width = array.shape[:2]
    face_box: Optional[Tuple[int, int, int, int]] = None
    try:
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(
            cv2.cvtColor(array, cv2.COLOR_RGB2GRAY),
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(max(28, width // 14), max(28, height // 14)),
        )
        if len(faces):
            # Prefer the largest central face. Background faces or printed
            # portraits near an edge must not steer the hair crop.
            def face_rank(box: np.ndarray) -> float:
                x, y, w, h = (float(value) for value in box)
                centre_penalty = abs((x + w / 2.0) - width / 2.0) / max(1.0, width)
                return w * h * (1.0 - min(0.7, centre_penalty))

            face_box = tuple(int(value) for value in max(faces, key=face_rank))
    except Exception as exc:
        logger.debug("[UniformVL] Face-centred hair crop fallback: %s", exc)

    if face_box is None:
        # Conservative portrait fallback: remove only narrow side margins and
        # lower torso. It preserves crown, both ears, long hair and shoulders.
        return rgb.crop((
            int(width * 0.04), 0,
            max(1, int(width * 0.96)), max(1, int(height * 0.84)),
        ))

    x, y, face_w, face_h = face_box
    centre_x = x + face_w / 2.0
    crop_left = max(0, int(centre_x - face_w * 1.65))
    crop_right = min(width, int(centre_x + face_w * 1.65))
    crop_top = max(0, int(y - face_h * 1.15))
    crop_bottom = min(height, int(y + face_h * 2.35))
    # Avoid a very narrow crop when the detector sees only the inner face;
    # clips and long side hair commonly sit well beyond that box.
    # Hair flowers, bows and tied ends often sit at the extreme portrait
    # edges.  A 72%-wide face crop kept the eyes/face crisp but could omit the
    # dimmer accessory on one side, causing VL to report a bilateral pair as
    # one object.  Retain nearly the full source width for the focused hair
    # pass; the lower torso is still removed below.
    minimum_width = int(width * 0.96)
    if crop_right - crop_left < minimum_width:
        crop_left = max(0, int(centre_x - minimum_width / 2))
        crop_right = min(width, crop_left + minimum_width)
        crop_left = max(0, crop_right - minimum_width)
    return rgb.crop((crop_left, crop_top, crop_right, crop_bottom))


def make_face_inspection_crop(image: Image.Image) -> Image.Image:
    """Return a tight face/ears crop for identity-trait analysis.

    Wide classroom photographs reduce the face to too few VL visual tokens.
    The full portrait remains available for pose and body context; this crop
    supplies facial maturity, complexion, eyewear reflection and both-ear
    jewelry evidence without asking the model to infer them from clothing.
    """
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    height, width = array.shape[:2]
    try:
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(
            cv2.cvtColor(array, cv2.COLOR_RGB2GRAY),
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(max(28, width // 14), max(28, height // 14)),
        )
        if len(faces):
            def face_rank(box: np.ndarray) -> float:
                x, y, face_w, face_h = (float(value) for value in box)
                centre_penalty = abs((x + face_w / 2.0) - width / 2.0) / max(1.0, width)
                return face_w * face_h * (1.0 - min(.7, centre_penalty))

            x, y, face_w, face_h = (
                int(value) for value in max(faces, key=face_rank)
            )
            return rgb.crop((
                max(0, int(x - face_w * .48)),
                max(0, int(y - face_h * .48)),
                min(width, int(x + face_w * 1.48)),
                min(height, int(y + face_h * 1.52)),
            ))
    except Exception as exc:
        logger.debug("[UniformVL] Face-detail crop fallback: %s", exc)

    hair_crop = make_hair_inspection_crop(rgb)
    crop_w, crop_h = hair_crop.size
    return hair_crop.crop((
        int(crop_w * .16), int(crop_h * .08),
        max(1, int(crop_w * .84)), max(1, int(crop_h * .76)),
    ))


def _analysis_cache_path(kind: str, *images: Image.Image) -> Path:
    """Return a stable cache path for analysis of identical image pixels."""
    digest = hashlib.sha256(f"{VL_CACHE_VERSION}:{kind}".encode("ascii"))
    for image in images:
        rgb = image.convert("RGB")
        digest.update(f"{rgb.width}x{rgb.height}".encode("ascii"))
        digest.update(rgb.tobytes())
    cache_dir = Path(tempfile.gettempdir()) / "vsoul_cache" / "vl_analysis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}_{digest.hexdigest()[:24]}.json"


_BNB_FAILURE_MARKERS = (
    "bitsandbytes",
    "application control policy",
    "cquantize_blockwise",
    "winerror 4551",
)

def _run_vl_worker(worker_path: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a VL worker, retrying unquantized when 4-bit loading is unusable.

    bitsandbytes' native DLL is blocked by Application Control on this machine.
    The blocked load does not raise inside the worker: it takes the process
    down mid-weight-load with an access violation (0xC0000005 / 3221225477),
    which no in-process ``except`` can catch. Every uniform job then reached
    the "exact portrait/template analysis failed" gate and aborted. Detect that
    crash here and re-run the same script once with the unquantized path, which
    is what the prompt encoder already does for its own worker.
    """
    run_args = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, str(worker_path)], timeout=timeout, **run_args,
        )
        if completed.returncode == 0:
            return completed
        stderr = (completed.stderr or "").lower()
        # 3221225477 is the Windows access violation this blocked DLL produces.
        crashed_in_native_code = completed.returncode in (3221225477, -1073741819)
        retry = crashed_in_native_code or any(
            marker in stderr for marker in _BNB_FAILURE_MARKERS
        )
        if not retry:
            return completed
        reason = f"returncode={completed.returncode}"
    except subprocess.TimeoutExpired:
        # A blocked 4-bit load can also hang rather than crash. Retrying is
        # still better than discarding every visual observation and falling
        # back to geometry, which aborts the whole uniform job downstream.
        timed_out = True
        reason = f"timed out after {timeout}s"
    logger.warning(
        "[UniformVL] 4-bit worker unusable (%s); retrying unquantized", reason,
    )
    # The unquantized model is larger and slower to load than the 4-bit one,
    # so reusing the quantized budget would cut the retry off mid-load and
    # report the same geometry fallback the retry exists to prevent.
    retry_timeout = max(timeout, 900) if timed_out else max(timeout * 2, 600)
    retry_env = {**os.environ, "VSOUL_VL_UNQUANTIZED": "1"}
    try:
        return subprocess.run(
            [sys.executable, str(worker_path)],
            env=retry_env, timeout=retry_timeout, **run_args,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "[UniformVL] Unquantized retry exceeded %ds; visual analysis unavailable",
            retry_timeout,
        )
        return subprocess.CompletedProcess(
            [sys.executable, str(worker_path)], 1, "",
            f"Unquantized VL worker exceeded {retry_timeout}s",
        )

def _read_analysis_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def merge_portrait_pixel_evidence(data: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    """Add conservative lighting evidence that does not depend on VL wording."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    labels = None
    try:
        from pipelines.schp_service import parse as parse_body_parts

        labels = parse_body_parts(rgb)["labels"]
    except Exception:
        labels = None
    fallback_region = rgb[
        int(height * .17):int(height * .57),
        int(width * .27):int(width * .73),
    ]
    if labels is not None and int(np.count_nonzero(labels == 13)) >= 128:
        face_pixels = rgb[labels == 13]
    else:
        face_pixels = fallback_region.reshape(-1, 3)
    region = face_pixels.reshape(-1, 1, 3)
    if region.size >= 256:
        lab_l = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)[:, :, 0]
        hsv_s = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)[:, :, 1]
        p20, p95 = np.percentile(lab_l, (20, 95))
        saturation_p80 = float(np.percentile(hsv_s, 80))
        contrast = float(p95 - p20)
        data["measured_light_p20"] = round(float(p20), 2)
        data["measured_light_p95"] = round(float(p95), 2)
        data["measured_light_contrast"] = round(contrast, 2)
        data["measured_light_saturation_p80"] = round(saturation_p80, 2)
        data["measured_sunlight_severity"] = (
            "extreme"
            if p95 >= 198.0 and contrast >= 95.0 and saturation_p80 >= 210.0
            else "hard"
            if p95 >= 185.0 and contrast >= 90.0 and saturation_p80 >= 175.0
            else "normal"
        )
        hard_sun = bool(
            p95 >= 185.0 and contrast >= 90.0 and saturation_p80 >= 175.0
        )
        # Bright silver strands can need local hair correction without the
        # person's face being sunlit. Keep those two facts separate; treating
        # every crown highlight as full-face sunlight made Qwen redraw and
        # beautify otherwise-correct heads.
        hair_hotspot = False
        if labels is not None and int(np.count_nonzero(labels == 2)) >= 128:
            hair_rgb = rgb[labels == 2].reshape(-1, 1, 3)
            hair_l = cv2.cvtColor(hair_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
            hair_p20, hair_p95 = np.percentile(hair_l, (20, 95))
            hair_hotspot = bool(hair_p95 >= 182.0 and hair_p95 - hair_p20 >= 125.0)
            data["measured_hair_light_contrast"] = round(
                float(hair_p95 - hair_p20), 2,
            )
        data["direct_sunlight_present"] = hard_sun
        data["head_hair_hotspot_present"] = hair_hotspot
        if hard_sun:
            data["head_hair_hotspot_present"] = True
            if str(data.get("sunlight_type") or "").lower() in {"", "none", "unknown"}:
                data["sunlight_type"] = "hard direct sun"
            evidence_text = str(data.get("sunlight_evidence") or "").strip().lower()
            if (
                evidence_text in {"", "none", "unknown", "uncertain"}
                or evidence_text.startswith("no ")
                or "not present" in evidence_text
            ):
                data["sunlight_evidence"] = "high face/crown contrast with saturated directional light"
            existing_regions = str(data.get("sunlight_regions") or "").strip()
            if existing_regions.lower() in {"", "none", "unknown", "uncertain"}:
                existing_regions = "forehead, hairline and crown"
            # Small VL responses sometimes serialize an array into a string.
            # Normalize that representation before it reaches the edit prompt.
            existing_regions = existing_regions.strip("[]")
            region_terms = [
                term.strip().strip("'\"")
                for term in existing_regions.split(",")
                if term.strip().strip("'\"")
            ]
            if not any("face" in term.lower() for term in region_terms):
                region_terms.append("face")
            data["sunlight_regions"] = ", ".join(dict.fromkeys(region_terms))
        elif hair_hotspot:
            data["sunlight_type"] = "localized hair glare"
            data["sunlight_evidence"] = "bright crown strands without measured facial sunlight"
            data["sunlight_regions"] = "hair and crown only"
        else:
            data["sunlight_type"] = "none"
            data["sunlight_evidence"] = "none"
            data["sunlight_regions"] = "none"
    return data


def _write_analysis_cache(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def clean_and_repair_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON from VL responses, handling code fences, trailing commas, and unclosed braces."""
    if not text or not text.strip():
        raise ValueError("Empty VL response")

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = match.group(1) if match else None
    if not candidate:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text.strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    repaired = cleaned.strip()
    # Deterministic VL occasionally emits valid one-line JSON members but
    # omits the comma between two adjacent lines.  Repair that narrow syntax
    # error before falling back; otherwise Uniform Swap silently receives the
    # generic geometry plan and spends a full diffusion run on the wrong
    # garment/head facts.
    repaired = re.sub(
        r'(?<=[\]}"\deElL])\s*\n\s*(?="[^"\n]+"\s*:)',
        ',\n',
        repaired,
        flags=re.IGNORECASE,
    )
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    open_braces = repaired.count('{') - repaired.count('}')
    if open_braces > 0:
        repaired += '}' * open_braces

    parsed = None
    for attempt in (repaired, candidate, cleaned):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Sanitize unescaped control characters/newlines inside string literals
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', lambda m: ' ' if m.group(0) in '\r\n\t' else '', attempt)
        try:
            parsed = json.loads(sanitized, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Could not parse valid JSON from VL response: {text[:200]}")
    raise ValueError(f"Parsed object was not a dictionary: {type(parsed)}")


def flatten_vl_dict(data: Any) -> Dict[str, Any]:
    """Recursively flatten nested dictionaries from VL outputs and format list values as clean strings."""
    if not isinstance(data, dict):
        return {}
    flat: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            sub_flat = flatten_vl_dict(v)
            for sub_k, sub_v in sub_flat.items():
                if sub_k not in flat or flat[sub_k] in (None, "none", "unknown", False, []):
                    flat[sub_k] = sub_v
        else:
            flat[k] = v

    # Normalize list values that should be strings for prompt builders
    for str_key in (
        "hair_accessories",
        "visible_wearables",
        "headwear_description",
        "clothing_description",
        "clothing_color_pattern",
        "clothing_collar",
        "clothing_fasteners",
        "hair_style",
        "hair_color",
        "hair_parting",
        "hair_length",
        "hair_texture",
        "hair_crown_shape",
        "hair_left_side",
        "hair_right_side",
        "hair_ends",
        "hair_fringe",
        "ponytail_side",
        "ponytail_anchor",
        "ponytail_extent",
        "tied_section_count",
        "tied_section_sides",
        "hair_accessory_count",
        "hair_accessory_confidence",
        "hair_accessory_details",
        "glasses_reflection_description",
        "skin_tone",
        "face_shape",
        "eye_description",
        "eyebrow_description",
        "nose_description",
        "mouth_description",
        "earring_description",
        "expression",
        "face_orientation",
        "jewelry_description",
        "shirt_collar",
        "outer_neckline",
        "shirt_color_and_pattern",
        "shirt_primary_color",
        "shirt_secondary_color",
        "shirt_pattern_type",
        "shirt_pattern_scale",
        "outer_primary_color",
        "outer_pattern_type",
        "button_color",
        "outer_garment_color_and_shape",
        "outer_color_under_neutral_light",
        "outer_color_confidence",
        "accent_trim_color",
        "accent_trim_placement",
        "shoulder_trim_present",
    ):
        if str_key in flat and isinstance(flat[str_key], list):
            flat[str_key] = ", ".join(str(item) for item in flat[str_key] if item) or "none"

    return flat


def normalize_hair_accessories(data: Dict[str, Any]) -> Dict[str, Any]:
    """Discard common non-hair objects that VL sometimes labels as clips."""
    value = str(data.get("hair_accessories") or "").strip().lower()
    details = str(data.get("hair_accessory_details") or "").strip().lower()
    raw_hair = data.get("hair_observations_raw")
    if isinstance(raw_hair, dict):
        raw_value = str(raw_hair.get("hair_accessories") or "").strip().lower()
        raw_details = str(raw_hair.get("hair_accessory_details") or "").strip().lower()
        raw_evidence = f"{raw_value} {raw_details}"
        # Some VL responses return a nested accessory object which the JSON
        # flattener exposes as shape/color/side/count instead of placing it in
        # hair_accessories. Preserve that visible evidence rather than letting
        # the empty summary erase real flowers, bows, ribbons or clips.
        raw_shape = str(raw_hair.get("shape") or raw_hair.get("type") or "").strip()
        raw_color = str(raw_hair.get("color") or "").strip()
        raw_side = str(raw_hair.get("side") or "").strip()
        raw_count = raw_hair.get("count")
        recognized_shape = raw_shape.lower() in {
            "flower", "bow", "ribbon", "clip", "barrette", "headband",
            "hair band", "hair tie", "elastic", "scrunchie",
        }
        if recognized_shape and value in {"", "none", "unknown", "uncertain"}:
            try:
                inferred_count = max(1, int(raw_count))
            except (TypeError, ValueError):
                inferred_count = 1
            description = " ".join(
                part for part in (raw_color, raw_shape, f"viewer-{raw_side.lower()}" if raw_side else "")
                if part
            )
            data["hair_accessories"] = raw_shape
            data["hair_accessory_details"] = description or raw_shape
            data["hair_accessory_count"] = inferred_count
            value = raw_shape.lower()
            details = description.lower()
        if any(term in raw_evidence for term in ("earring", "ear ring", "ear stud", "hoop")):
            current_wearables = str(data.get("visible_wearables") or "").strip()
            if not current_wearables or current_wearables.lower() in {"none", "unknown", "uncertain"}:
                data["visible_wearables"] = raw_details if raw_details not in {"", "none", "unknown", "uncertain"} else raw_value
            current_earring = str(data.get("earring_description") or "").strip()
            if not current_earring or current_earring.lower() in {"none", "unknown", "uncertain"}:
                data["earring_description"] = raw_details if raw_details not in {"", "none", "unknown", "uncertain"} else raw_value
    actual_hair_objects = (
        "headband", "hair band", "bow", "ribbon", "clip", "barrette",
        "flower", "elastic", "hair tie", "scrunchie",
    )
    non_hair_wearables = (
        "necklace", "chain", "pendant", "earring", "ear ring", "ear stud",
        "hoop", "bindi", "glasses", "eyeglass", "spectacle",
    )
    # Mixed lists such as "flower; necklace" used to pass merely because one
    # valid hair object was present. That made the necklace part of the hair
    # count and directly instructed Qwen to preserve it. Sanitize each object
    # independently. If any item is removed, leave the count unknown and let
    # the face-free full-hair visual reference provide the literal count.
    value_parts = [
        part.strip() for part in re.split(r"[,;]", value) if part.strip()
    ]
    detail_parts = [
        part.strip() for part in re.split(r"[;]", details) if part.strip()
    ]
    removed_wearables = [
        part for part in value_parts + detail_parts
        if any(term in part for term in non_hair_wearables)
    ]
    valid_value_parts = [
        part for part in value_parts
        if any(term in part for term in actual_hair_objects)
        and not any(term in part for term in non_hair_wearables)
    ]
    valid_detail_parts = [
        part for part in detail_parts
        if any(term in part for term in actual_hair_objects)
        and not any(term in part for term in non_hair_wearables)
    ]
    mixed_accessory_taxonomy = bool(removed_wearables)
    if mixed_accessory_taxonomy:
        data["hair_accessories"] = "; ".join(valid_value_parts) or "none"
        data["hair_accessory_details"] = "; ".join(valid_detail_parts) or "none"
        data["hair_accessory_count"] = "unknown" if valid_value_parts else 0
        data["hair_accessory_confidence"] = "low"
        current_wearables = str(data.get("visible_wearables") or "").strip()
        wearable_additions = [] if current_wearables.lower() in {
            "", "none", "unknown", "uncertain",
        } else [current_wearables]
        wearable_additions.extend(removed_wearables)
        data["visible_wearables"] = "; ".join(dict.fromkeys(wearable_additions))
        removed_earring_details = [
            part for part in removed_wearables
            if any(term in part for term in ("earring", "ear ring", "ear stud", "hoop"))
        ]
        if removed_earring_details:
            current_earring = str(data.get("earring_description") or "").strip().lower()
            if current_earring in {"", "none", "unknown", "uncertain"}:
                # Prefer the most descriptive entry (usually the details field)
                # rather than the generic category from hair_accessories.
                data["earring_description"] = max(
                    removed_earring_details, key=lambda item: len(item.split())
                )
        value = str(data["hair_accessories"]).lower()
        details = str(data["hair_accessory_details"]).lower()
    if value and value not in {"none", "unknown", "uncertain"}:
        if not any(term in value for term in actual_hair_objects):
            # The focused hair pass occasionally identifies a real earring but
            # puts it in the hair-accessory field. Do not silently erase that
            # evidence: move it to the wearable fields used by generation.
            # This remains image-driven and does not invent an accessory for
            # portraits where neither response saw one.
            misplaced = f"{value} {details}"
            if any(term in misplaced for term in ("earring", "ear ring", "ear stud", "hoop")):
                current_wearables = str(data.get("visible_wearables") or "").strip()
                if not current_wearables or current_wearables.lower() in {"none", "unknown", "uncertain"}:
                    data["visible_wearables"] = details if details not in {"", "none", "unknown", "uncertain"} else value
                current_earring = str(data.get("earring_description") or "").strip()
                if not current_earring or current_earring.lower() in {"none", "unknown", "uncertain"}:
                    data["earring_description"] = details if details not in {"", "none", "unknown", "uncertain"} else value
            data["hair_accessories"] = "none"
            data["hair_accessory_details"] = "none"
            data["hair_accessory_count"] = 0
    elif value in {"none", "unknown", "uncertain", ""}:
        # Do not allow a stale count from another response branch to invent an
        # accessory. Preserve explicit uncertain location details for review.
        if not any(term in details for term in actual_hair_objects):
            data["hair_accessory_count"] = 0
            if value == "none":
                data["hair_accessory_details"] = "none"

    # A tie/elastic is not a free-standing crown ornament: it must visibly
    # gather hair into a ponytail, bun, braid or similar tied structure. The VL
    # model occasionally calls two tiny bright crown clips a single "hair tie"
    # even while reporting loose curls and no traceable tail. Do not turn that
    # internally contradictory label into a generation fact. The source pixels
    # remain authoritative and the generator receives the literal head crop.
    value = str(data.get("hair_accessories") or "").strip().lower()
    details = str(data.get("hair_accessory_details") or "").strip().lower()
    tie_claimed = any(term in f"{value} {details}" for term in (
        "hair tie", "elastic", "scrunchie",
    ))
    tied_style = any(term in str(data.get("hair_style") or "").lower() for term in (
        "ponytail", "bun", "braid", "plait", "tied",
    ))
    tail_extent = str(data.get("ponytail_extent") or "").strip().lower()
    concrete_tail = tail_extent not in {
        "", "none", "unknown", "uncertain", "not visible", "not detected",
        "n/a", "not applicable",
    }
    if tie_claimed and not (tied_style or concrete_tail):
        data["hair_accessories"] = "uncertain source hair accessory"
        data["hair_accessory_details"] = (
            "unresolved small source-visible crown accessories; copy literal source pixels"
        )
        data["hair_accessory_count"] = "unknown"
        data["hair_accessory_confidence"] = "low"
    count = data.get("hair_accessory_count")
    try:
        if count is not None:
            data["hair_accessory_count"] = max(0, int(count))
    except (TypeError, ValueError):
        data["hair_accessory_count"] = "unknown"
    confidence = str(data.get("hair_accessory_confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        data["hair_accessory_confidence"] = "low" if "uncertain" in f"{value} {details}" else "unknown"

    # Merge corroborating wearable fields instead of allowing one short VL
    # list to erase another observation. A response may say
    # visible_wearables="earrings" while independently reporting reflections
    # on both eyeglass lenses; both facts describe the same source portrait.
    wearable_text = str(data.get("visible_wearables") or "").strip()
    wearable_terms = [
        term.strip() for term in re.split(r"[,;]", wearable_text)
        if term.strip() and term.strip().lower() not in {"none", "unknown", "uncertain"}
    ]
    reflection_value = data.get("glasses_reflection_present")
    reflection_present = (
        reflection_value is True
        or str(reflection_value or "").strip().lower() in {"true", "yes", "1"}
    )
    reflection_detail = str(data.get("glasses_reflection_description") or "").strip().lower()
    reflection_supported = reflection_present or (
        reflection_detail not in {"", "none", "no", "no reflections", "unknown", "uncertain"}
        and any(term in reflection_detail for term in ("lens", "lenses", "reflection", "glare"))
    )
    if reflection_supported and not any(
        term in item.lower() for item in wearable_terms
        for term in ("glasses", "eyeglass", "spectacle")
    ):
        wearable_terms.append("glasses")
    earring_text = str(data.get("earring_description") or "").strip().lower()
    if (
        earring_text not in {"", "none", "unknown", "uncertain", "source earrings"}
        and not any("earring" in item.lower() or "hoop" in item.lower() for item in wearable_terms)
    ):
        wearable_terms.append("earrings")
    # Prefer the descriptive observation over a generic category introduced by
    # another VL branch.  Keeping both (for example ``earrings`` and ``small
    # silver earrings``) wastes prompt budget and can be interpreted as two
    # different pairs by the image editor.
    if any(
        item.lower() != "earrings" and "earring" in item.lower()
        for item in wearable_terms
    ):
        wearable_terms = [item for item in wearable_terms if item.lower() != "earrings"]
    data["visible_wearables"] = ", ".join(wearable_terms) if wearable_terms else "none"
    return data


def build_portrait_repair_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert noisy VL observations into one conservative edit contract.

    Semantic labels are useful for locating a defect, but the uploaded pixels
    remain authoritative for identity, hair geometry and jewelry.  This layer
    prevents contradictory labels (for example ``ponytail`` plus no visible
    tail, or a flower described as a bow) from becoming generation facts.
    """
    sunlight = _coerce_bool(data.get("direct_sunlight_present")) or _coerce_bool(
        data.get("head_hair_hotspot_present")
    )
    regions = str(data.get("sunlight_regions") or "").strip()
    if regions.lower() in {"", "none", "unknown", "uncertain"}:
        regions = "the visibly overlit crown, hairline and face" if sunlight else "none"

    style = str(data.get("hair_style") or "").strip()
    ponytail_side = str(data.get("ponytail_side") or "").strip().lower()
    try:
        tied_section_count = max(0, int(data.get("tied_section_count") or 0))
    except (TypeError, ValueError):
        tied_section_count = 0
    tied_section_sides = str(data.get("tied_section_sides") or "").strip().lower()
    left_side = str(data.get("hair_left_side") or "").strip().lower()
    right_side = str(data.get("hair_right_side") or "").strip().lower()
    tied_claim = any(term in style.lower() for term in ("ponytail", "pigtail", "braid", "bun", "tied"))
    # Side fields must describe actual hair geometry. A small model can put an
    # accessory label such as "bow" or "clip" into hair_left_side; that is not
    # evidence for a tied tail and previously made a hallucinated ponytail look
    # trustworthy to generation.
    non_geometry_side_values = {
        "", "none", "unknown", "uncertain", "not visible", "not-visible",
        "bow", "clip", "flower", "ribbon", "barrette", "hair tie", "elastic",
    }
    bilateral_tied_geometry = bool(
        tied_section_count >= 2
        and any(term in tied_section_sides for term in ("both", "bilateral"))
        or (
            tied_section_count >= 2
            and "left" in tied_section_sides
            and "right" in tied_section_sides
        )
    )
    single_tied_geometry = bool(
        tied_section_count == 1
        and any(term in tied_section_sides for term in (
            "left", "right", "center", "back",
        ))
    )
    legacy_tied_geometry = bool(
        ponytail_side not in {
            "", "none", "not visible", "not-visible", "unknown", "uncertain",
        }
        and tied_section_count == 1
    )
    tied_evidence_missing = bool(
        tied_claim
        and not (bilateral_tied_geometry or single_tied_geometry or legacy_tied_geometry)
    )
    tied_geometry_ambiguous = bool(
        tied_claim
        and tied_section_count == 0
        and (
            left_side not in non_geometry_side_values
            or right_side not in non_geometry_side_values
        )
    )
    if tied_geometry_ambiguous:
        tied_evidence_missing = True

    accessory_confidence = str(data.get("hair_accessory_confidence") or "").strip().lower()
    try:
        accessory_count = max(0, int(data.get("hair_accessory_count") or 0))
    except (TypeError, ValueError):
        accessory_count = 0
    accessory_details = str(data.get("hair_accessory_details") or "").strip()
    accessory_terms = ("flower", "bow", "ribbon", "clip", "barrette", "band", "tie", "elastic")
    accessories_trusted = (
        accessory_count > 0
        and accessory_confidence in {"high", "medium"}
        and any(term in accessory_details.lower() for term in accessory_terms)
    )

    conflicts: List[str] = []
    if tied_evidence_missing:
        conflicts.append("tied hairstyle label lacks visible tail geometry")
    raw_hair = data.get("hair_observations_raw")
    raw_accessories = str(raw_hair.get("hair_accessories") or "") if isinstance(raw_hair, dict) else ""
    summary_accessories = str(data.get("hair_accessories") or "")
    if raw_accessories and summary_accessories and raw_accessories.lower() != summary_accessories.lower():
        conflicts.append("hair accessory labels disagree")

    clauses: List[str] = []
    if sunlight:
        clauses.append(
            f"Relight {regions} with soft neutral frontal studio light using this subject's shaded pixels as complexion and hair-color authority; remove direct-sun hot spots, orange/yellow cast, metallic hair glare and paired hard shadows without whitening, tanning, smoothing or changing geometry"
        )
    else:
        clauses.append("Keep the source head lighting and natural skin texture")
    if tied_evidence_missing:
        clauses.append(
            "Hairstyle labels conflict; copy the visible silhouette, part, curls and ends"
        )
    else:
        clauses.append("Copy visible hairstyle geometry without restyling")
    if accessories_trusted:
        clauses.append(
            "Preserve every visible hair accessory at its literal source position; do not trust a numeric count over the pixels"
        )
    else:
        clauses.append("Copy only visible hair accessories; invent none")
    clauses.append(
        "Source pixels override VL wording for facial geometry and ear accessories"
    )

    stable_face_fields = (
        ("face", "face_shape"),
        ("eyes", "eye_description"),
        ("brows", "eyebrow_description"),
        ("nose", "nose_description"),
        ("mouth", "mouth_description"),
    )
    face_observations: List[str] = []
    for label, key in stable_face_fields:
        observation = str(data.get(key) or "").strip()
        if observation.lower() in {
            "", "none", "unknown", "uncertain", "source face shape",
            "source eye geometry",
        }:
            continue
        face_observations.append(f"{label}: {observation[:64]}")
    if face_observations:
        clauses.append(
            "VL identity cues: "
            + "; ".join(face_observations)
        )

    return {
        "repair_required": sunlight,
        "repair_targets": [regions] if sunlight else [],
        "preserve_targets": [
            "face identity and current apparent age",
            "hair silhouette, part, texture and visible ends",
            "hair accessory count and side",
            "ear jewelry pixels",
        ],
        "hair_geometry_trusted": not tied_evidence_missing,
        "hair_accessories_trusted": accessories_trusted,
        "earring_semantics_trusted": False,
        "conflicts": conflicts,
        "prompt": ". ".join(clauses) + ".",
    }


def normalize_uniform_analysis(data: Any) -> Dict[str, Any]:
    """Keep garment-layer context when VL returns objects instead of strings."""
    if not isinstance(data, dict):
        return {}

    def describe(value: Any) -> str:
        if isinstance(value, dict):
            return "; ".join(f"{key.replace('_', ' ')}: {describe(item)}"
                             for key, item in value.items() if item is not None)
        if isinstance(value, list):
            return ", ".join(describe(item) for item in value if item is not None)
        return str(value)

    result = flatten_vl_dict(data)
    descriptions = {
        "garment_components", "shirt_color_and_pattern", "outer_garment_color_and_shape",
        "shirt_collar", "outer_neckline", "button_layout", "button_color",
        "pocket_count", "pocket_placement", "pocket_style",
        "fabric_detail", "construction_detail", "outer_color_under_neutral_light", "outer_color_confidence",
        "accent_trim_placement", "shoulder_trim_present",
    }
    # Preserve canonical fields even inside a response wrapper. Never promote a
    # generic 'color' leaf into another garment's color field.
    def visit(node: Dict[str, Any]) -> None:
        for key, value in node.items():
            if key in descriptions:
                result[key] = describe(value) if value is not None else "unknown"
            elif isinstance(value, dict):
                visit(value)

    visit(data)
    components = str(result.get("garment_components") or "").lower()
    outer_terms = (
        "vest", "waistcoat", "pinafore", "blazer", "jacket", "sweater",
        "cardigan", "coat", "outer garment", "outer layer",
    )
    if components and not any(term in components for term in outer_terms):
        # A contrast collar, placket, cuff or pocket is trim on one shirt, not
        # a separate outer garment. Promoting accent cloth to an outer layer
        # makes Qwen invent shoulder yokes, epaulettes and coloured caps.
        result["outer_garment_color_and_shape"] = "none"
        result["outer_primary_color"] = "none"
        result["outer_pattern_type"] = "none"
        result["outer_color_under_neutral_light"] = "none"
        result["outer_neutral_rgb"] = None
        result["outer_color_confidence"] = "not_applicable"
    # Preserve the literal collar construction reported from the template.
    # A former heuristic rewrote every standing/band collar on a checked shirt
    # into a folded pointed collar. Pattern and trim do not determine collar
    # geometry; that rewrite caused Qwen to replace the uploaded mandarin-style
    # collar with ordinary triangular points in d0bce584.
    return result


def _coerce_bool(value: Any) -> bool:
    """Interpret Qwen-VL boolean fields without treating the string "false" as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"true", "yes", "1", "present"}


def parse_bbox_pixels(bbox: Any, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box into (x1, y1, x2, y2) pixel integers, handling normalized (0-1, 0-1000) and absolute pixels."""
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    if max(vals) <= 1.05:
        c0, c1, c2, c3 = [v * 1000.0 for v in vals]
        is_normalized = True
    else:
        c0, c1, c2, c3 = vals
        is_normalized = (max(vals) <= 1000.0 and (w > 1100 or h > 1100))

    if is_normalized:
        y1_norm, y2_norm = min(c0, c2), max(c0, c2)
        x1_norm, x2_norm = min(c1, c3), max(c1, c3)
        x1 = int(round(x1_norm * w / 1000.0))
        x2 = int(round(x2_norm * w / 1000.0))
        y1 = int(round(y1_norm * h / 1000.0))
        y2 = int(round(y2_norm * h / 1000.0))
    else:
        pA = (min(c0, c2), max(c0, c2))
        pB = (min(c1, c3), max(c1, c3))
        if pB[1] > w and pB[1] <= h * 1.05:
            x1, x2 = int(round(pA[0])), int(round(pA[1]))
            y1, y2 = int(round(pB[0])), int(round(pB[1]))
        elif pA[1] > w and pA[1] <= h * 1.05:
            x1, x2 = int(round(pB[0])), int(round(pB[1]))
            y1, y2 = int(round(pA[0])), int(round(pA[1]))
        else:
            x1, x2 = int(round(pA[0])), int(round(pA[1]))
            y1, y2 = int(round(pB[0])), int(round(pB[1]))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rgb_hex(value: np.ndarray) -> str:
    """Format an RGB colour measurement for a prompt without rounding ambiguity."""
    r, g, b = (int(np.clip(channel, 0, 255)) for channel in value[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def extract_template_visual_evidence(template: Image.Image, components: Optional[str] = None) -> Dict[str, str]:
    """Measure palette anchors from the prepared, unpadded uniform template."""
    rgb = np.asarray(template.convert("RGB"))
    if rgb.size == 0:
        return {}
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    yy, xx = np.indices((height, width))
    non_canvas = lab[:, :, 0] < 242
    outer_terms = ("vest", "waistcoat", "pinafore", "blazer", "jacket", "sweater", "cardigan", "coat")
    has_outer = components is None or any(term in str(components).lower() for term in outer_terms)
    outer = non_canvas & (lab[:, :, 0] < 130) & (yy > height * 0.04)
    shirt_area = (xx > width * 0.12) & (xx < width * 0.88) & (yy > height * 0.02)
    shirt = (
        non_canvas
        & shirt_area
        & (lab[:, :, 0] >= 130)
        & (rgb[:, :, 2].astype(np.int16) >= rgb[:, :, 0].astype(np.int16) + 6)
    )
    evidence: Dict[str, str] = {}
    if np.count_nonzero(shirt) >= 100:
        evidence["measured_shirt_base_color"] = _rgb_hex(np.median(rgb[shirt], axis=0))
    if has_outer and np.count_nonzero(outer) >= 100:
        evidence["measured_outer_garment_color"] = _rgb_hex(np.median(rgb[outer], axis=0))
    if evidence:
        evidence["template_palette_evidence"] = "; ".join(
            # This blue-selective mask measures the colored checks, not the
            # white base fabric. Calling it a base color tinted the whole shirt.
            f"{'shirt colored pattern threads (not base fabric)' if key == 'measured_shirt_base_color' else key.removeprefix('measured_').replace('_', ' ')} {value}"
            for key, value in evidence.items()
            if key.startswith("measured_")
        )
    return evidence



class UniformVLAnalyzer:
    """Extract garment-fit and portrait constraints without retaining a VL model in VRAM."""

    _PROMPT = """You inspect two images for a formal school uniform fitting system.
Image 1 is the person portrait.
Image 2 is {second_image}.
Inspect both images carefully and return a JSON object with visible facts only:

Person (Image 1):
- hair_style: concise overall hairstyle (e.g. "curly dark hair")
- hair_parting: visible parting and placement (e.g. "center part", "side part", "none visible")
- hair_length: visible length relative to ears and shoulders (e.g. "shoulder length", "below ears")
- hair_texture: visible strand structure (e.g. "loose curls", "waves", "straight", "braids")
- hair_color: natural hair color (e.g. "dark brown", "black")
- hair_accessories: list any visible headbands, bows, ribbons, clips, or "none" (e.g. "red headband with decorative bow")
- hair_accessory_details: each visible accessory's side, color, material, and shape; use "none" when absent
- visible_wearables: visible earrings, bindi, necklaces, glasses, or "none"

Uniform Blueprint (Image 2):
- garment_components: list of visible clothing pieces (e.g. ["shirt", "vest"] or ["shirt", "tie", "blazer"] or ["polo shirt"])
- shirt_collar: exact shirt collar construction (e.g. "folded pointed collar with two free downward points", "button-down pointed collar", "mandarin standing band collar without points", "crossover overlap band collar without free points", "round collar")
- outer_neckline: exact neckline of the outer garment (e.g. "deep V-neck vest", "crew neck", "blazer lapel", "none")
- shirt_color_and_pattern: shirt color and pattern (e.g. "blue and white micro-checkered", "plain white")
- outer_garment_color_and_shape: observed outer garment color and style
- outer_color_under_neutral_light: perceived base fabric color after discounting shadows, highlights, and color cast
- outer_neutral_rgb: three integers [R, G, B] estimating that base fabric under neutral studio light
- outer_color_confidence: high, medium, or low
- sleeve_length: sleeve length (e.g. "short sleeves", "long sleeves", "sleeveless", or "unknown" if cropped)
- button_layout: description of visible buttons on shirt and outer garment (e.g. "front placket with 2 visible white buttons")
- badge_present: boolean (true if any school crest, emblem, patch, logo, or text badge is visible on the uniform; false otherwise)
- badge_location: location of the badge (e.g. "left chest of vest", "right breast", or "none")
- badge_bbox: [ymin, xmin, ymax, xmax] coordinates of the badge on a 0-1000 normalized scale, or null if none

Key garment guidelines:
- Classify collar geometry from visible edges, not from fabric pattern, trim color, or a typical school-uniform expectation. Report "folded pointed collar" only when TWO distinct free triangular tips fold downward onto the chest. A standing/mandarin band wraps the neck without those free tips. If left and right collar bands cross, overlap, or converge at the center neck while remaining attached along their lower edges, report "crossover overlap band collar without free points". Do not turn an overlap seam or V-shaped opening into invented triangular points.
- Distinguish a V-neck outer garment (e.g. V-neck vest) from a high collar: if the outer layer plunges or opens in front to expose the shirt, outer_neckline is "V-neck" or "open vest".
- Distinguish a sleeveless vest from a blazer/jacket: a blazer has a separate front opening and folded lapels that frame the shirt; report "blazer with lapels" whenever those folded panels are visible. Do not call a blazer a vest merely because its sleeves are cropped or hidden.
- Treat flower-shaped clips, bows, ribbons and hair ties as distinct accessories. Describe their visible shape and side exactly; never substitute one for another.
Return JSON ONLY with keys: garment_components, shirt_collar, outer_neckline, shirt_color_and_pattern, outer_garment_color_and_shape, outer_color_under_neutral_light, outer_neutral_rgb, outer_color_confidence, sleeve_length, button_layout, badge_present, badge_location, badge_bbox, hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_accessories, hair_accessory_details, visible_wearables.
Do not describe or identify the person."""

    _GARMENT_RETRY_PROMPT = """Analyze this uniform template only. Return one JSON object and nothing else.
Required keys: garment_components, shirt_color_and_pattern, outer_garment_color_and_shape,
outer_color_under_neutral_light, outer_neutral_rgb, outer_color_confidence,
shirt_collar, outer_neckline, sleeve_length, button_layout, badge_present, badge_location, badge_bbox.
Describe only visible garment facts. Do not describe a person.
Classify collar geometry literally. A folded pointed collar requires two distinct free triangular tips lying down on the chest. A standing/mandarin band has no free tips; when its two bands cross or overlap at center, report "crossover overlap band collar without free points". Pattern and trim color never determine collar type. Distinguish a V-neck vest from a high collar.
badge_bbox must be [ymin, xmin, ymax, xmax] normalized to 1000, or null if no badge.
Use concise values and complete the JSON object."""

    _HAIR_RETRY_PROMPT = """Analyze only the source portrait's visible hair from crown through both sides and the back. Return one JSON object and nothing else.
Required keys: hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_crown_shape, hair_left_side, hair_right_side, hair_ends, hair_fringe, tied_section_count, tied_section_sides, ponytail_side, ponytail_anchor, ponytail_extent, hair_accessories, hair_accessory_count, hair_accessory_confidence, hair_accessory_details, visible_wearables.
Trace the visible hair silhouette before answering: crown shape, left edge, right edge, hair behind each ear, ends near the neck or shoulders, and any visible ponytail, bun, braid, loose strand or fringe. Record those observations separately in hair_crown_shape, hair_left_side, hair_right_side, hair_ends and hair_fringe.
For every tied section, inspect the strand construction all the way from its root to its visible end. In hair_style explicitly distinguish an interwoven braid/plait from an unbraided ponytail, twist, loose section or curl; never call a visibly plaited section merely "pigtails". hair_texture describes both the strand texture and any visible weave/plait construction.
Count independently tied hair sections in tied_section_count. A pair of pigtails or two low side ponytails is count 2, never one generic ponytail. tied_section_sides must enumerate viewer-left and viewer-right separately, say bilateral/both sides only when both are visible, or say centered-back/not-visible. If exactly one ponytail is visible, ponytail_side must say viewer-left, viewer-right, centered-back, or not-visible; ponytail_anchor must state where it emerges relative to the ear/nape; ponytail_extent must state its visible end relative to neck/collar/shoulder. Never infer a single side from which tail appears larger; these are literal image coordinates, not the person's anatomical left/right.
hair_accessories may contain only an object physically attached to the hair: headband, bow, ribbon, clip, flower, elastic, band, bead cluster or hair tie. Inspect the root and end of every tied section separately. A bow near the ear and a beaded elastic at the lower end are two distinct accessories on that section and must both be enumerated. State each object's side, color, attachment type, material, literal outline/geometry and count in hair_accessory_details, return the integer total in hair_accessory_count, and set hair_accessory_confidence to high, medium or low. If an object is a clip with a decorative outline, keep its type as clip and describe that outline separately; do not rename a clip as a flower merely because its silhouette has several lobes.
Eyeglass frames and temples, earrings, bindis, necklaces, clothing, skin highlights, background edges and glare are NOT hair accessories. Put glasses, earrings, bindi or necklace in visible_wearables instead.
If an object is unclear, say uncertain and describe its location with low confidence; do not convert glasses or earrings into clips. Derive base hair color from shaded strands, not sunlight or backdrop spill. Do not report an accessory as none until both sides and the crown have been checked.
Do not describe or identify the person."""

    _BOARD_PROMPT = """This is one three-panel fitting board. Left panel is PERSON,
middle panel is the exact UNIFORM TEMPLATE, and right panel is a ROUGH FIT.
Inspect only the visual geometry needed to fit the exact template onto the person.
Return JSON only with keys: pose, torso_visibility, shoulders_visible,
neck_visibility, template_type, collar, badge_present, risks,
collar_anchor_ratio, uniform_scale, and fit_confidence.
collar_anchor_ratio is a number from 0.88 to 1.05 measured from the top of the
detected face box to the desired top of the collar. uniform_scale is a number
from 0.88 to 1.10. Do not identify, describe, or alter the person."""

    _PORTRAIT_PROMPT = """Inspect this portrait only to prepare a constrained studio enhancement plan.
Return JSON only with keys: source_subject_count, crown_near_top_edge, hair_edge_risk,
direct_sunlight_present, head_hair_hotspot_present, sunlight_type, sunlight_direction, sunlight_evidence, sunlight_regions,
glasses_present, headwear_present, headwear_description,
glasses_reflection_present, glasses_reflection_description,
hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_accessories, hair_accessory_details,
clothing_description, clothing_color_pattern, clothing_collar, clothing_fasteners, garment_visibility, skin_tone,
face_orientation, head_tilt, shoulder_alignment, expression, face_shape, eye_description, eyebrow_description, nose_description,
mouth_description, earring_description, jewelry_description, apparent_age_band, facial_maturity,
cheek_fullness, jaw_shape, chin_shape, and face_width.

Guidelines:
source_subject_count is the number of people visibly present in the source image.
crown_near_top_edge is true only when the top of the hair is close to the image edge.
direct_sunlight_present is true only for visible directional sunlight: a bright hot spot paired with a hard shadow, or an obvious outdoor sun wash. Do not mark ordinary soft indoor light as sunlight.
head_hair_hotspot_present is true only when bright glare is visible on the forehead, hairline, crown, or hair.
sunlight_type is one of "hard direct sun", "broad sun wash", "soft indoor", or "none". sunlight_direction is the visible direction, such as "upper left", or "none".
sunlight_evidence is a concise reason (or "none"). sunlight_regions must inspect the complete visible person and list every affected region (or "none"), including forehead, crown, either cheek, hair, ears, neck, chest, shoulders, arms or hands when visibly affected. Do not stop after finding light on the face.
hair_accessories must list visible headbands, bows, ribbons, clips, or "none".
hair_accessory_details must enumerate every visible accessory separately by left/right side, color, shape and count.
Flowers are flowers, not headbands; clips are clips, not jewelry. Inspect both sides independently and return "none"
instead of guessing when uncertain.
Count partially occluded matching accessories on opposite sides separately. Never mirror an accessory that is absent,
but do not collapse two visible side accessories into one. apparent_age_band and facial_maturity must describe the
current photographed age only from facial evidence; never de-age, age-up, or infer age from clothing or context.
cheek_fullness, jaw_shape, chin_shape and face_width must describe literal visible proportions without beautification.
headwear_present is true only for hats, caps, turbans, hijabs (distinct from hair accessories).
glasses_reflection_present is true only when a visible highlight or scene reflection crosses either eyeglass lens. glasses_reflection_description states left lens, right lens or both and its visible color/intensity. Do not mistake the iris highlight for lens reflection.
face_orientation must state whether the face is frontal, three-quarter, profile, or uncertain. head_tilt must state upright, tilted viewer-left, tilted viewer-right, or uncertain. shoulder_alignment must state level/front-facing, viewer-left higher, viewer-right higher, rotated, or uncertain. Describe only visible geometry; do not infer a pose from hairstyle or clothing.
jewelry_description must list visible earrings, necklaces, chains, pendants, bindis, or "none".
face_shape, eye_description, eyebrow_description, nose_description and mouth_description must concisely describe
only stable visible geometry. Do not infer ethnicity, nationality or identity. earring_description must describe the
source earrings' type, size, metal/color and position, or "none".
clothing_description must be a short generic description of the visible source garment.
clothing_color_pattern, clothing_collar, and clothing_fasteners must describe only visible features.
skin_tone must concisely describe the natural base complexion and undertone visible in evenly lit facial areas
(for example "medium-brown with warm-neutral undertone"). Ignore sunlight, highlights, shadow, makeup, background
spill and camera white-balance cast; never infer base skin color from a bright hot spot.
hair_color must describe the base pigment visible in shaded strands. Describe silver, blue or gold glare in sunlight_evidence instead of treating it as the base hair color. If the base color is unclear, say "uncertain".
Use concise generic descriptions. Do not identify the person."""

    @staticmethod
    def inpaint_badge(
        image: Image.Image,
        badge_bbox: Optional[List[int]] = None,
    ) -> Image.Image:
        """Inpaint school badge/emblem area with surrounding garment fabric using seamless texture blending."""
        if not badge_bbox or len(badge_bbox) != 4:
            return image
        try:
            rgba = image.convert("RGBA")
            arr = np.array(rgba)
            h, w = arr.shape[:2]

            box = parse_bbox_pixels(badge_bbox, w, h)
            if not box:
                return image
            x1, y1, x2, y2 = box

            # Sample fabric to the left of the badge (or right if near left edge)
            if x1 > 40:
                sample_x1, sample_x2 = max(0, x1 - 60), max(0, x1 - 10)
            else:
                sample_x1, sample_x2 = min(w, x2 + 10), min(w, x2 + 60)
            sample_y1, sample_y2 = max(0, y1), min(h, y2)
            fabric_sample = arr[sample_y1:sample_y2, sample_x1:sample_x2, :3]
            if fabric_sample.size == 0:
                mean_color = np.array([30, 40, 55], dtype=np.float32)
                std_color = np.array([6, 6, 8], dtype=np.float32)
            else:
                mean_color = np.mean(fabric_sample, axis=(0, 1))
                std_color = np.std(fabric_sample, axis=(0, 1))

            mask = np.zeros((h, w), dtype=np.uint8)
            center = ((x1 + x2) // 2, min(h - 1, (y1 + y2) // 2 + 12))
            axes = (max(4, (x2 - x1) // 2 + 12), max(4, (y2 - y1) // 2 + 30))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            noise = np.random.normal(mean_color, std_color, (h, w, 3)).clip(0, 255).astype(np.uint8)
            mask_f = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
            mask_3d = mask_f[:, :, None]

            rgb = arr[:, :, :3].astype(np.float32)
            blended = rgb * (1.0 - mask_3d) + noise.astype(np.float32) * mask_3d
            arr[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)

            logger.info("[UniformVL] Seamlessly inpainted badge in template bbox=[%d, %d, %d, %d]", x1, y1, x2, y2)
            return Image.fromarray(arr)
        except Exception as err:
            logger.warning("[UniformVL] Badge inpainting failed: %s", err)
            return image

    def analyze_portrait(self, image: Image.Image) -> Dict[str, Any]:
        """Run short-lived Qwen VL analysis before Qwen Image Edit portrait work in isolated subprocess."""
        result: Dict[str, Any] = {
            "source_subject_count": 1,
            "crown_near_top_edge": False,
            "hair_edge_risk": "low",
            "direct_sunlight_present": False,
            "head_hair_hotspot_present": False,
            "sunlight_type": "none",
            "sunlight_direction": "none",
            "sunlight_evidence": "none",
            "sunlight_regions": "none",
            "glasses_present": False,
            "glasses_reflection_present": False,
            "glasses_reflection_description": "none",
            "headwear_present": False,
            "headwear_description": "none",
            "hair_style": "copy source-visible hairstyle without reinterpretation",
            "hair_parting": "source hair parting",
            "hair_length": "source hair length",
            "hair_texture": "source hair texture",
            "hair_color": "source-visible hair color under neutral light",
            "hair_crown_shape": "source crown shape",
            "hair_left_side": "source left hair silhouette",
            "hair_right_side": "source right hair silhouette",
            "hair_ends": "source visible hair ends",
            "hair_fringe": "source fringe and loose strands",
            "tied_section_count": 0,
            "tied_section_sides": "unknown",
            "hair_accessories": "copy source-visible hair accessories only",
            "hair_accessory_count": None,
            "hair_accessory_confidence": "unknown",
            "hair_accessory_details": "none",
            "clothing_description": "source clothing",
            "clothing_color_pattern": "source clothing color and pattern",
            "clothing_collar": "source neckline",
            "clothing_fasteners": "source garment details",
            "garment_visibility": "upper garment visible",
            "skin_tone": "source natural skin tone",
            "face_shape": "source face shape",
            "eye_description": "source eye geometry",
            "eyebrow_description": "source eyebrow geometry",
            "nose_description": "source nose geometry",
            "mouth_description": "source mouth geometry",
            "earring_description": "source earrings",
            "face_orientation": "source camera orientation",
            "head_tilt": "uncertain",
            "shoulder_alignment": "uncertain",
            "expression": "source expression",
            "jewelry_description": "visible source jewelry",
            "apparent_age_band": "source apparent age",
            "facial_maturity": "source facial maturity",
            "cheek_fullness": "source cheek fullness",
            "jaw_shape": "source jaw shape",
            "chin_shape": "source chin shape",
            "face_width": "source face width",
            "source": "fallback",
        }
        if not MODEL_DIR.is_dir():
            return result
        analysis_cache = _analysis_cache_path("portrait", image)
        cached = _read_analysis_cache(analysis_cache)
        if cached is not None:
            logger.info("[PortraitVL] Analysis cache hit: %s", analysis_cache.name)
            cached = merge_portrait_pixel_evidence(
                normalize_hair_accessories(cached), image
            )
            cached["portrait_repair_plan"] = build_portrait_repair_plan(cached)
            return cached
        cache_dir = Path(tempfile.gettempdir()) / "vsoul_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        image_path = cache_dir / f"portrait_vl_{stamp}.png"
        hair_path = cache_dir / f"portrait_hair_{stamp}.png"
        hair_left_path = cache_dir / f"portrait_hair_left_{stamp}.png"
        hair_right_path = cache_dir / f"portrait_hair_right_{stamp}.png"
        face_path = cache_dir / f"portrait_face_{stamp}.png"
        worker_path = cache_dir / f"portrait_vl_{stamp}.py"
        try:
            inspection = image.convert("RGB")
            inspection.thumbnail((512, 512), Image.Resampling.LANCZOS)
            inspection.save(image_path)
            hair_inspection = make_hair_inspection_crop(image)
            hair_inspection.thumbnail((640, 640), Image.Resampling.LANCZOS)
            hair_inspection.save(hair_path)
            crop_w, crop_h = hair_inspection.size
            hair_inspection.crop((0, 0, max(1, int(crop_w * .60)), crop_h)).resize(
                (384, 512), Image.Resampling.LANCZOS
            ).save(hair_left_path)
            hair_inspection.crop((min(crop_w - 1, int(crop_w * .40)), 0, crop_w, crop_h)).resize(
                (384, 512), Image.Resampling.LANCZOS
            ).save(hair_right_path)
            face_inspection = make_face_inspection_crop(image)
            face_inspection.thumbnail((640, 640), Image.Resampling.LANCZOS)
            face_inspection.save(face_path)
            worker_code = f'''import gc
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
image_path = Path({str(image_path.resolve())!r})
hair_path = Path({str(hair_path.resolve())!r})
hair_left_path = Path({str(hair_left_path.resolve())!r})
hair_right_path = Path({str(hair_right_path.resolve())!r})
face_path = Path({str(face_path.resolve())!r})
prompt = {self._PORTRAIT_PROMPT!r}
hair_prompt = {self._HAIR_RETRY_PROMPT!r}
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
# Prefer the 4-bit load: it is what keeps this analysis inside an 8 GB GPU
# alongside Qwen Image Edit. bitsandbytes ships a native DLL that Application
# Control blocks on this machine; the blocked load then takes the whole worker
# down with an access violation during weight loading, which a try/except in
# this process cannot catch. The parent re-runs this script with
# VSOUL_VL_UNQUANTIZED=1 after such a crash, so honour that first and keep the
# in-process except for the softer failures.
def _load_unquantized(reason):
    # This 3B checkpoint is ~7.1 GB in bf16 against 8.5 GB of VRAM. It fits on
    # the GPU, but only whole: device_map="auto" splits it and streams the
    # remainder through host RAM, which fails here with "The paging file is too
    # small for this operation" (os error 1455). Load it onto one device.
    print("VL_WORKER_UNQUANTIZED_FALLBACK: " + str(reason)[:300], file=sys.stderr)
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, device_map={{"": target}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as gpu_error:
        if target == "cpu":
            raise
        print(
            "VL_WORKER_CPU_FALLBACK: " + str(gpu_error)[:300], file=sys.stderr,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float32, device_map={{"": "cpu"}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()

if os.environ.get("VSOUL_VL_UNQUANTIZED") == "1":
    model = _load_unquantized("parent requested the unquantized retry")
else:
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ),
            device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as quantized_load_error:
        model = _load_unquantized(quantized_load_error)
portrait = Image.open(image_path).convert("RGB")
head_crop = Image.open(hair_path).convert("RGB")
head_left = Image.open(hair_left_path).convert("RGB")
head_right = Image.open(hair_right_path).convert("RGB")
face_crop = Image.open(face_path).convert("RGB")
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": portrait}},
    {{"type": "image", "image": face_crop}},
    {{"type": "text", "text": (
        "Image 1 is the full portrait for pose, body and complete hair context. Image 2 is a detailed crop of the same face and both ears; use it for facial geometry, current apparent age, complexion, eyeglasses, lens reflections and ear jewelry. Do not infer age from clothing, school context or background. " + prompt
    )}}
]}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[portrait, face_crop], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
hair_messages = [{{"role": "user", "content": [
    {{"type": "image", "image": portrait}},
    {{"type": "image", "image": head_crop}},
    {{"type": "image", "image": head_left}},
    {{"type": "image", "image": head_right}},
    {{"type": "text", "text": (
        "Image 1 shows the full portrait and is required for hair below the ears, at the nape, or beside the neck. "
        "Image 2 is the complete head crop. Image 3 enlarges the viewer-left side and image 4 enlarges the viewer-right side of that same head. "
        "Reconcile all views and count only actually visible accessories on each side. Do not call the hair short merely because the crown is smooth; "
        "check image 1 for a tied ponytail, braid, bun, or hair bundle behind either shoulder. " + hair_prompt
    )}}
]}}]
hair_text = processor.apply_chat_template(hair_messages, tokenize=False, add_generation_prompt=True)
hair_inputs = processor(
    text=[hair_text], images=[portrait, head_crop, head_left, head_right],
    padding=True, return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    hair_generated = model.generate(**hair_inputs, max_new_tokens=384, do_sample=False)
hair_response = processor.batch_decode(
    hair_generated[:, hair_inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0].strip()
print("---PORTRAIT_ANALYSIS_START---")
print(json.dumps({{"portrait_raw": response, "hair_raw": hair_response}}))
print("---PORTRAIT_ANALYSIS_END---")
del generated, hair_generated, inputs, hair_inputs, portrait, head_crop, head_left, head_right, face_crop, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            completed = _run_vl_worker(worker_path, timeout=120)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "Qwen VL worker failed")
            response = completed.stdout.strip()
            start_marker = "---PORTRAIT_ANALYSIS_START---"
            end_marker = "---PORTRAIT_ANALYSIS_END---"
            if start_marker not in response or end_marker not in response:
                raise RuntimeError(
                    "Qwen VL worker returned no analysis payload: " + response[-800:]
                )
            payload = response.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
            combined = json.loads(payload)
            portrait_raw = str(combined.get("portrait_raw") or "").strip()
            hair_raw = str(combined.get("hair_raw") or "").strip()
            try:
                parsed = flatten_vl_dict(clean_and_repair_json(portrait_raw))
            except Exception as portrait_parse_error:
                logger.warning(
                    "[PortraitVL] Portrait JSON invalid: %s; raw=%s",
                    portrait_parse_error, portrait_raw[-600:],
                )
                parsed = {}
            if isinstance(parsed, dict):
                result.update(parsed)
                try:
                    hair_parsed = flatten_vl_dict(clean_and_repair_json(hair_raw))
                except Exception as hair_parse_error:
                    logger.warning(
                        "[PortraitVL] Hair JSON invalid: %s; raw=%s",
                        hair_parse_error, hair_raw[-600:],
                    )
                    hair_parsed = {}
                if isinstance(hair_parsed, dict):
                    for key in (
                        "hair_style", "hair_parting", "hair_length", "hair_texture",
                        "hair_color", "hair_crown_shape", "hair_left_side", "hair_right_side",
                        "hair_ends", "hair_fringe", "tied_section_count", "tied_section_sides",
                        "ponytail_side", "ponytail_anchor", "ponytail_extent",
                        "hair_accessories", "hair_accessory_count",
                        "hair_accessory_confidence", "hair_accessory_details",
                    ):
                        value = hair_parsed.get(key)
                        if value not in (None, "", "unknown", "uncertain"):
                            result[key] = value
                    result["hair_observations_raw"] = hair_parsed
                normalize_hair_accessories(result)
                merge_portrait_pixel_evidence(result, image)
                for key in (
                    "crown_near_top_edge",
                    "direct_sunlight_present",
                    "head_hair_hotspot_present",
                    "glasses_present",
                    "glasses_reflection_present",
                    "headwear_present",
                ):
                    result[key] = _coerce_bool(result.get(key))
                result["source"] = "qwen2.5-vl"
                result["portrait_repair_plan"] = build_portrait_repair_plan(result)
                logger.info(
                    "[PortraitVL] source=%s hair_acc=%s headwear=%s clothing=%s skin=%s",
                    result["source"],
                    result.get("hair_accessories"),
                    result.get("headwear_description"),
                    str(result.get("clothing_description", "source clothing"))[:80],
                    result.get("skin_tone"),
                )
                _write_analysis_cache(analysis_cache, result)
        except Exception as exc:
            logger.warning("[PortraitVL] Analysis fallback: %s", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image_path.unlink(missing_ok=True)
            hair_path.unlink(missing_ok=True)
            hair_left_path.unlink(missing_ok=True)
            hair_right_path.unlink(missing_ok=True)
            face_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)
        result = merge_portrait_pixel_evidence(
            normalize_hair_accessories(result), image
        )
        result["portrait_repair_plan"] = build_portrait_repair_plan(result)
        return result

    @staticmethod
    def fallback(person: Image.Image, template: Image.Image) -> Dict[str, Any]:
        width, height = person.size
        result = {
            "pose": "front portrait",
            "torso_visibility": "limited" if height < width * 1.2 else "visible",
            "shoulders_visible": True,
            "head_to_waist_visible": False,
            "template_type": "product cutout",
            "collar_free_tip_count": None,
            "badge_present": False,
            "badge_location": "none",
            "badge_bbox": None,
            "garment_components": ["shirt", "vest"],
            "shirt_color_and_pattern": "checkered shirt",
            "outer_garment_color_and_shape": "unknown outer garment color and shape",
            "outer_color_under_neutral_light": "unknown",
            "outer_neutral_rgb": None,
            "outer_color_confidence": "low",
            "sleeve_length": "unknown",
            "button_layout": "front placket buttons",
            "shirt_collar": "mandarin band collar at neck base",
            "outer_neckline": "V-neck vest",
            "hair_style": "copy source-visible hairstyle without reinterpretation",
            "hair_color": "source-visible hair color under neutral light",
            "hair_crown_shape": "source crown shape",
            "hair_left_side": "source left hair silhouette",
            "hair_right_side": "source right hair silhouette",
            "hair_ends": "source visible hair ends",
            "hair_fringe": "source fringe and loose strands",
            "hair_accessories": "copy source-visible hair accessories only",
            "hair_accessory_count": None,
            "hair_accessory_confidence": "unknown",
            "visible_wearables": "none",
            "gender": "unknown",
            "apparent_age_band": "unknown",
            "glasses_reflection_present": False,
            "glasses_reflection_description": "none",
            "direct_sunlight_present": False,
            "head_hair_hotspot_present": False,
            "sunlight_type": "none",
            "sunlight_direction": "none",
            "sunlight_evidence": "none",
            "sunlight_regions": "none",
            "garment_risks": ["Preserve every visible template layer."],
            "risks": ["Use deterministic face lock and seam-only repair."],
            "source": "geometry fallback",
        }
        result.update(extract_template_visual_evidence(template))
        return result

    def analyze(
        self, person: Image.Image, template: Image.Image, second_image: str = "the required uniform template"
    ) -> Dict[str, Any]:
        """Run uniform and person VL analysis in an isolated subprocess to reclaim all VRAM upon completion."""
        result = self.fallback(person, template)
        if not MODEL_DIR.is_dir():
            return result
        analysis_cache = _analysis_cache_path("uniform", person, template)
        cached = _read_analysis_cache(analysis_cache)
        if cached is not None:
            logger.info("[UniformVL] Analysis cache hit: %s", analysis_cache.name)
            cached = merge_portrait_pixel_evidence(
                normalize_hair_accessories(cached), person
            )
            cached["portrait_repair_plan"] = build_portrait_repair_plan(cached)
            return cached

        cache_dir = Path(tempfile.gettempdir()) / "vsoul_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        person_path = cache_dir / f"uniform_p_{stamp}.png"
        hair_path = cache_dir / f"uniform_h_{stamp}.png"
        crown_path = cache_dir / f"uniform_c_{stamp}.png"
        face_path = cache_dir / f"uniform_f_{stamp}.png"
        template_path = cache_dir / f"uniform_t_{stamp}.png"
        worker_path = cache_dir / f"uniform_vl_{stamp}.py"
        try:
            person_inspection = person.convert("RGB")
            person_inspection.thumbnail((512, 512), Image.Resampling.LANCZOS)
            person_inspection.save(person_path)
            hair_inspection = make_hair_inspection_crop(person)
            # Small phone portraits can leave clips and studs only a handful
            # of pixels wide. Upscale the inspection image before the VL
            # processor tokenises it so tiny source-visible objects survive
            # the model's own resize.
            if max(hair_inspection.size) < 768:
                inspection_scale = 768.0 / max(1, max(hair_inspection.size))
                hair_inspection = hair_inspection.resize(
                    (
                        max(1, int(round(hair_inspection.width * inspection_scale))),
                        max(1, int(round(hair_inspection.height * inspection_scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
            hair_inspection.thumbnail((896, 896), Image.Resampling.LANCZOS)
            hair_inspection.save(hair_path)
            crown_inspection = hair_inspection.crop((
                0, 0, hair_inspection.width,
                max(1, int(round(hair_inspection.height * .52))),
            ))
            crown_inspection.save(crown_path)
            face_inspection = make_face_inspection_crop(person)
            if max(face_inspection.size) < 768:
                inspection_scale = 768.0 / max(1, max(face_inspection.size))
                face_inspection = face_inspection.resize(
                    (
                        max(1, int(round(face_inspection.width * inspection_scale))),
                        max(1, int(round(face_inspection.height * inspection_scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
            face_inspection.thumbnail((896, 896), Image.Resampling.LANCZOS)
            face_inspection.save(face_path)
            template_inspection = template.convert("RGB")
            template_inspection.thumbnail((640, 640), Image.Resampling.LANCZOS)
            template_inspection.save(template_path)
            p_prompt = (
                "Images 1 and 2 show the same person. Image 1 is the full portrait for pose, body, clothing and complete hair context. Image 2 is a detailed face-and-ears crop and is authoritative for facial geometry, current apparent age, complexion, eyeglasses, lens reflections and ear jewelry. "
                "Inspect this portrait only. Return JSON with keys: gender, apparent_age_band, hair_style, hair_parting, hair_length, hair_texture, "
                "hair_color, hair_accessories, hair_accessory_details, visible_wearables, clothing_description, "
                "clothing_color_pattern, clothing_collar, clothing_fasteners, skin_tone, face_orientation, expression, "
                "face_shape, eye_description, eyebrow_description, nose_description, mouth_description, earring_description, face_detail, "
                "glasses_reflection_present, glasses_reflection_description, "
                "direct_sunlight_present, head_hair_hotspot_present, sunlight_type, sunlight_direction, sunlight_evidence, sunlight_regions.\n"
                "gender must be 'female' or 'male' based on visible facial features, jewelry/earrings, hairstyle, and presentation.\n"
                "apparent_age_band should be 'school child', 'teenager', 'young adult', or 'adult', based only on visible facial maturity in image 2. Never infer age from a uniform, school setting, body size or background; use uncertain when facial evidence is ambiguous.\n"
                "hair_accessories must list only objects physically attached to hair: headbands, bows, ribbons, clips, flowers, elastics or hair ties, or 'none'.\n"
                "hair_accessory_details must state side, color, and shape for each visible accessory.\n"
                "Inspect both sides of the crown separately for small clips. Use uncertain for obscured details, not none.\n"
                "Eyeglass frames and temples, earrings, bindis, necklaces, clothing edges, highlights and background marks are not hair accessories; list glasses and jewelry only under visible_wearables.\n"
                "Trace the crown, both side silhouettes, hair behind each ear, and any visible ends, ponytail, bun, braid, fringe or loose strands before deciding hair_style and hair_length.\n"
                "Describe face_shape, eye_description, eyebrow_description, nose_description and mouth_description separately using only stable visible geometry. "
                "face_detail must be a concise cross-check of those same visible traits. Do not infer ethnicity, nationality, attractiveness or identity, and do not propose beautification.\n"
                "CRITICAL: Image 1 is the exact source person identity authority. Identify all features with maximum precision so the exact individual (complexion, face geometry, eyes, nose, mouth, hair, accessories) is completely preserved without any alterations.\n"
                "direct_sunlight_present is true for visible directional sunlight, hard shadows, or broad outdoor sun wash.\n"
                "head_hair_hotspot_present is true for bright glare on the forehead, hairline, crown, roots, or hair.\n"
                "sunlight_type is hard direct sun, broad sun wash, soft indoor, or none. sunlight_direction states its visible direction.\n"
                "sunlight_evidence briefly describes the evidence. Inspect the full visible person: sunlight_regions lists every affected area, including forehead, crown, roots, either hair side, ears, neck, chest, shoulders, arms or hands. Do not stop after detecting the face.\n"
                "hair_color and skin_tone describe base pigment, not sunlight glare. Report orange/yellow skin cast and silver, cyan, white or gold crown glare as lighting evidence; do not call these soft or normal lighting.\n"
                "Describe hair and clothing only from visible pixels; do not infer or invent missing details.\n"
                "visible_wearables must list visible earrings, bindi, glasses, necklaces, or 'none'. Inspect both ears in image 2 and report the literal earring count/type without mirroring or dropping the dimmer side.\n"
                "glasses_reflection_present is true only for a real highlight or scene reflection over an eyeglass lens; glasses_reflection_description states left, right or both lenses and the reflection appearance.\n"
                "Return JSON only:"
            )
            u_prompt = (
                "Inspect this uniform template image only. Return JSON with keys: garment_components, "
                "shirt_color_and_pattern, shirt_primary_color, shirt_secondary_color, shirt_pattern_type, "
                "shirt_pattern_scale, outer_garment_color_and_shape, outer_primary_color, outer_pattern_type, "
                "outer_color_under_neutral_light, outer_neutral_rgb, outer_color_confidence, "
                "collar_free_tip_count, shirt_collar, outer_neckline, sleeve_length, button_layout, button_color, badge_present, "
                "badge_location, badge_bbox, pocket_count, pocket_placement, pocket_style, "
                "fabric_detail, construction_detail, accent_trim_color, accent_trim_placement, "
                "shoulder_trim_present.\n"
                "Rules:\n"
                "- Inspect literal template pixels only. Do not describe a generic school uniform or infer unseen fabric.\n"
                "- MANDATORY: The final output must perfectly match this uniform template. Inspect every component (collar, placket, seams, checks, colors, buttons, trim) with high precision so it is replicated as an authentic tailored fit.\n"
                "- Inspect each layer separately: collar, shirt front, sleeves, and outer garment. Do not let the backdrop or badge influence the fabric color description.\n"
                "- First decide whether this is one shirt with contrast trim or multiple garment layers. A contrasting collar, center placket, pocket edge or sleeve cuff remains part of the shirt and is NOT an outer garment.\n"
                "- accent_trim_placement must list only the exact regions made from solid contrast cloth, such as collar, center placket and sleeve-end cuffs.\n"
                "- accent_trim_color must state the literal color of that solid contrast cloth under neutral light, or none when no contrast trim exists. Do not substitute a plaid/check thread color.\n"
                "- shoulder_trim_present must be true only when a separate solid trim, epaulette, cap or piping is visibly sewn across the top shoulder seam. Printed plaid lines do not count.\n"
                "- Judge the outer fabric's real base color after discounting shadows, highlights, camera white balance and reflected backdrop light. Put that judgment in outer_color_under_neutral_light.\n"
                "- outer_neutral_rgb must be [R, G, B] integers for that same base color under neutral studio lighting, not a raw shadow pixel and not the backdrop color. Set confidence low if ambiguous.\n"
                "- Separate the cloth ground color from the thin pattern-line color. Describe the large spaces BETWEEN lines as the ground; do not call the whole shirt blue merely because it has blue lines. If unresolved say uncertain.\n"
                "- Classify the shirt pattern as check, stripe, plaid, print, or solid and its scale as micro, fine, medium, or large. Note line thickness, crossing directions and contrast; do not invent measurements.\n"
                "- A single-color outer garment can still have visible woven texture. Report color pattern and surface texture separately; do not equate solid color with smooth fabric.\n"
                "- COLLAR, STEP 1 of 2 - count before naming. A free tip is a corner of collar cloth that ends in mid-air over the chest, with background or shirt visible on BOTH sides of its point, so it could be lifted away from the body. Trace the collar outline from the centre front outward on each side and set collar_free_tip_count to the integer number of such free corners: 0, 1 or 2. A corner whose entire lower edge stays sewn to the shirt is NOT free and does not count.\n"
                "- COLLAR, STEP 2 of 2 - derive shirt_collar from that count alone, never from the check pattern, the trim colour, or what a school uniform usually looks like. 2 free tips means folded pointed collar. 0 free tips with a band wrapping the neck means standing mandarin band collar. 0 free tips with left and right bands crossing or converging at centre while their lower edges stay attached means crossover overlap band collar without free points. If the count and your first impression disagree, trust the count. Do not mistake a centre overlap seam or a V-shaped opening for a free tip, and do not report crossover merely because two free tips happen to meet near the centre.\n"
                "- Describe collar ends, attachment edges, center overlap/crossover, and whether two free folded triangular tips are actually present. Do not invent points, piping or trim.\n"
                "- Separate the light base cloth from colored grid threads; specify their contrast and spacing relative to a button.\n"
                "- fabric_detail describes visible weave and surface finish for each layer. Inspect dark fabric for fine grain before calling it smooth. Use uncertain if unresolved.\n"
                "- construction_detail must describe the actual left and right front edges, angled or curved panel shapes, seam lines, topstitching and overlapping layers. A V-neck description alone is insufficient. Distinguish stitched panel edges from printed stripes and shadows. Report a blazer or jacket with folded lapels when those folded triangular panels frame the shirt; do not downgrade it to a vest because sleeves are cropped or hidden.\n"
                "- button_layout and button_color describe each visible layer separately: position, visible count, color and round/other shape. Do not infer hidden buttons below the crop.\n"
                "- pocket_count is the integer number of visible functional chest pockets. Require a visible pocket opening or independent pocket perimeter. A badge, crest backing panel, emblem, printed square, seam or color block attached to otherwise continuous fabric is NOT a pocket.\n"
                "- pocket_placement must state viewer-left, viewer-right, both sides, centered, or none. Never mirror a single pocket.\n"
                "- pocket_style must state plain patch, flap, buttoned flap, welt, hidden, none, or uncertain from visible construction. Do not invent a flap or button when the pocket edge is plain.\n"
                "- Before returning JSON, cross-check collar, cloth ground versus grid color, outer texture and panel edges against the image. Replace unsupported claims with uncertain. Use concise string values for descriptions, not generic uniform recommendations.\n"
                "- Distinguish a V-neck outer vest from a blazer with lapels: a vest has no folded jacket panels framing the shirt.\n"
                "- badge_present: true if an emblem, crest, logo patch, or school badge is visible.\n"
                "- badge_bbox: [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax] coordinates on this image, or null if no badge.\n"
                "Return JSON only:"
            )
            h_prompt = (
                "Images 1-3 show the same source head: image 1 is the complete hair silhouette, image 2 is an enlarged crown/accessory view, and image 3 is the face-and-both-ears detail. Describe literal visible pixels, not a likely hairstyle. Return JSON with exactly these keys: "
                "hair_style, hair_parting, hair_length, hair_texture, hair_color, hair_crown_shape, hair_left_side, hair_right_side, hair_ends, hair_fringe, "
                "tied_section_count, tied_section_sides, ponytail_side, ponytail_anchor, ponytail_extent, hair_accessories, hair_accessory_count, hair_accessory_confidence, hair_accessory_details, "
                "earring_description, earring_count, earring_details. "
                "First trace the crown, viewer-left outline, viewer-right outline, hair behind each ear and visible ends. Loose side hair or curls ending near the jaw, neck or shoulders are not a ponytail. Small clipped or tied crown sections are not proof of a ponytail; report a ponytail only when one continuous tied tail and its anchor are visibly traceable. "
                "Count every independently tied hair section in tied_section_count. For two low side ponytails/pigtails return 2 and enumerate viewer-left plus viewer-right in tied_section_sides; never collapse a bilateral style into one generic ponytail or choose only the larger side. For one tied section return 1 and its literal viewer side/centered-back location. For loose hair return 0 and none. "
                "Then inspect the crown in four zones: viewer-left outer, viewer-left inner, viewer-right inner and viewer-right outer. Do not let the brighter or larger side hide a pale, shadowed or partly cropped matching object on the opposite side. Count each physically separate clip, flower, bow, ribbon, band, elastic or tie exactly once. A real flower requires visibly separate organic petals; a bow requires fabric loops. A small rigid or metallic fastener remains a clip even when its decorative outline has lobes. State every accessory separately with viewer side, color, attachment type, material, literal outline/geometry and approximate crown position. Before returning JSON, enumerate the viewer-left and viewer-right accessories again and verify that hair_accessory_count equals the number of separately described physical objects; never collapse two side objects into one shared description. "
                "Finally inspect both ears separately. Earrings are ear jewelry, never hair accessories. Give the literal earring count, small/medium/large scale, metal or color, stud/hoop/drop shape and viewer side. If resolution is insufficient, say uncertain rather than inventing or mirroring an item. "
                "Every value must be a scalar string except tied_section_count, hair_accessory_count and earring_count, which must be integers. Do not return nested objects or arrays. Put all separately enumerated accessories into one semicolon-separated hair_accessory_details string. Put both-ear findings into one semicolon-separated earring_details string. "
                "Do not call curls braids without visibly interwoven strands. Describe base hair color from shaded strands, not glare. Do not infer gender. Return JSON only."
            )

            worker_code = f'''import gc
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
person_path = Path({str(person_path.resolve())!r})
hair_path = Path({str(hair_path.resolve())!r})
crown_path = Path({str(crown_path.resolve())!r})
face_path = Path({str(face_path.resolve())!r})
template_path = Path({str(template_path.resolve())!r})
p_prompt = {p_prompt!r}
u_prompt = {u_prompt!r}
h_prompt = {h_prompt!r}

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
# Prefer the 4-bit load: it is what keeps this analysis inside an 8 GB GPU
# alongside Qwen Image Edit. bitsandbytes ships a native DLL that Application
# Control blocks on this machine; the blocked load then takes the whole worker
# down with an access violation during weight loading, which a try/except in
# this process cannot catch. The parent re-runs this script with
# VSOUL_VL_UNQUANTIZED=1 after such a crash, so honour that first and keep the
# in-process except for the softer failures.
def _load_unquantized(reason):
    # This 3B checkpoint is ~7.1 GB in bf16 against 8.5 GB of VRAM. It fits on
    # the GPU, but only whole: device_map="auto" splits it and streams the
    # remainder through host RAM, which fails here with "The paging file is too
    # small for this operation" (os error 1455). Load it onto one device.
    print("VL_WORKER_UNQUANTIZED_FALLBACK: " + str(reason)[:300], file=sys.stderr)
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, device_map={{"": target}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as gpu_error:
        if target == "cpu":
            raise
        print(
            "VL_WORKER_CPU_FALLBACK: " + str(gpu_error)[:300], file=sys.stderr,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float32, device_map={{"": "cpu"}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()

if os.environ.get("VSOUL_VL_UNQUANTIZED") == "1":
    model = _load_unquantized("parent requested the unquantized retry")
else:
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ),
            device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as quantized_load_error:
        model = _load_unquantized(quantized_load_error)

person_img = Image.open(person_path).convert("RGB")
head_crop = Image.open(hair_path).convert("RGB")
crown_crop = Image.open(crown_path).convert("RGB")
face_crop = Image.open(face_path).convert("RGB")
template_img = Image.open(template_path).convert("RGB")

msg_p = [{{"role": "user", "content": [
    {{"type": "image", "image": person_img}},
    {{"type": "image", "image": face_crop}},
    {{"type": "text", "text": p_prompt}}
]}}]
txt_p = processor.apply_chat_template(msg_p, tokenize=False, add_generation_prompt=True)
inp_p = processor(text=[txt_p], images=[person_img, face_crop], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    gen_p = model.generate(**inp_p, max_new_tokens=768, do_sample=False)
out_p = processor.batch_decode(gen_p[:, inp_p.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

msg_h = [{{"role": "user", "content": [
    {{"type": "image", "image": head_crop}},
    {{"type": "image", "image": crown_crop}},
    {{"type": "image", "image": face_crop}},
    {{"type": "text", "text": h_prompt}}
]}}]
txt_h = processor.apply_chat_template(msg_h, tokenize=False, add_generation_prompt=True)
inp_h = processor(text=[txt_h], images=[head_crop, crown_crop, face_crop], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    gen_h = model.generate(**inp_h, max_new_tokens=640, do_sample=False)
out_h = processor.batch_decode(gen_h[:, inp_h.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

msg_u = [{{"role": "user", "content": [{{"type": "image", "image": template_img}}, {{"type": "text", "text": u_prompt}}]}}]
txt_u = processor.apply_chat_template(msg_u, tokenize=False, add_generation_prompt=True)
inp_u = processor(text=[txt_u], images=[template_img], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    gen_u = model.generate(**inp_u, max_new_tokens=768, do_sample=False)
out_u = processor.batch_decode(gen_u[:, inp_u.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

combined = {{"person_raw": out_p, "hair_raw": out_h, "uniform_raw": out_u}}
print("---COMBINED_START---")
print(json.dumps(combined))
print("---COMBINED_END---")

del gen_p, gen_h, gen_u, inp_p, inp_h, inp_u, person_img, head_crop, crown_crop, face_crop, template_img, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            # Three focused observations need longer than the previous
            # ceiling on a busy local GPU. A timeout here discarded all
            # visual evidence and silently forced geometry fallback.
            completed = _run_vl_worker(worker_path, timeout=420)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr[-1200:] or "Uniform VL worker failed")

            response = completed.stdout.strip()
            # Retain the latest raw multi-pass response.  This contains no
            # model weights or secrets; it is essential for repairing a
            # deterministic malformed JSON response without running diffusion.
            (cache_dir / "last_uniform_vl_response.txt").write_text(
                response, encoding="utf-8",
            )
            match = re.search(r"---COMBINED_START---\s*(\{.*?\})\s*---COMBINED_END---", response, flags=re.DOTALL)
            if match:
                combined_dict = json.loads(match.group(1))
                p_dict = flatten_vl_dict(clean_and_repair_json(combined_dict.get("person_raw", "")))
                hair_observations = clean_and_repair_json(combined_dict.get("hair_raw", ""))
                h_dict = flatten_vl_dict(hair_observations)
                uniform_observations = clean_and_repair_json(combined_dict.get("uniform_raw", ""))
                u_dict = normalize_uniform_analysis(uniform_observations)
                result["uniform_observations_raw"] = uniform_observations
                # Missing model fields must not masquerade as observed facts.
                for key in ("garment_components", "shirt_color_and_pattern", "outer_garment_color_and_shape",
                            "outer_color_under_neutral_light", "outer_neutral_rgb", "outer_color_confidence",
                            "collar_free_tip_count", "shirt_collar", "outer_neckline", "button_layout", "pocket_count",
                            "pocket_placement", "pocket_style", "hair_style", "hair_color",
                            "hair_accessories", "visible_wearables", "gender", "apparent_age_band"):
                    result[key] = "unknown"
                result.update(p_dict)
                for key in ("direct_sunlight_present", "head_hair_hotspot_present", "glasses_reflection_present"):
                    result[key] = _coerce_bool(result.get(key))
                for key in (
                    "hair_style", "hair_parting", "hair_length", "hair_texture", "hair_color",
                    "hair_crown_shape", "hair_left_side", "hair_right_side", "hair_ends", "hair_fringe",
                    "tied_section_count", "tied_section_sides",
                    "ponytail_side", "ponytail_anchor", "ponytail_extent",
                    "hair_accessories", "hair_accessory_count", "hair_accessory_confidence", "hair_accessory_details",
                    "earring_description", "earring_count", "earring_details",
                ):
                    value = h_dict.get(key)
                    if value not in (None, "", "unknown", "uncertain"):
                        result[key] = value
                # Reconcile independent views instead of blindly allowing the
                # tighter crop to undercount a pale edge accessory.  The full
                # portrait pass can clearly report bilateral objects while a
                # crown crop sees only the brighter one.  This is evidence-
                # driven for every upload: no accessory type or count is
                # assumed unless the model explicitly reports both sides.
                portrait_accessory_details = str(
                    p_dict.get("hair_accessory_details") or ""
                ).strip()
                bilateral_text = portrait_accessory_details.lower()
                bilateral_accessory_evidence = bool(
                    "both sides" in bilateral_text
                    or "each side" in bilateral_text
                    or (
                        any(term in bilateral_text for term in ("viewer-left", "left side"))
                        and any(term in bilateral_text for term in ("viewer-right", "right side"))
                    )
                )
                try:
                    focused_accessory_count = int(result.get("hair_accessory_count") or 0)
                except (TypeError, ValueError):
                    focused_accessory_count = 0
                if bilateral_accessory_evidence and focused_accessory_count < 2:
                    result["hair_accessory_count"] = 2
                    focused_details = str(result.get("hair_accessory_details") or "").strip()
                    result["hair_accessory_details"] = (
                        f"{portrait_accessory_details}; focused detail: {focused_details}"
                        if focused_details and focused_details.lower() not in {"none", "unknown", "uncertain"}
                        else portrait_accessory_details
                    )
                    result["hair_accessory_confidence"] = "high"
                result["hair_observations_raw"] = hair_observations
                normalize_hair_accessories(result)
                result.update(u_dict)
                result["source"] = "qwen2.5-vl"
            else:
                parsed = flatten_vl_dict(clean_and_repair_json(response))
                if isinstance(parsed, dict):
                    result.update(parsed)
                    result["source"] = "qwen2.5-vl"

            # Keep deterministic palette evidence even if VL uses an
            # incomplete colour name. Both refer to this prepared template.
            result.update(extract_template_visual_evidence(template, result.get("garment_components")))
            # Pixel evidence is authoritative when the language response says
            # "no sunlight" but simultaneously describes crown glare or a
            # strongly split face. Apply it in the combined uniform analyzer,
            # not only in the standalone Enhancement analyzer.
            merge_portrait_pixel_evidence(result, person)
            logger.info(
                "[UniformVL] Analysis succeeded: components=%s shirt=%s pattern=%s/%s outer=%s palette=%s badge=%s bbox=%s hair_acc=%s",
                result.get("garment_components"),
                result.get("shirt_primary_color") or result.get("shirt_color_and_pattern"),
                result.get("shirt_pattern_type"),
                result.get("shirt_pattern_scale"),
                result.get("outer_garment_color_and_shape"),
                result.get("template_palette_evidence"),
                result.get("badge_present"),
                result.get("badge_bbox"),
                result.get("hair_accessories"),
            )
            if result.get("source") == "qwen2.5-vl":
                result["portrait_repair_plan"] = build_portrait_repair_plan(result)
                _write_analysis_cache(analysis_cache, result)
        except Exception as exc:
            logger.warning("[UniformVL] Primary subprocess analysis failed: %s; using geometry fallback", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            person_path.unlink(missing_ok=True)
            hair_path.unlink(missing_ok=True)
            crown_path.unlink(missing_ok=True)
            face_path.unlink(missing_ok=True)
            template_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)

        result = merge_portrait_pixel_evidence(result, person)
        result["portrait_repair_plan"] = build_portrait_repair_plan(result)
        return result

    def analyze_board(self, board: Image.Image) -> Dict[str, Any]:
        """Analyze the combined person/template/rough-fit board in one VL image."""
        result = self.fallback(board, board)
        result.update({"collar_anchor_ratio": 0.96, "uniform_scale": 1.0, "fit_confidence": "limited"})
        if not MODEL_DIR.is_dir():
            return result
        analysis_cache = _analysis_cache_path("board", board)
        cached = _read_analysis_cache(analysis_cache)
        if cached is not None:
            logger.info("[UniformVL] Board-analysis cache hit: %s", analysis_cache.name)
            return cached

        cache_dir = Path(tempfile.gettempdir()) / "vsoul_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        board_path = cache_dir / f"board_vl_{stamp}.png"
        worker_path = cache_dir / f"board_vl_{stamp}.py"
        try:
            inspection = board.convert("RGB")
            inspection.thumbnail((640, 640), Image.Resampling.LANCZOS)
            inspection.save(board_path)
            worker_code = f'''import gc
import json
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import transformers.utils.import_utils as _transformers_import_utils
_transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

model_dir = Path({str(MODEL_DIR.resolve())!r})
board_path = Path({str(board_path.resolve())!r})
prompt = {self._BOARD_PROMPT!r}

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
# Prefer the 4-bit load: it is what keeps this analysis inside an 8 GB GPU
# alongside Qwen Image Edit. bitsandbytes ships a native DLL that Application
# Control blocks on this machine; the blocked load then takes the whole worker
# down with an access violation during weight loading, which a try/except in
# this process cannot catch. The parent re-runs this script with
# VSOUL_VL_UNQUANTIZED=1 after such a crash, so honour that first and keep the
# in-process except for the softer failures.
def _load_unquantized(reason):
    # This 3B checkpoint is ~7.1 GB in bf16 against 8.5 GB of VRAM. It fits on
    # the GPU, but only whole: device_map="auto" splits it and streams the
    # remainder through host RAM, which fails here with "The paging file is too
    # small for this operation" (os error 1455). Load it onto one device.
    print("VL_WORKER_UNQUANTIZED_FALLBACK: " + str(reason)[:300], file=sys.stderr)
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, device_map={{"": target}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as gpu_error:
        if target == "cpu":
            raise
        print(
            "VL_WORKER_CPU_FALLBACK: " + str(gpu_error)[:300], file=sys.stderr,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float32, device_map={{"": "cpu"}},
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()

if os.environ.get("VSOUL_VL_UNQUANTIZED") == "1":
    model = _load_unquantized("parent requested the unquantized retry")
else:
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir), quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ),
            device_map={{"": 0}} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True, local_files_only=True,
        ).eval()
    except Exception as quantized_load_error:
        model = _load_unquantized(quantized_load_error)

board_img = Image.open(board_path).convert("RGB")
messages = [{{"role": "user", "content": [{{"type": "image", "image": board_img}}, {{"type": "text", "text": prompt}}]}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[board_img], padding=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
print(response)

del generated, inputs, board_img, model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
            worker_path.write_text(worker_code, encoding="utf-8")
            completed = _run_vl_worker(worker_path, timeout=120)
            if completed.returncode == 0:
                parsed = clean_and_repair_json(completed.stdout.strip())
                if isinstance(parsed, dict):
                    result.update(parsed)
                    result["source"] = "qwen2.5-vl-board"
                    _write_analysis_cache(analysis_cache, result)
        except Exception as exc:
            logger.warning("[UniformVL] Board analysis fallback: %s", exc)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            board_path.unlink(missing_ok=True)
            worker_path.unlink(missing_ok=True)

        return result


uniform_vl_analyzer = UniformVLAnalyzer()
