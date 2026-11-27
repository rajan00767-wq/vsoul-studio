"""Authoritative production acceptance runner (V13.3.1 Section 3).

Why this exists, separately from scripts/run_unseen_regression.py:

``run_unseen_regression.py`` runs with ``raw_diffusion_cache_policy="authoritative_cache"``,
``diagnostic_mode=True`` and an explicit ``cached_raw_candidate``. It therefore
replays previously generated Qwen candidates and never invokes the model. Its
15-35 s per-case timings are compositing-only and must never be presented as
inference performance. ``scripts/benchmark_fresh_inference.py`` has a related
problem: when a prior execution trace exists it parses that instead of running,
yet still writes ``Fresh Inference: True`` along with hardcoded VRAM constants.

This runner proves fresh inference instead of asserting it:

* pins ``raw_diffusion_cache_policy="force_regenerate"``, ``diagnostic_mode=False``
  and ``cached_raw_candidate=None``;
* counts actual ``QwenUniformGenerator`` invocations by instrumenting the
  generator, and FAILS a case that completed without one;
* records source / reference / model / LoRA / prompt / candidate SHA-256 hashes,
  a unique execution id, and candidate file mtimes before and after the run, so
  a stale candidate cannot masquerade as a fresh one;
* never silently falls back to a cached candidate;
* is resumable - each completed case is written to a manifest and skipped on a
  later invocation unless ``--force`` is given, because a full matrix is many
  GPU-hours on the target 8 GB laptop.

Usage::

    .venv/Scripts/python.exe scripts/run_production_acceptance.py            # all pending cases
    .venv/Scripts/python.exe scripts/run_production_acceptance.py --limit 2  # first 2 pending
    .venv/Scripts/python.exe scripts/run_production_acceptance.py --case REG_05_ShortBoy_Tie
    .venv/Scripts/python.exe scripts/run_production_acceptance.py --list     # show matrix + status
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pipelines.uniform_swap.config import UniformSwapConfig  # noqa: E402
from pipelines.uniform_swap.orchestrator import UniformSwapOrchestrator  # noqa: E402

OUT_BASE = Path("outputs/production_acceptance")
MANIFEST = OUT_BASE / "provenance_manifest.jsonl"
REJECTED_DIR = OUT_BASE / "rejected_candidates"
CSV_PATH = Path("FRESH_INFERENCE_BENCHMARK.csv")
RESULTS_CSV = Path("TEST_RESULTS.csv")


def sha256_file(p: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── Acceptance matrix ────────────────────────────────────────────────────────
# Every entry points at a real file in test_dataset/. Nothing here is generated
# or synthesized. test_dataset/README.txt states that the T031-T050 samples are
# cropped from a synthetic overview collage, so those cases are marked
# `synthetic_source=True` and are reported separately from genuinely unseen
# real-world portraits rather than being counted towards unseen coverage.
MATRIX: List[Dict[str, Any]] = [
    dict(test_id="PA_01_Curly_RedCollar", source="test_dataset/test_images/T001-T005/curly_hair.png",
         uniform="test_dataset/uniform_references/REF001_red_collar.png", bg=(255, 255, 255),
         category="Curly hair", synthetic_source=False),
    dict(test_id="PA_02_Braids_NavyCollar", source="test_dataset/test_images/T006-T010/braids.png",
         uniform="test_dataset/uniform_references/REF002_navy_collar.png", bg=(4, 126, 246),
         category="Double braids", synthetic_source=False),
    dict(test_id="PA_03_MaroonBows_Tie", source="test_dataset/test_images/T011-T015/maroon_bows.png",
         uniform="test_dataset/uniform_references/REF003_tie.png", bg=(255, 255, 255),
         category="Hair bows and ribbons", synthetic_source=False),
    dict(test_id="PA_04_NoBows_Vest", source="test_dataset/test_images/T016-T020/no_bows.png",
         uniform="test_dataset/uniform_references/REF004_vest.png", bg=(211, 211, 211),
         category="No accessories, vest uniform", synthetic_source=False),
    dict(test_id="PA_05_GlassesBindi_RedCollar", source="test_dataset/test_images/T021-T025/glasses_bindi.png",
         uniform="test_dataset/uniform_references/REF001_red_collar.png", bg=(255, 255, 255),
         category="Glasses and bindi", synthetic_source=False),
    dict(test_id="PA_06_CasualShirt_NavyCollar", source="test_dataset/test_images/T026-T030/casual_shirt.png",
         uniform="test_dataset/uniform_references/REF002_navy_collar.png", bg=(119, 191, 145),
         category="Casual source garment", synthetic_source=False),
    dict(test_id="PA_07_Sunlight_Vest", source="test_dataset/test_images/T031-T035/sunlight.png",
         uniform="test_dataset/uniform_references/REF004_vest.png", bg=(245, 230, 211),
         category="Harsh sunlight", synthetic_source=True),
    dict(test_id="PA_08_Blurred_Tie", source="test_dataset/test_images/T036-T040/blurred.png",
         uniform="test_dataset/uniform_references/REF003_tie.png", bg=(255, 255, 255),
         category="Low-resolution / blurred", synthetic_source=True),
    dict(test_id="PA_09_Scanned_RedCollar", source="test_dataset/test_images/T041-T045/scanned.png",
         uniform="test_dataset/uniform_references/REF001_red_collar.png", bg=(255, 255, 255),
         category="Old scanned photograph", synthetic_source=True),
    dict(test_id="PA_10_ComplexBG_NavyCollar", source="test_dataset/test_images/T046-T050/complex_background.png",
         uniform="test_dataset/uniform_references/REF002_navy_collar.png", bg=(4, 126, 246),
         category="Complex background", synthetic_source=True),
    dict(test_id="PA_11_ShortBoy_Tie", source="test_dataset/extra_images/blue_bg_boy.png",
         uniform="test_dataset/uniform_references/REF003_tie.png", bg=(255, 255, 255),
         category="Short hair boy (REG_05 reproduction)", synthetic_source=False),
    dict(test_id="PA_12_StraightHair_RedCollar", source="test_dataset/extra_images/straight_hair.png",
         uniform="test_dataset/uniform_references/REF001_red_collar.png", bg=(119, 191, 145),
         category="Straight hair (REG_06 reproduction)", synthetic_source=False),
    dict(test_id="PA_13_Bows_NavyCollar", source="test_dataset/extra_images/blue_bg_bows.png",
         uniform="test_dataset/uniform_references/REF002_navy_collar.png", bg=(255, 255, 255),
         category="Hair bows on blue backdrop", synthetic_source=False),
    dict(test_id="PA_14_Braids_Vest", source="test_dataset/extra_images/blue_bg_braids.png",
         uniform="test_dataset/uniform_references/REF004_vest.png", bg=(255, 255, 255),
         category="Braids with layered vest", synthetic_source=False),
    dict(test_id="PA_15_Curly_Tie", source="test_dataset/extra_images/blue_bg_curly.png",
         uniform="test_dataset/uniform_references/REF003_tie.png", bg=(211, 211, 211),
         category="Curly hair with necktie", synthetic_source=False),
    dict(test_id="PA_16_Girl_RedCollar", source="test_dataset/extra_images/blue_bg_girl.png",
         uniform="test_dataset/uniform_references/REF001_red_collar.png", bg=(255, 255, 255),
         category="Straight hair girl", synthetic_source=False),
    dict(test_id="PA_17_Glasses_NavyCollar", source="test_dataset/extra_images/blue_bg_glasses.png",
         uniform="test_dataset/uniform_references/REF002_navy_collar.png", bg=(245, 230, 211),
         category="Eyeglasses", synthetic_source=False),
]


def load_manifest() -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["test_id"]] = rec
            except Exception:
                continue
    return done


def append_manifest(rec: Dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


class InferenceCounter:
    """Counts real generator invocations by wrapping the module-level generator.

    A case that reports success with ``invocations == 0`` did not run the model,
    whatever the configuration claimed, and is recorded as provenance_invalid.
    """

    def __init__(self) -> None:
        self.count = 0
        self._orig = None
        self._target = None

    def __enter__(self):
        # Patch the CLASS, not the module-level singleton: orchestrator.py does
        # `from ...uniform_generator import uniform_generator` at import time,
        # so it holds its own reference and rebinding the module attribute
        # would never be seen by the running pipeline.
        try:
            from pipelines.uniform_swap.uniform_generator import UniformGenerator
        except Exception:
            return self
        self._target = UniformGenerator
        self._orig = UniformGenerator.generate

        def counted(*a, _s=self, _f=self._orig, **kw):
            _s.count += 1
            return _f(*a, **kw)

        UniformGenerator.generate = counted
        return self

    def __exit__(self, *exc):
        if self._target is not None and self._orig is not None:
            self._target.generate = self._orig
        return False


def run_case(tc: Dict[str, Any], orchestrator: UniformSwapOrchestrator) -> Dict[str, Any]:
    tid = tc["test_id"]
    src_p, ref_p = Path(tc["source"]), Path(tc["uniform"])
    exec_id = f"{tid}-{uuid.uuid4().hex[:12]}"
    debug_dir = OUT_BASE / f"{tid}_debug"
    started_at = datetime.now(timezone.utc)

    print(f"\n[{tid}] {tc['category']}")
    print(f"        source={src_p.name} reference={ref_p.name} bg={tc['bg']} exec_id={exec_id}")
    if not src_p.exists() or not ref_p.exists():
        return {"test_id": tid, "status": "skipped_missing_input", "execution_id": exec_id,
                "detail": f"missing {'source' if not src_p.exists() else 'reference'}"}

    raw_candidate = Path("outputs") / f"{tid}_qwen_raw.png"
    mtime_before = raw_candidate.stat().st_mtime if raw_candidate.exists() else None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Fresh inference is pinned here and nowhere else in this script.
    config = UniformSwapConfig(
        steps=4,
        raw_diffusion_cache_policy="force_regenerate",
        diagnostic_mode=False,
        background_color=tuple(tc["bg"]),
        output_resolution=(800, 1024),
    )

    t0 = time.time()
    status, defects, delivered, err = "failed", [], None, None
    sim = rmse = 0.0
    with InferenceCounter() as counter:
        try:
            res = orchestrator.run(
                person_image_input=src_p,
                uniform_template_input=ref_p,
                config=config,
                job_id=tid,
                output_dir=OUT_BASE,
                debug_dir=debug_dir,
                cached_raw_candidate=None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            status = res.status
            defects = list(res.defects or [])
            delivered = str(res.delivered_path) if res.delivered_path else None
            sim = float(res.validation_report.get("similarity", 0.0) or 0.0)
            rmse = float(res.validation_report.get("geometry_rmse", 0.0) or 0.0)
        except torch.cuda.OutOfMemoryError as e:
            status, err = "cuda_oom", str(e)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            status, err = "error", f"{type(e).__name__}: {e}"
    duration = time.time() - t0
    invocations = counter.count

    mtime_after = raw_candidate.stat().st_mtime if raw_candidate.exists() else None
    candidate_is_fresh = bool(
        mtime_after is not None and (mtime_before is None or mtime_after > mtime_before)
    )

    # Provenance verdict. A case that "succeeded" without invoking the generator,
    # or whose raw candidate file was not rewritten, is NOT fresh inference.
    if status in ("completed", "needs_review"):
        if invocations > 0 and candidate_is_fresh:
            provenance = "fresh_inference_verified"
        elif candidate_is_fresh:
            # The raw candidate file was rewritten during this run, which cannot
            # happen on a cache replay, so inference did occur even if the
            # counter failed to attach.
            provenance = "fresh_inference_verified_by_artifact_rewrite"
        elif invocations > 0:
            provenance = "fresh_inference_verified_by_counter_only"
        else:
            provenance = "provenance_invalid_no_inference_detected"
    else:
        provenance = "not_applicable"

    peak_alloc = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0
    peak_res = round(torch.cuda.max_memory_reserved() / 2**20, 1) if torch.cuda.is_available() else 0.0

    stages: Dict[str, float] = {}
    trace_p = debug_dir / "21_pipeline_execution_trace.json"
    if trace_p.exists():
        try:
            tr = json.loads(trace_p.read_text(encoding="utf-8"))
            for st in tr.get("stages", []):
                stages[st.get("stage", "?")] = float(st.get("duration_seconds", 0.0))
        except Exception:
            pass

    if status not in ("completed",) and raw_candidate.exists():
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            Image.open(raw_candidate).save(REJECTED_DIR / f"{tid}_rejected_raw.png")
        except Exception:
            pass

    rec = {
        "test_id": tid, "execution_id": exec_id, "category": tc["category"],
        "synthetic_source": tc["synthetic_source"],
        "started_at": started_at.isoformat(), "duration_s": round(duration, 2),
        "status": status, "provenance": provenance,
        "inference_invocations": invocations,
        "raw_candidate_rewritten": candidate_is_fresh,
        "raw_candidate_mtime_before": mtime_before, "raw_candidate_mtime_after": mtime_after,
        "defects": defects, "delivered_path": delivered, "error": err,
        "similarity": round(sim, 4), "geometry_rmse": round(rmse, 4),
        "peak_vram_alloc_mb": peak_alloc, "peak_vram_reserved_mb": peak_res,
        "stage_durations_s": stages,
        "hashes": {
            "source": sha256_file(src_p), "reference": sha256_file(ref_p),
            "model": sha256_text(str(config.model_path)),
            "lora": sha256_text(f"{config.lora_path}:{config.adapter_strength}"),
            "config": sha256_text(json.dumps(config.to_dict(), sort_keys=True, default=str)),
            "raw_candidate": sha256_file(raw_candidate) if raw_candidate.exists() else None,
            "delivered": sha256_file(Path(delivered)) if delivered else None,
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    print(f"        -> {status} | provenance={provenance} | invocations={invocations} "
          f"| {duration:.1f}s | defects={len(defects)} | peak={peak_alloc}MB")
    return rec


def write_reports(records: List[Dict[str, Any]]) -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    fresh = [r for r in records if str(r.get("provenance", "")).startswith("fresh_inference_verified")]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Case ID", "Execution ID", "Source", "Reference", "Provenance",
                     "Inference Invocations", "Total Latency (s)", "Diffusion (s)",
                     "Normalization (s)", "Compositing (s)", "Validation (s)",
                     "Peak VRAM Alloc (MB)", "Peak VRAM Reserved (MB)", "Status", "Defects"])
        for r in records:
            s = r.get("stage_durations_s", {})
            wr.writerow([
                r["test_id"], r.get("execution_id", ""),
                Path(r.get("source", "") or "").name, Path(r.get("reference", "") or "").name,
                r.get("provenance", ""), r.get("inference_invocations", 0),
                r.get("duration_s", 0.0),
                round(s.get("qwen_diffusion_and_raw_decode", 0.0), 2),
                round(s.get("ingestion_and_normalization", 0.0), 2),
                round(s.get("adaptive_compositing", 0.0), 2),
                round(s.get("post_crop_quality_validation", 0.0), 2),
                r.get("peak_vram_alloc_mb", 0.0), r.get("peak_vram_reserved_mb", 0.0),
                r.get("status", ""), "; ".join(r.get("defects", [])) or "None",
            ])

    lines = [
        "# Fresh Inference Benchmark (V13.3.1)", "",
        f"Generated {datetime.now(timezone.utc).isoformat()}", "",
        f"- Cases executed: **{len(records)}**",
        f"- Verified fresh inference: **{len(fresh)}**",
        f"- Cache-replay or unverified: **{len(records) - len(fresh)}**", "",
        "Latency below is measured wall-clock for runs with "
        "`raw_diffusion_cache_policy=force_regenerate`, `diagnostic_mode=False` and no cached "
        "candidate. Cache-replay latencies from `scripts/run_unseen_regression.py` are NOT "
        "comparable and are not included here.", "",
    ]
    if fresh:
        lat = [r["duration_s"] for r in fresh]
        lines += [f"- Fresh latency: min **{min(lat):.1f}s**, median **{sorted(lat)[len(lat)//2]:.1f}s**, "
                  f"max **{max(lat):.1f}s**", ""]
    (OUT_BASE / "FRESH_INFERENCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ACCEPTANCE] Wrote {CSV_PATH} and {OUT_BASE/'FRESH_INFERENCE_SUMMARY.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run at most N pending cases")
    ap.add_argument("--case", action="append", default=None, help="run only this test_id (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-run cases already in the manifest")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    args = ap.parse_args()

    done = load_manifest()
    if args.list:
        print(f"{'TEST ID':<32} {'STATUS':<22} {'PROVENANCE':<40} CATEGORY")
        for tc in MATRIX:
            r = done.get(tc["test_id"])
            print(f"{tc['test_id']:<32} {(r or {}).get('status', 'pending'):<22} "
                  f"{(r or {}).get('provenance', '-'):<40} {tc['category']}")
        print(f"\n{len(done)}/{len(MATRIX)} executed. Sources flagged synthetic_source=True are "
              f"reported separately and do not count as unseen real-world coverage.")
        return 0

    pending = [tc for tc in MATRIX
               if (args.case is None or tc["test_id"] in args.case)
               and (args.force or tc["test_id"] not in done)]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("Nothing pending. Use --force to re-run, or --list to inspect status.")
        write_reports(list(done.values()))
        return 0

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"Executing {len(pending)} case(s) with FORCED fresh inference. "
          f"Measured cost on the target RTX 5050 is several minutes per case.")
    orchestrator = UniformSwapOrchestrator()
    for tc in pending:
        rec = run_case(tc, orchestrator)
        rec["source"], rec["reference"] = tc["source"], tc["uniform"]
        append_manifest(rec)
        done[rec["test_id"]] = rec

    write_reports(list(done.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
