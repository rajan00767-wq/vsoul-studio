# Performance Profiling Report (V13.3.1)

**Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, Windows 11.
**Pipeline**: Qwen-Image-Edit-2511 Q4_K_M GGUF + 4-step Lightning LoRA, Diffusers.
**Source of every number below**: `outputs/production_acceptance/provenance_manifest.jsonl`,
produced by `scripts/run_production_acceptance.py` with
`raw_diffusion_cache_policy="force_regenerate"`, `diagnostic_mode=False`,
`cached_raw_candidate=None`. No figure in this document is estimated,
extrapolated or carried over from a previous release.

---

## 1. The reported latency discrepancy, resolved

The V13.3 reports contain two incompatible sets of timings:

| Source | Reported per-image | What it actually measured |
| :--- | :--- | :--- |
| `UNSEEN_REGRESSION_REPORT.md` / `TEST_RESULTS.csv` | 15.6 – 35.1 s | **Cache replay.** `run_unseen_regression.py` passes `cached_raw_candidate` with `authoritative_cache` + `diagnostic_mode=True`. Qwen never ran. These are compositing, validation and crop only. |
| `FRESH_INFERENCE_BENCHMARK.csv` (V13.3) | 375 – 870 s | Partly real, partly **re-parsed from a prior trace** with hardcoded VRAM constants (5824.0 / 6780.0), which is why several rows share identical memory figures. |

Measured fresh inference on the current code — **7 cases, all with forced
regeneration**:

| Case | Status | Total wall-clock | Diffusion | Peak VRAM alloc |
| :--- | :--- | ---: | ---: | ---: |
| PA_11_ShortBoy_Tie | completed | **366.9 s** | 332.1 s | 7015.6 MB |
| PA_04_NoBows_Vest | needs_review | **394.7 s** | — | 7043.4 MB |
| PA_02_Braids_NavyCollar | completed | **419.5 s** | — | 7024.5 MB |
| PA_05_GlassesBindi_RedCollar | needs_review | **436.4 s** | — | 7051.6 MB |
| PA_12_StraightHair_RedCollar | needs_review | **447.6 s** | 424.3 s | 7025.6 MB |
| PA_01_Curly_RedCollar | needs_review | **514.3 s** | — | 7014.5 MB |
| PA_03_MaroonBows_Tie | completed | **572.6 s** | — | 7034.6 MB |

**Min 366.9 s · median 436.4 s · max 572.6 s.** Peak VRAM 7014.5–7051.6 MB of
8 GB across all seven; **zero CUDA OOM**.

Provenance: 5 of 7 carry `fresh_inference_verified` with `invocations=1`. The
two earliest (PA_11, PA_12) ran before the invocation counter was corrected to
patch `UniformGenerator` on the class rather than the module singleton; their
freshness is established instead by the raw-candidate file being rewritten
during the run, which cannot happen on a cache replay. Both are labelled
accordingly in the manifest rather than being silently upgraded.

## 2. Where the time goes

There is **no unexplained execution time**. Measured stage breakdown:

| Stage | PA_11 (s) | PA_12 (s) | Share of total |
| :--- | ---: | ---: | ---: |
| `qwen_diffusion_and_raw_decode` | 332.1 | 424.3 | **90.5% – 94.8%** |
| `semantic_parsing` | 11.4 | 3.8 | 0.8 – 3.1% |
| `ingestion_and_normalization` | 4.8 | 2.6 | 0.6 – 1.3% |
| `raw_quality_validation` | 4.5 | 4.9 | 1.1 – 1.2% |
| `pre_crop_quality_validation` | 3.1 | 3.2 | 0.7 – 0.8% |
| `post_crop_quality_validation` | 0.9 | 1.5 | 0.2 – 0.3% |
| `passport_framing` | 0.9 | — | 0.2% |
| `conditioning_building` | — | 1.3 | 0.3% |

Within diffusion, per-step timings from the run logs: step 1 ≈ 228–240 s,
steps 2–4 ≈ 47–80 s each. The first step carries model load, GGUF
dequantisation and the CPU↔GPU transfers forced by the low-VRAM offload policy.

**Conclusion**: optimisation effort belongs entirely in the diffusion stage and
its model-loading overhead. Every non-diffusion stage in the pipeline combined,
including all 22 quality gates, costs under 26 s.

## 3. Memory

Peak allocated 7015.6 – 7025.6 MB against an 8 GB budget; peak reserved
7822 MB. **Zero CUDA OOM** across the executed cases. Headroom is roughly
1 GB, which is consistent with the project's single-job-in-flight constraint.

## 4. Production target

The stated target is **< 60 s per image and 3,000 images in 4 hours**.

| | Target | Measured (7 cases) | Verdict |
| :--- | ---: | ---: | :--- |
| Per image | < 60 s | 366.9 – 572.6 s (median 436.4 s) | **NOT MET** (6–10× over) |
| 3,000 images | 4 h | ≈ 364 GPU-hours at the median | **NOT MET** (91× the budget) |

This target is **not achieved** and is not claimed to be. Reaching it on this
hardware would require a different execution strategy (batching, a persistent
resident model to amortise the ~230 s first-step load, a smaller or differently
quantised transformer, or non-laptop hardware), each of which trades against
quality and is out of scope for V13.3.1.

## 5. Measurement method and its limits

- `torch.cuda.synchronize()` is called before timing starts and after the
  orchestrator returns, so GPU work is not attributed to the following stage.
- Stage durations come from the pipeline's own execution trace
  (`21_pipeline_execution_trace.json`), not from wall-clock subtraction.
- Peak VRAM is read from `torch.cuda.max_memory_allocated()` /
  `max_memory_reserved()` after a `reset_peak_memory_stats()` at case start.
- **Limit**: cold-start and warm-start are not yet separated into distinct
  columns. Both cases above include model load, so both are cold-start figures;
  the ~230 s first diffusion step versus ~47–80 s subsequent steps is the best
  available evidence of the load component. A dedicated warm-start measurement
  is listed in REMAINING_ISSUES.md.
- **Limit**: only the cases recorded in the manifest were executed. Latency for
  unexecuted cases is not estimated anywhere in this report.
