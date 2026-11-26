# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (all commands below assume it, or call .venv/Scripts/python.exe directly)
.\.venv\Scripts\activate

# Run the app — FastAPI + the index.html UI on http://127.0.0.1:8000
python main.py
# API docs: /api/docs   Health: /health and /api/health

# Alternative: Gradio lab UI mounted at /lab on port 7860 (same FastAPI app)
python gradio_app.py

# Download only the model assets this workspace uses (~Qwen Edit GGUF, VL, BiRefNet, SCHP, RealESRGAN)
python download_models.py

# All tests
.venv/Scripts/python.exe -m unittest discover -s tests

# One test module / one test
.venv/Scripts/python.exe -m unittest tests.test_uniform_composite
.venv/Scripts/python.exe -m unittest tests.test_uniform_composite.CompositeTests.test_preserves_head_and_places_garment

# _run_tests.py does the same but pre-disables transformers' optional `kernels`
# integration — use it if a suite fails on import with a kernels/DLL error.
.venv/Scripts/python.exe _run_tests.py tests.test_uniform_prompt

# Run one real static_test uniform swap end-to-end against the working tree
# (fixed uploads from job fc7db399, no server) — the only way to exercise the
# full GPU path, since tests never load models.
.venv/Scripts/python.exe _run_static_job.py [job_id]
```

All relative paths (`models/`, `uploads/`, `outputs/`, `scratch/`, `config/`) resolve from the working directory — always run from the project root.

## Architecture

Single FastAPI process. There is no separate worker process despite docstrings in [main.py](main.py) and [api/routes.py](api/routes.py) referring to `ai_worker.py` — that file does not exist here; inference runs in-process on a thread executor.

### Request lifecycle

1. [index.html](index.html) (2k-line single-file UI, no build step) posts to `/api/qwen/enhance` or `/api/uniform`, then polls `/api/job/{id}`. A `/ws/{job_id}` WebSocket also exists (polls Redis every 1.5s) but the shipped UI uses polling.
2. [api/routes.py](api/routes.py) writes the job record and pushes onto a priority queue. It does **not** run the pipeline.
3. [main.py](main.py) `_queue_worker_loop()` pops one job at a time and dispatches on `metadata["type"]` in `_execute_queued_job()` — strictly one job in flight, because one 8–12 GB GPU cannot hold two Qwen jobs.
4. Heavy work is wrapped in `loop.run_in_executor(None, ...)` so the event loop keeps serving status polls.

Job types dispatched in `_execute_queued_job`:

| `metadata["type"]` | Handler |
|---|---|
| `qwen_enhance` | `gradio_app.process_single_enhance` |
| `qwen_uniform_swap` | `QwenEditPipeline.qwen_vl_image_edit_uniform_swap` |
| `uniform_swap` | same, via the module-level `qwen_service` singleton |
| anything else | `PipelineOrchestrator.run` (legacy SDXL/CodeFormer chain) |

### Job store

[job_store.py](job_store.py) is Redis-backed (`ai:job:{id}` hashes, `ai:queue:{high,normal,bulk}` lists) with a **transparent in-memory fallback** — if Redis is not running everything still works, scoped to the process. Every `get_job`/`update_job` tries Redis first and silently degrades. `health_check()` returning False means in-memory mode, not an outage.

Progress callbacks are the only channel for status; they clamp percentage monotonically (`min(99, max(current, pct))`) and raise `JobCancelledError` when a client cancels mid-generation.

### Model execution strategy (the core constraint)

Target hardware is a ~8 GB laptop GPU, so [pipelines/qwen_edit_pipeline.py](pipelines/qwen_edit_pipeline.py) never keeps all three Qwen components resident:

- **Transformer**: Qwen-Image-Edit-2511 as a Q4_K_M **GGUF** (`models/qwen_image_edit/`), with the architecture config pinned in-repo at [config/qwen_image_edit_2511_transformer.json](config/qwen_image_edit_2511_transformer.json) — the 2511 GGUF needs the 64-channel config, and a stale 128-channel cached config leaves parameters on `meta`.
- **Lightning LoRA**: 4-step adapter is the shipped path. The 8-step file exists but is deliberately unused — 8 steps disables the Lightning schedule.
- **Prompt encoding** runs in a **throwaway subprocess** (`_encode_prompt_isolated`), because bitsandbytes NF4 leaks VRAM that only process exit reclaims. Falls back from 4-bit to a BF16 GPU/CPU split if the native kernel access-violates. Results land in a content-hashed cache under `scratch/cache/prompt_embeddings/`.
- **VL analysis** ([pipelines/uniform_vl_analyzer.py](pipelines/uniform_vl_analyzer.py)) uses the same subprocess-and-exit pattern. Its cache key includes `VL_CACHE_VERSION` — **bump that constant whenever you change a VL prompt or a normalization rule**, or identical uploads will silently reuse the old (wrong) "vest"/"bow" reading.
- VAE decode runs on CPU; `transformers`' optional `kernels` package is force-disabled at import in [main.py](main.py) (blocked DLL on the target Windows machine).
- [config/opt_pol.yaml](config/opt_pol.yaml) documents the low/mid/high-VRAM offload policies.

### Uniform swap pipeline

`qwen_vl_image_edit_uniform_swap` is the product's centerpiece (most of the 14k lines in `qwen_edit_pipeline.py`). Shape:

1. Heal sensor/scan defects, keep an untouched copy of the upload as the **sole** biometric and hair authority.
2. `_prepare_uniform_reference` — VL and Qwen must see the *same* prepared template; analysing a padded source while conditioning on a later crop produced contradictory fit plans.
3. VL analysis → `vl_plan` dict (collar geometry, fabric color, badge bbox, hair accessories, measured sunlight severity). The prompts in `UniformVLAnalyzer` are extremely literal on purpose — collar type from edges not pattern, tied-hair sections counted per viewer side.
4. Rough composite guide ([pipelines/uniform_composite.py](pipelines/uniform_composite.py)), pre-pass in [uniform_precomposite.py](pipelines/uniform_precomposite.py), using SCHP human parsing ([schp_service.py](pipelines/schp_service.py)) fused with BiRefNet matting ([birefnet_service.py](pipelines/birefnet_service.py)). [uniform_source_protection.py](pipelines/uniform_source_protection.py) carries the untouched-source guarantees forward.
5. One 4-step Qwen Image Edit pass with chunked transformer streaming.
6. Post-generation review and repair: `_has_acceptable_portrait_identity` / `check_uniform_identity` ([uniform_review.py](pipelines/uniform_review.py)) gate the candidate; `_lock_uniform_source_head` pastes the source head back only when identity drifted. Then badge cleanup ([uniform_badge_cleanup.py](pipelines/uniform_badge_cleanup.py)), backdrop normalization, accessory/hair restoration, tone finishing ([uniform_finishing.py](pipelines/uniform_finishing.py), [face_masked_retinex.py](pipelines/face_masked_retinex.py)).
7. Final deliverable is `outputs/{job_id}_uniform.png` at 300 DPI; `_cleanup_successful_uniform_artifacts` deletes the dozens of intermediate debug PNGs on success and keeps them on failure.

`prompt_mode` is `"dynamic"` (default) or `"static_test"` — static reuses a cached VL plan plus a fixed HEAD LOCK/HAIR LOCK prompt and a lower CFG for a more source-faithful, reproducible result. The UI's selector maps "Simple Static Prompt" → `static_test`.

### Legacy / unused code

- [pipelines/orchestrator.py](pipelines/orchestrator.py) (2.8k lines, SDXL + CodeFormer/GFPGAN + Real-ESRGAN + rembg) is instantiated at import of `routes.py` and still serves `/api/uniform`, `/api/sdxl/*`, `/api/analyze`. It is a different, older product than the Qwen path and its `models/sdxl` weights are not in `download_models.py`.
- `qwen_edit_uniform_swap` (end of `qwen_edit_pipeline.py`) is marked legacy and unreachable.
- `catvton_pipeline.py`, `gusuq_pipeline.py`, `studio_uniform_pipeline.py`, `vton_compositor.py` are only reachable from `gradio_app.py` / lazy fallbacks, not from the main HTTP path.
- `_diag_*.py` (~50 files in root) are one-off throwaway investigations, not part of the app.

## Conventions

- Tests are plain `unittest` with synthetic `PIL.Image` fixtures and numeric pixel assertions (`assertLess(max(abs(a-b) ...))`) — no pytest, no fixtures framework, no model loading. New image-logic changes get a test in that style.
- Comments in this codebase explain *why a specific failure drove the code*, often naming the symptom (stale browser cache, cleanup racing an in-flight job, Qwen inventing scenery). Preserve them when editing nearby; they are regression notes.
- Logs are tagged with bracketed topics (`[QUEUE_WORKER]`, `[UNIFORM_IDENTITY_REVIEW]`, `[STATIC_PROMPT]`, `[QWEN_ENCODE_CACHE]`) — grep by tag when debugging a run.
- `models/`, `uploads/`, `outputs/`, `scratch/cache/`, `external/`, `third_party/`, and all weight extensions are gitignored. `scratch/` holds debug images and ad-hoc trigger scripts.
- `/` is served with `no-store` deliberately: Chrome kept a stale empty `index.html` renderer across server restarts while health polling still succeeded.
