"""
FastAPI Application Entry Point.
AI pipeline runs in ai_worker.py (separate process, managed by Supervisor).
This process handles only HTTP requests, WebSocket streaming, and cleanup.
"""

import asyncio
import gc
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils.job_eta import estimate_job_time
from fastapi.responses import FileResponse

# Transformers 5.x imports the optional `kernels` package whenever it is
# installed. On this managed Windows machine its native kernels_data DLL is
# blocked by Application Control, although none of this application's models
# require Hub Kernels. Mark that optional integration unavailable before the
# API imports SCHP/Qwen modules so server startup remains deterministic.
os.environ.setdefault("USE_HUB_KERNELS", "NO")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics"))
# This process runs every queued job back to back in one CUDA context, so the
# caching allocator's fragmentation accumulates across jobs. Job two then
# failed with "CUDA out of memory. Tried to allocate 80.00 MiB ... 371.77 MiB
# is reserved by PyTorch but unallocated" on an otherwise idle 8 GB card. The
# isolated prompt-encoder worker already sets this; the server that actually
# runs diffusion did not. It must be set before torch is first imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    import transformers.utils.import_utils as _transformers_import_utils
    _transformers_import_utils.is_kernels_available = lambda *args, **kwargs: False
except Exception:
    pass

from api.routes import router as api_router, orchestrator
from utils.logger import get_logger
import job_store as _js

logger = get_logger(__name__)

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ─── Background tasks (API process only) ──────────────────────────────────────

async def _cleanup_loop():
    """Every 30 min: delete old files and expired job records in Redis."""
    while True:
        await asyncio.sleep(1800)
        # Keep completed portraits for seven days. Some Windows filesystems
        # report mtimes with a one-day clock skew relative to Python's epoch;
        # a 24-hour cutoff could therefore delete a result immediately after
        # generation and leave a completed job pointing at a missing file.
        cutoff = time.time() - (7 * 86400)

        # Delete intermediate step files — but only ones older than an hour.
        # Deleting unconditionally raced with in-flight jobs: a cleanup tick
        # could remove _bg.png/_enhanced.png between two pipeline stages,
        # leaving the job "done" pointing at a file that no longer exists.
        intermediate_cutoff = time.time() - 3600
        intermediate_suffixes = (
            "_enhanced.png", "_face.png", "_bg.png", "_leveled.png",
            "_upscaled.png", "_enhanced.jpg", "_face.jpg"
        )
        for f in list(OUTPUTS_DIR.iterdir()):
            try:
                if (
                    f.is_file()
                    and any(f.name.endswith(s) for s in intermediate_suffixes)
                    and f.stat().st_mtime < intermediate_cutoff
                ):
                    f.unlink()
            except OSError:
                pass

        # Delete old uploads / outputs
        for directory in (UPLOADS_DIR, OUTPUTS_DIR):
            for f in directory.iterdir():
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass

        # Purge old Redis job records
        try:
            stale = []
            for jid in _js.scan_all_jobs():
                j = _js.get_job(jid)
                if j and j.get("status") in ("done", "failed", "timeout"):
                    if j.get("_ts", time.time()) < cutoff:
                        stale.append(jid)
            for jid in stale:
                _js.delete_job(jid)
            if stale:
                logger.info("[CLEANUP] Removed %d stale job records from Redis", len(stale))
        except Exception as e:
            logger.warning("[CLEANUP] Redis scan error: %s", e)

        # Purge old ZIP files
        zip_cutoff = time.time() - 86400
        for f in OUTPUTS_DIR.iterdir():
            try:
                if f.is_file() and f.name.startswith("zip_") and f.stat().st_mtime < zip_cutoff:
                    f.unlink()
            except OSError:
                pass


# ─── Dedicated Single-Worker Queue Loop (1 Process/Job at a Time) ──────────────

async def _execute_queued_job(job_id: str, job: dict):
    """Executes a queued job strictly 1-by-1 sequentially."""
    metadata = job.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    job_type = metadata.get("type", "orchestrator")
    loop = asyncio.get_event_loop()

    try:
        if job_type == "qwen_enhance":
            def _cb(n, desc="", status="processing", stage="portrait_processing"):
                current_job = _js.get_job(job_id)
                if current_job and current_job.get("status") == "cancelled":
                    raise JobCancelledError("Enhancement cancelled by user")
                pct = int(round((n * 100) if isinstance(n, float) and n <= 1 else n))
                current_progress = int((current_job or {}).get("progress") or 0)
                _js.update_job(
                    job_id,
                    status=status,
                    # Pipeline stages can overlap slightly; never make the
                    # user-facing percentage move backwards.
                    progress=min(99, max(1, current_progress, pct)),
                    stage=stage,
                    message=desc or "Enhancing portrait…",
                    _last_update=str(time.time()),
                )

            _cb(0.05, "Preparing portrait…")

            input_path = job.get("input_path")
            upscale_factor = int(metadata.get("upscale_factor", 1))
            orig_filename = job.get("original_filename") or metadata.get("download_name") or "photo.png"

            from gradio_app import (
                ENGINE_PORTRAIT_RECOMMENDED,
                map_client_background,
                process_single_enhance,
            )

            # Physical output dimensions are fixed in the enhancement route;
            # the user-selected studio backdrop remains available.
            background_color = metadata.get("background_color", "white")
            bg_name, custom_hex = map_client_background(str(background_color))

            result = await loop.run_in_executor(
                None,
                lambda: process_single_enhance(
                    input_path,
                    ENGINE_PORTRAIT_RECOMMENDED,
                    "Standard ICAO Passport (35 x 45 mm / 3.5 x 4.5 cm)",
                    True,
                    True,
                    bg_name,
                    custom_hex,
                    1.12,
                    True,
                    True,
                    upscale_factor,
                    # Enhancement uses the versioned backend prompt only.
                    # Ignore values from older cached browser pages.
                    "",
                    _cb,
                    int(metadata.get("steps", 20)),
                    prompt_mode=str(metadata.get("prompt_mode") or "dynamic"),
                ),
            )
            named_path, msg, _candidate_path = (*result, None)[:3]
            if not named_path:
                raise RuntimeError(msg or "Enhancement failed")

            if (_js.get_job(job_id) or {}).get("status") == "cancelled":
                return

            # The pipeline output is keyed by job id.  Do not copy it onto the
            # original filename: that mutable URL can show a prior browser-cached
            # result while a newer job has already completed.
            rel_out = f"outputs/{Path(named_path).name}"
            _js.update_job(
                job_id,
                status="done",
                stage="done",
                progress=100,
                message=msg or "✓ Enhancement Complete",
                output_path=rel_out,
                result_url=f"/{rel_out}",
                original_filename=orig_filename,
                completed_at=str(time.time()),
            )
            logger.info("[QUEUE_WORKER_DONE] Qwen Enhance job=%s -> %s", job_id, named_path)

        elif job_type == "uniform_swap":
            _js.update_job(
                job_id,
                status="processing",
                stage="uniform_merge",
                progress=8,
                message="Fitting school uniform…",
                started_at=str(time.time()),
            )
            person_path = job.get("input_path")
            uniform_path = metadata.get("uniform_path")
            orig_name = job.get("original_filename") or metadata.get("download_name") or "person.png"
            bg_color = metadata.get("bg_color") or "4,126,246"
            generation_mode = str(metadata.get("generation_mode") or metadata.get("profile") or "quality")

            # qwen_vl_image_edit_uniform_swap pipeline execution
            from pipelines.uniform_swap import UniformSwapConfig, uniform_swap_orchestrator

            # Authoritative configuration propagation from job metadata
            config = UniformSwapConfig.from_job_metadata(
                metadata,
                requested_width=metadata.get("width"),
                requested_height=metadata.get("height"),
            )
            logger.info(
                "[QUEUE_WORKER] Uniform swap job=%s config: strategy=%s, latent_lock=%s, rgb_rest=%s, steps=%d, res=%s, bg=%s",
                job_id, config.conditioning_strategy, config.latent_identity_lock, config.rgb_identity_restoration,
                config.steps, config.generation_resolution, config.background_color
            )

            def _cb(n, desc=""):
                pct = int(round((n * 100) if n <= 1 else n))
                current_job = _js.get_job(job_id)
                # See _uniform_cb above: a progress tick that unconditionally
                # writes status="processing" silently undoes a user's cancel.
                if current_job and current_job.get("status") == "cancelled":
                    raise JobCancelledError("Uniform swap cancelled by user")
                _js.update_job(
                    job_id,
                    status="processing",
                    progress=min(99, max(8, pct)),
                    message=desc or "Fitting school uniform…",
                    _last_update=str(time.time()),
                )

            delivery_res = await loop.run_in_executor(
                None,
                lambda: uniform_swap_orchestrator.run(
                    person_path,
                    uniform_path,
                    config=config,
                    job_id=f"qwen_uniform_{job_id[:8]}",
                    progress_callback=_cb,
                ),
            )
            if not delivery_res:
                raise RuntimeError("Qwen Image Edit uniform swap failed")

            # Task 7: Correct production delivery gating
            if hasattr(delivery_res, "passed") and not delivery_res.passed:
                diag_name = delivery_res.diagnostic_path.name if delivery_res.diagnostic_path else "candidate.png"
                failure_msg = f"Needs review: {'; '.join(delivery_res.defects)}" if delivery_res.defects else "Uniform swap flagged for review"
                logger.warning("[QUEUE_WORKER] Job %s gated for review: %s", job_id, failure_msg)
                import json as _json
                stage_timings = getattr(delivery_res, "trace", {}).get("stages", {})
                exec_time = getattr(delivery_res, "trace", {}).get("total_duration_seconds", 0)
                _js.update_job(
                    job_id,
                    status="needs_review",
                    stage="needs_review",
                    progress=100,
                    message=failure_msg,
                    source_quality_notes=_json.dumps(delivery_res.defects),
                    diagnostic_path=f"outputs/debug_phases/{diag_name}",
                    output_path="",
                    result_url="",
                    defects=delivery_res.defects,
                    stage_timings=stage_timings,
                    execution_time=exec_time,
                )
                return

            named_path = delivery_res.delivered_path if hasattr(delivery_res, "delivered_path") and delivery_res.delivered_path else Path(delivery_res)
            rel_out = f"outputs/{Path(named_path).name}"
            # Surface measured ceilings on this upload to whoever submitted
            # it. A log line reaches the operator of the server, not the
            # person who chose the photo, and these two limits are properties
            # of the upload that no pipeline stage can recover: the only fix
            # is a different source image.
            quality_notes = []
            import json as _json
            try:
                audit_path = Path("outputs") / (
                    Path(named_path).name.replace("_uniform.png", "_instructions.json")
                )
                if audit_path.exists():
                    audit = _json.loads(audit_path.read_text(encoding="utf-8"))
                    if audit.get("source_detail_sufficient") is False:
                        quality_notes.append(
                            "the uploaded face is smaller than the delivered frame, "
                            "so facial detail stays soft"
                        )
                    if audit.get("torso_evidence_sufficient") is False:
                        quality_notes.append(
                            "the upload shows little chest below the chin, "
                            "so the body fit is approximate"
                        )
            except Exception as note_error:
                logger.warning(
                    "[QUEUE_WORKER] Could not read source-quality notes: %s",
                    note_error,
                )
            # Defects measured on the delivered portrait itself. These used to
            # reach only the server log, so the person who uploaded the photo
            # was never told the render had a known problem - job 5a1bbeaa
            # shipped a visibly stretched neck with a clean completion
            # message. The review stage always delivers, so a note is the
            # only way the user learns to retry or pick another photo.
            try:
                review_path = Path("outputs") / (
                    Path(named_path).name.replace(
                        "_uniform.png", "_visual_review.json",
                    )
                )
                if review_path.exists():
                    review = _json.loads(review_path.read_text(encoding="utf-8"))
                    if "neck_stretched_measured" in (
                        review.get("advisory_failures") or []
                    ):
                        quality_notes.append(
                            "the fitted collar sits lower than in your photo, "
                            "so the neck looks longer than it should"
                        )
                    for warning, text in (
                        ("head_cropped", "the top of the head is clipped"),
                        ("face_artifacts", "the face shows rendering artifacts"),
                        ("harsh_hair_glare", "bright glare remains on the hair"),
                    ):
                        if warning in (review.get("quality_warnings") or []):
                            quality_notes.append(text)
            except Exception as review_error:
                logger.warning(
                    "[QUEUE_WORKER] Could not read delivered-portrait review: %s",
                    review_error,
                )
            completion_message = "School uniform fitting complete!"
            if quality_notes:
                completion_message += (
                    " Note: " + "; ".join(quality_notes)
                    + ". A higher-resolution half-body photo gives a better result."
                )
            stage_timings = getattr(delivery_res, "trace", {}).get("stages", {})
            exec_time = getattr(delivery_res, "trace", {}).get("total_duration_seconds", 0)
            _js.update_job(
                job_id,
                status="done",
                stage="done",
                progress=100,
                message=completion_message,
                source_quality_notes=_json.dumps(quality_notes) if quality_notes else "",
                output_path=rel_out,
                result_url=f"/{rel_out}",
                original_filename=orig_name,
                completed_at=str(time.time()),
                stage_timings=stage_timings,
                execution_time=exec_time,
            )
            logger.info("[QUEUE_WORKER_DONE] Uniform job=%s -> %s", job_id, named_path)

        else:
            # Standard pipeline job
            config = job.get("config", {})
            if isinstance(config, str):
                import json
                try:
                    config = json.loads(config)
                except Exception:
                    config = {}

            def _cb(progress, message, status="processing", stage="", extra=None):
                # Same cooperative-cancel contract as the uniform routes: stop
                # at the next progress tick instead of writing "processing"
                # back over the user's cancel and running to completion.
                current_job = _js.get_job(job_id)
                if current_job and current_job.get("status") == "cancelled":
                    raise JobCancelledError("Enhancement cancelled by user")
                fields = {
                    "status": status,
                    "progress": int(progress),
                    "message": message,
                    "_last_update": str(time.time()),
                }
                if stage:
                    fields["stage"] = stage
                if extra:
                    fields.update(extra)
                _js.update_job(job_id, **fields)

            _js.update_job(
                job_id,
                status="processing",
                stage="init",
                progress=10,
                message="Enhancing photo — Stage: Initializing",
                started_at=str(time.time()),
            )

            output_path = await orchestrator.run(
                job_id=job_id,
                input_path=job["input_path"],
                config=config,
                progress_callback=_cb,
            )

            rel_out = f"outputs/{Path(output_path).name}"
            _js.update_job(
                job_id,
                status="done",
                stage="done",
                progress=100,
                message="Enhancement complete",
                output_path=rel_out,
                result_url=f"/{rel_out}",
                completed_at=str(time.time()),
            )
            logger.info("[QUEUE_WORKER_DONE] Orchestrator job=%s -> %s", job_id, output_path)

    except JobCancelledError:
        _js.remove_from_queue(job_id)
        logger.info("[QUEUE_WORKER] Job %s cancelled during processing", job_id)
        return
    except Exception as exc:
        logger.exception("[QUEUE_WORKER] Job %s failed: %s", job_id, exc)
        _js.remove_from_queue(job_id)
        if (_js.get_job(job_id) or {}).get("status") == "cancelled":
            return
        _js.update_job(
            job_id,
            status="failed",
            stage="failed",
            progress=0,
            message=f"Processing failed: {str(exc)[:120]}",
            error_message=str(exc),
            completed_at=str(time.time()),
        )
    finally:
        _release_cuda_between_jobs(job_id)


def _release_cuda_between_jobs(job_id: str) -> None:
    """Drop this job's GPU allocations before the next one starts.

    Every queued job shares one CUDA context in this process, and the uniform
    route runs ~1400 lines of post-processing after diffusion without releasing
    anything. Job two then died with "CUDA out of memory. Tried to allocate
    80.00 MiB" while 5.56 GiB was still allocated on an idle card. Reclaiming at
    the job boundary keeps a long-running server from degrading into a restart.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        before = torch.cuda.memory_allocated()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        logger.info(
            "[QUEUE_WORKER] Released GPU memory after job %s: %.2f GiB -> %.2f GiB",
            job_id, before / 1024 ** 3, after / 1024 ** 3,
        )
    except Exception as cleanup_error:
        # Never let reclamation failure mask the job's own outcome.
        logger.warning(
            "[QUEUE_WORKER] GPU cleanup after job %s failed: %s", job_id, cleanup_error,
        )


async def _queue_worker_loop():
    """
    Dedicated FIFO / Priority Single-Worker Queue Loop.
    Executes strictly ONE job at a time sequentially.
    """
    logger.info("[QUEUE_WORKER] Dedicated 1-by-1 Queue Worker running...")
    loop = asyncio.get_event_loop()
    while True:
        try:
            _js.worker_heartbeat()

            # Pop next job from queue with 1s timeout
            job_id = await loop.run_in_executor(None, _js.pop_from_queue, 1)
            if not job_id:
                await asyncio.sleep(0.5)
                continue

            job = await loop.run_in_executor(None, _js.get_job, job_id)
            if not job:
                continue

            if job.get("status") in ("done", "failed", "cancelled"):
                _js.remove_from_queue(job_id)
                continue

            logger.info("[QUEUE_WORKER] Processing job %s strictly 1-by-1", job_id)
            await _execute_queued_job(job_id, job)

        except asyncio.CancelledError:
            logger.info("[QUEUE_WORKER] Queue worker task stopped.")
            break
        except Exception as e:
            logger.exception("[QUEUE_WORKER] Worker loop error: %s", e)
            await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] AI Image Platform (API) starting up...")
    logger.info("[STARTUP] Upload dir: %s", UPLOADS_DIR.absolute())
    logger.info("[STARTUP] Output dir: %s", OUTPUTS_DIR.absolute())
    logger.info("[STARTUP] Redis: %s", "OK" if _js.health_check() else "UNAVAILABLE")
    logger.info("[STARTUP] Queue length on start: %d", _js.queue_length())

    # Pre-warm CPU/RAM models on startup so they are not loaded repeatedly
    # during jobs. On memory-constrained launches this can be disabled without
    # changing the pipeline; both services then initialize lazily on first use.
    skip_cpu_prewarm = os.getenv("VSOUL_SKIP_CPU_PREWARM", "0").strip().lower() in {
        "1", "true", "yes",
    }
    if skip_cpu_prewarm:
        logger.info("[STARTUP] CPU model pre-warm disabled; models will load lazily")
    else:
        try:
            import numpy as np
            from pipelines.schp_service import parse as parse_schp
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            parse_schp(dummy)
            logger.info("[STARTUP] SCHP model pre-warmed in CPU RAM")
        except Exception as exc:
            logger.warning("[STARTUP] SCHP pre-warm skipped: %s", exc)

        try:
            from pipelines.photo_restoration import _get_insight_app
            _get_insight_app()
            logger.info("[STARTUP] InsightFace buffalo_l pre-warmed in CPU RAM")
        except Exception as exc:
            logger.warning("[STARTUP] InsightFace pre-warm skipped: %s", exc)
    cleanup_task = asyncio.create_task(_cleanup_loop())
    worker_task = asyncio.create_task(_queue_worker_loop())
    yield
    cleanup_task.cancel()
    worker_task.cancel()
    logger.info("[SHUTDOWN] API process shutting down...")



app = FastAPI(
    title="AI Image Enhancement Platform",
    description="Fully self-hosted AI image enhancement",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
async def serve_frontend():
    # The UI is a single local HTML document that changes frequently during
    # development. Chrome can otherwise keep a stale/empty renderer document
    # across a server restart while health polling continues successfully.
    # Always serve the current file and force revalidation of the page itself;
    # generated images remain independently cacheable under /outputs.
    return FileResponse(
        "index.html",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


app.include_router(api_router, prefix="/api")


# ─── WebSocket: stream job progress from Redis ────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    Streams job progress to the connected client by polling Redis every 1.5s.
    Closes automatically when job reaches 'done' or 'failed'.
    """
    await websocket.accept()
    last_progress = -1
    last_status   = ""
    loop = asyncio.get_event_loop()

    try:
        for _ in range(500):   # max ~12.5 minutes
            job = await loop.run_in_executor(None, _js.get_job, job_id)

            if not job:
                await websocket.send_json({"type": "error", "message": "Job not found"})
                break

            status   = job.get("status", "queued")
            progress = int(job.get("progress", 0))

            if progress != last_progress or status != last_status:
                payload = {
                    "type":     "progress",
                    "progress": progress,
                    "message":  job.get("message", ""),
                    "stage":    job.get("stage", ""),
                    "status":   status,
                }

                if status == "done":
                    payload["type"]        = "complete"
                    payload["output_path"] = job.get("output_path", "")
                    payload["output_url"]  = job.get("result_url") or ("/" + job.get("output_path", "") if job.get("output_path") else "")
                elif status in ("failed", "timeout"):
                    payload["type"]    = "error"
                    payload["message"] = job.get("error_message") or job.get("message", "Enhancement failed. Retry?")

                # Add queue position for queued jobs
                if status == "queued":
                    payload["queue_position"] = await loop.run_in_executor(
                        None, _js.queue_position, job_id
                    )

                payload.update(estimate_job_time(job, payload.get("queue_position", 0)))

                await websocket.send_json(payload)
                last_progress = progress
                last_status   = status

                if status in ("done", "failed", "timeout"):
                    break

            await asyncio.sleep(1.5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error for job %s: %s", job_id, e)


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":       "ok",
        "server_status": "healthy" if _js.health_check() else "degraded",
        "timestamp":    time.time(),
        "version":      "2.0.0",
        "gpu_available": _check_gpu(),
        "models_loaded": _check_models(),
        "redis":        _js.health_check(),
        "queue_length": _js.queue_length(),
        "active_workers": _js.active_workers(),
        "max_workers": int(__import__("os").getenv("AI_MAX_WORKERS", "1")),
        "average_processing_time": round(_js.average_processing_time(), 1),
        "estimated_wait_time": _js.estimated_wait_time(max_workers=int(__import__("os").getenv("AI_MAX_WORKERS", "1"))),
        "job_counts": _js.job_counts(),
    }


def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_models() -> dict:
    model_dirs = ["sdxl", "realesrgan", "gfpgan", "controlnet", "sam"]
    status = {}
    for m in model_dirs:
        path = Path("models") / m
        status[m] = path.exists() and any(path.iterdir()) if path.exists() else False
    return status


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

class JobCancelledError(RuntimeError):
    """Raised by a progress callback when the user cancels a running job."""
