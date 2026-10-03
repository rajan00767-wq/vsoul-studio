"""
API Routes - All REST endpoints for the AI platform.
Job state is stored in Redis (via job_store.py).
AI pipeline is processed by ai_worker.py (separate process).
"""

import uuid
import os
import io
import time
import zipfile
import asyncio
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pipelines.orchestrator import PipelineOrchestrator
from pipelines.sdxl_pipeline import SDXLPipeline
from utils.logger import get_logger
import job_store as _js

logger = get_logger(__name__)
router = APIRouter()

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")

orchestrator = PipelineOrchestrator()
sdxl = SDXLPipeline()

# Max upload: 20 MB
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Max jobs allowed in queue at once (prevents runaway pile-up)
_MAX_QUEUE_LENGTH = 20


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mem_mb() -> int:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
    except Exception:
        return 0


# ─── Models ───────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Optional[dict] = None


class EnhanceRequest(BaseModel):
    style_mode: str = "professional"
    upscale_factor: int = 2
    face_restore: bool = True
    background_replace: bool = False
    background_prompt: str = ""
    background_color: str = "red"
    output_width_px: int = 413
    output_height_px: int = 531
    output_dpi: int = 300
    margin_top_px: int = 24
    margin_bottom_px: int = 24
    margin_left_px: int = 18
    margin_right_px: int = 18
    denoise: bool = True
    sharpen: bool = True
    relight: bool = False
    passport_compose: bool = True
    head_ratio: float = 0.50
    face_restore_if_needed: bool = False
    upscale_if_needed: bool = False


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def api_health():
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            name = _torch.cuda.get_device_name(0)
            gpu = {"available": True, "type": "cuda", "name": name}
        elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            gpu = {"available": True, "type": "mps", "name": "Apple Silicon MPS"}
        else:
            gpu = {"available": False, "type": "cpu", "name": "CPU only"}
    except ImportError:
        gpu = {"available": False, "type": "unknown", "name": "torch not installed"}
    counts  = _js.job_counts()
    avg_t   = _js.average_processing_time()
    queued  = counts.get("queued", 0)
    active  = counts.get("processing", 0)
    qc      = _js.queue_counts()
    return {
        "status": "ok",
        "server_status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "gpu": gpu,
        "redis": _js.health_check(),
        "queue_length": _js.queue_length(),
        "queue_counts": qc,
        "active_jobs": active,
        "active_workers": _js.active_workers(),
        "max_workers": 1,
        "average_processing_time": avg_t,
        "estimated_wait_time": queued * avg_t,
        "queued_jobs": queued,
        "completed_jobs_24h": counts.get("done", 0),
        "failed_jobs_24h": counts.get("failed", 0),
    }

# ─── Upload Endpoint ──────────────────────────────────────────────────────────

@router.post("/upload", response_model=dict)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    style_mode: str = Form("professional"),
    upscale_factor: int = Form(2),
    face_restore: bool = Form(True),
    background_replace: bool = Form(False),
    background_prompt: str = Form(""),
    repair_prompt: str = Form(""),
    background_color: str = Form("white"),  # Professional white background (was: red)
    output_width_px: int = Form(413),
    output_height_px: int = Form(531),
    output_dpi: int = Form(300),
    margin_top_px: int = Form(24),
    margin_bottom_px: int = Form(24),
    margin_left_px: int = Form(18),
    margin_right_px: int = Form(18),
    denoise: bool = Form(True),
    sharpen: bool = Form(True),
    relight: bool = Form(False),
    relight_engine: str = Form("classical"),      # "classical" | "ic_light"
    relight_prompt: str = Form(""),
    relight_strength: float = Form(0.6),
    background_engine: str = Form("studio"),       # "studio" | "identity_studio" | "controlnet"
    controlnet_scale: float = Form(0.6),
    passport_compose: bool = Form(True),
    head_ratio: float = Form(0.50),
    face_restore_if_needed: bool = Form(False),
    upscale_if_needed: bool = Form(False),
    ai_planner: bool = Form(True),
    studio_regenerate_full: bool = Form(False),
    priority: str = Form("normal"),                # "normal" | "bulk"
    background_tasks: BackgroundTasks = None,
):
    logger.info("[UPLOAD_START] file=%s type=%s", file.filename, file.content_type)

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, f"Not an image: {file.content_type}")

    # Global queue depth guard
    q_len = _js.queue_length()
    if q_len >= _MAX_QUEUE_LENGTH:
        raise HTTPException(503,
            f"Server queue is full ({q_len} jobs). Please try again in a few minutes.")

    job_id  = str(uuid.uuid4())
    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large: {len(content)//1024} KB (max 20 MB)")

    # Normalise to PNG via PIL, cap large inputs to avoid memory/timeout issues
    _MAX_INPUT_PX = 1500  # longest edge; AI upscales to 630×810 anyway
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(io.BytesIO(content)) as pil_img:
            pil_img.load()
            rgb = pil_img.convert("RGB")
            img_w, img_h = rgb.size
            if max(img_w, img_h) > _MAX_INPUT_PX:
                scale = _MAX_INPUT_PX / max(img_w, img_h)
                rgb = rgb.resize((int(img_w * scale), int(img_h * scale)), _PILImage.LANCZOS)
                img_w, img_h = rgb.size
                logger.info("[UPLOAD_RESIZE] Downscaled large input to %dx%d", img_w, img_h)
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")

    input_path = UPLOADS_DIR / f"{job_id}.png"
    rgb.save(str(input_path), "PNG")

    logger.info("[UPLOAD_COMPLETE] job=%s size_kb=%d px=%dx%d mem_mb=%d",
                job_id, len(content) // 1024, img_w, img_h, _mem_mb())

    # Build job record
    job_data = {
        "job_id":      job_id,
        "status":      "queued",
        "progress":    0,
        "message":     "Job queued — waiting for AI worker",
        "stage":       "queued",
        "input_path":  str(input_path),
        "output_path": None,
        "metadata":    {"original_filename": file.filename, "width": img_w, "height": img_h},
        "_ts":         time.time(),
        "_timing":     {},
        "config": {
            "style_mode":        style_mode,
            "upscale_factor":    upscale_factor,
            "face_restore":      face_restore,
            "background_replace": background_replace,
            "background_prompt": background_prompt,
            "repair_prompt":     repair_prompt[:1200],
            "background_color":  background_color,
            "output_width_px":   output_width_px,
            "output_height_px":  output_height_px,
            "output_dpi":        output_dpi,
            "margin_top_px":     margin_top_px,
            "margin_bottom_px":  margin_bottom_px,
            "margin_left_px":    margin_left_px,
            "margin_right_px":   margin_right_px,
            "denoise":           denoise,
            "sharpen":           sharpen,
            "relight":           relight,
            "relight_engine":    relight_engine,
            "relight_prompt":    relight_prompt,
            "relight_strength":  relight_strength,
            "background_engine": background_engine,
            "controlnet_scale":  controlnet_scale,
            "passport_compose":  passport_compose,
            "head_ratio":        head_ratio,
            "face_restore_if_needed": face_restore_if_needed,
            "upscale_if_needed":      upscale_if_needed,
            "ai_planner":             ai_planner,
            "studio_regenerate_full": studio_regenerate_full,
        },
    }

    # Persist in Redis
    _js.create_job(job_id, job_data)

    # Push to worker queue — "bulk" lands on the dedicated bulk worker
    # (ai:queue:bulk), which yields to any high-priority job so single-photo
    # uploads never wait behind a batch.
    queue_pos = _js.push_to_queue_priority(job_id, priority if priority in ("high", "normal", "bulk") else "normal")

    logger.info("[QUEUE_PUSHED] job=%s queue_position=%d", job_id, queue_pos)

    base = str(request.base_url).rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    return {
        "job_id":        job_id,
        "status":        "queued",
        "queue_position": queue_pos,
        "message":       f"Image uploaded — position #{queue_pos} in queue",
        "websocket_url": f"{base}/ws/{job_id}",
    }


# ─── Bulk Enhancement ─────────────────────────────────────────────────────────
# Reuses the existing priority-queue infra (job_store.QUEUE_KEYS["bulk"] /
# ai_worker_pool.py's dedicated bulk worker, which yields whenever a
# high-priority job is waiting) — no new worker code needed, just a way to
# submit many files as one batch instead of N separate /upload calls.

@router.post("/bulk")
async def bulk_upload(
    request: Request,
    files: List[UploadFile] = File(...),
    style_mode: str = Form("professional"),
    upscale_factor: int = Form(2),
    face_restore: bool = Form(True),
    background_replace: bool = Form(False),
    background_prompt: str = Form(""),
    repair_prompt: str = Form(""),
    background_color: str = Form("red"),
    background_engine: str = Form("studio"),
    relight: bool = Form(False),
    relight_engine: str = Form("classical"),
    relight_prompt: str = Form(""),
    relight_strength: float = Form(0.6),
    denoise: bool = Form(True),
    sharpen: bool = Form(True),
    passport_compose: bool = Form(True),
    head_ratio: float = Form(0.50),
    output_width_px: int = Form(413),
    output_height_px: int = Form(531),
    output_dpi: int = Form(300),
    ai_planner: bool = Form(True),
    studio_regenerate_full: bool = Form(False),
):
    if len(files) > 50:
        raise HTTPException(400, f"Batch too large: {len(files)} files (max 50 per /bulk call)")

    q_len = _js.queue_length()
    if q_len + len(files) > _MAX_QUEUE_LENGTH * 5:   # bulk gets a wider ceiling than live traffic
        raise HTTPException(503, f"Server queue is full ({q_len} jobs). Please try again shortly.")

    _MAX_INPUT_PX = 1500
    shared_config = {
        "style_mode": style_mode, "upscale_factor": upscale_factor, "face_restore": face_restore,
        "background_replace": background_replace, "background_prompt": background_prompt,
        "repair_prompt": repair_prompt[:1200],
        "background_color": background_color, "background_engine": background_engine,
        "relight": relight, "relight_engine": relight_engine, "relight_prompt": relight_prompt,
        "relight_strength": relight_strength, "denoise": denoise, "sharpen": sharpen,
        "passport_compose": passport_compose, "head_ratio": head_ratio,
        "output_width_px": output_width_px, "output_height_px": output_height_px,
        "output_dpi": output_dpi,
        "margin_top_px": 24, "margin_bottom_px": 24, "margin_left_px": 18, "margin_right_px": 18,
        "face_restore_if_needed": False, "upscale_if_needed": False,
        "ai_planner": ai_planner,
        "studio_regenerate_full": studio_regenerate_full,
    }

    from PIL import Image as _PILImage

    jobs = []
    for file in files:
        if not (file.content_type or "").startswith("image/"):
            jobs.append({"filename": file.filename, "error": f"Not an image: {file.content_type}"})
            continue

        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            jobs.append({"filename": file.filename, "error": "File too large (max 20 MB)"})
            continue

        try:
            with _PILImage.open(io.BytesIO(content)) as pil_img:
                pil_img.load()
                rgb = pil_img.convert("RGB")
                w, h = rgb.size
                if max(w, h) > _MAX_INPUT_PX:
                    scale = _MAX_INPUT_PX / max(w, h)
                    rgb = rgb.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
                    w, h = rgb.size
        except Exception as exc:
            jobs.append({"filename": file.filename, "error": f"Invalid image: {exc}"})
            continue

        job_id = str(uuid.uuid4())
        input_path = UPLOADS_DIR / f"{job_id}.png"
        rgb.save(str(input_path), "PNG")

        _js.create_job(job_id, {
            "job_id": job_id, "status": "queued", "progress": 0,
            "message": "Job queued — waiting for AI worker (bulk batch)", "stage": "queued",
            "input_path": str(input_path), "output_path": None,
            "metadata": {"original_filename": file.filename, "width": w, "height": h, "batch": True},
            "_ts": time.time(), "_timing": {},
            "config": dict(shared_config),
        })
        queue_pos = _js.push_to_queue_priority(job_id, "bulk")
        jobs.append({"filename": file.filename, "job_id": job_id, "queue_position": queue_pos})

    logger.info("[BULK_UPLOAD] %d files, %d queued", len(files), sum(1 for j in jobs if "job_id" in j))

    base = str(request.base_url).rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    return {
        "batch_size": len(files),
        "queued": sum(1 for j in jobs if "job_id" in j),
        "failed": sum(1 for j in jobs if "error" in j),
        "jobs": jobs,
        "poll_hint": f"{base.replace('ws://', 'http://').replace('wss://', 'https://')}/api/bulk/status?ids=" +
                     ",".join(j["job_id"] for j in jobs if "job_id" in j),
    }


@router.get("/bulk/status")
async def bulk_status(ids: str):
    """Poll many jobs in one call: /api/bulk/status?ids=id1,id2,id3"""
    job_ids = [i.strip() for i in ids.split(",") if i.strip()]
    loop = asyncio.get_event_loop()
    results = []
    for jid in job_ids:
        job = await loop.run_in_executor(None, _js.get_job, jid)
        if not job:
            results.append({"job_id": jid, "status": "not_found"})
        else:
            results.append({
                "job_id": jid, "status": job.get("status"), "progress": job.get("progress"),
                "message": job.get("message"), "output_path": job.get("output_path"),
            })
    done = sum(1 for r in results if r["status"] in ("done", "failed", "timeout", "not_found"))
    return {"total": len(results), "done": done, "jobs": results}


# ─── Job Status ───────────────────────────────────────────────────────────────

@router.get("/job/{job_id}", response_model=dict)
async def get_job_status(job_id: str):
    loop = asyncio.get_event_loop()
    job = await loop.run_in_executor(None, _js.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    # Add live queue position for queued jobs (so frontend can show "#3 in queue")
    if job.get("status") == "queued":
        job["queue_position"] = await loop.run_in_executor(None, _js.queue_position, job_id)
    else:
        job["queue_position"] = 0

    # Strip internal keys
    return {k: v for k, v in job.items() if not k.startswith("_")}


@router.get("/jobs/{job_id}", response_model=dict)
async def get_job_status_alias(job_id: str):
    """Alias for /job/{job_id} — frontend polls plural form."""
    return await get_job_status(job_id)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    loop = asyncio.get_event_loop()
    job  = await loop.run_in_executor(None, _js.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    terminal = {"done", "failed", "timeout", "cancelled"}
    if job.get("status") in terminal:
        return {"status": job.get("status"), "message": "Job already finished"}
    await loop.run_in_executor(None, _js.remove_from_queue, job_id)
    _js.update_job(job_id, status="cancelled", stage="cancelled", progress=0,
                   message="Enhancement cancelled", completed_at=str(time.time()))
    logger.info("[JOB_CANCEL] job=%s", job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    loop = asyncio.get_event_loop()
    job = await loop.run_in_executor(None, _js.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if int(job.get("retry_count") or 0) >= 2:
        raise HTTPException(409, "This job has already been retried twice")
    if job.get("status") not in ("failed", "timeout", "cancelled"):
        raise HTTPException(409, f"Job cannot be retried from status {job.get('status')}")
    _js.update_job(job_id, status="queued", stage="queued", progress=0,
                   message="Retry queued", started_at="", completed_at="", error_message="",
                   retry_count=int(job.get("retry_count") or 0) + 1)
    await loop.run_in_executor(None, _js.push_to_queue, job_id)
    job = await loop.run_in_executor(None, _js.get_job, job_id)
    return {k: v for k, v in job.items() if not k.startswith("_")}


@router.post("/admin/cleanup")
async def admin_cleanup():
    """Force-expire stuck processing jobs and stale queued jobs immediately."""
    loop = asyncio.get_event_loop()
    killed  = await loop.run_in_executor(None, _js.kill_stuck_processing_jobs, 180)
    cleaned = await loop.run_in_executor(None, _js.clean_stale_queued_jobs, 600)
    logger.info("[ADMIN_CLEANUP] killed=%d cleaned=%d", len(killed), len(cleaned))
    return {"killed": killed, "cleaned": cleaned, "total": len(killed) + len(cleaned)}


@router.get("/jobs", response_model=List[dict])
async def list_jobs(limit: int = 20):
    loop = asyncio.get_event_loop()
    all_ids = await loop.run_in_executor(None, _js.scan_all_jobs)
    jobs_out = []
    for jid in all_ids[-limit:]:
        j = await loop.run_in_executor(None, _js.get_job, jid)
        if j:
            jobs_out.append({k: v for k, v in j.items() if not k.startswith("_")})
    return jobs_out


# ─── Download ─────────────────────────────────────────────────────────────────

@router.get("/download/{job_id}")
async def download_result(job_id: str):
    loop = asyncio.get_event_loop()
    job = await loop.run_in_executor(None, _js.get_job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not complete. Status: {job['status']}")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(404, "Output file not found")

    original_name = job.get("metadata", {}).get("original_filename", "")
    stem = Path(original_name).stem if original_name else job_id[:8]
    return FileResponse(path=str(output_path), media_type="image/png",
                        filename=f"{stem}.png")


@router.get("/download/{job_id}/jpg")
async def download_result_jpg(job_id: str):
    loop = asyncio.get_event_loop()
    job = await loop.run_in_executor(None, _js.get_job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not complete. Status: {job['status']}")

    jpg_path = Path(job.get("output_jpg_path", ""))
    if not jpg_path.exists():
        jpg_path = Path(job["output_path"]).with_suffix(".jpg")
    if not jpg_path.exists():
        raise HTTPException(404, "JPEG output not found")

    original_name = job.get("metadata", {}).get("original_filename", "")
    stem = Path(original_name).stem if original_name else job_id[:8]
    return FileResponse(path=str(jpg_path), media_type="image/jpeg",
                        filename=f"{stem}.jpg")


# ─── Queue status ─────────────────────────────────────────────────────────────

@router.get("/queue/status")
async def queue_status():
    """Lightweight endpoint: current queue depth and worker activity."""
    loop = asyncio.get_event_loop()
    q_len = await loop.run_in_executor(None, _js.queue_length)
    return {
        "queue_length": q_len,
        "timestamp":    time.time(),
    }


# ─── Analyze ──────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content  = await file.read()
    tmp_path = UPLOADS_DIR / f"analyze_{uuid.uuid4()}.jpg"
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        analysis = await orchestrator.analyze_image(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    return analysis


# ─── Models ───────────────────────────────────────────────────────────────────

@router.get("/models/status")
async def models_status():
    return orchestrator.get_model_status()


# ─── Diagnostics ──────────────────────────────────────────────────────────────

@router.get("/diagnostics")
async def diagnostics():
    mem_mb = _mem_mb()
    loop = asyncio.get_event_loop()
    all_ids = await loop.run_in_executor(None, _js.scan_all_jobs)

    job_counts: dict = {}
    for jid in all_ids:
        j = await loop.run_in_executor(None, _js.get_job, jid)
        if j:
            s = j.get("status", "unknown")
            job_counts[s] = job_counts.get(s, 0) + 1

    try:
        import psutil
        proc = psutil.Process(os.getpid())
        cpu_pct = proc.cpu_percent(interval=0.2)
        sys_mem = psutil.virtual_memory()
        sys_mem_info = {
            "total_mb":    sys_mem.total // (1024 * 1024),
            "available_mb": sys_mem.available // (1024 * 1024),
            "used_pct":    sys_mem.percent,
        }
    except Exception:
        cpu_pct = -1
        sys_mem_info = {}

    q_len = await loop.run_in_executor(None, _js.queue_length)
    return {
        "process_mem_mb": mem_mb,
        "cpu_pct":        cpu_pct,
        "sys_mem":        sys_mem_info,
        "queue_length":   q_len,
        "total_jobs":     len(all_ids),
        "job_counts":     job_counts,
        "redis_ok":       _js.health_check(),
    }


# ─── Styles ───────────────────────────────────────────────────────────────────

@router.get("/styles")
async def get_styles():
    return {
        "styles": [
            {"id": "professional", "name": "Professional",
             "description": "Studio headshot: even skin, crisp eyes, clean neutral tones", "icon": "🎯",
             "prompt_suffix": "professional headshot, studio lighting, clean background, sharp details"},
            {"id": "luxury",      "name": "Luxury Mode",
             "description": "Gold accents, soft bokeh, premium studio lighting", "icon": "✨",
             "prompt_suffix": "luxury fashion photography, studio lighting, gold accents"},
            {"id": "beach",       "name": "Beach Mode",
             "description": "Golden hour, ocean backdrop, natural lighting", "icon": "🌊",
             "prompt_suffix": "beach photography, golden hour, ocean background"},
            {"id": "temple",      "name": "Temple Mode",
             "description": "Sacred architecture, dramatic light rays", "icon": "🏛️",
             "prompt_suffix": "ancient temple background, dramatic god rays, mystical atmosphere"},
            {"id": "corporate",   "name": "Corporate Mode",
             "description": "Professional office, business portrait", "icon": "💼",
             "prompt_suffix": "corporate headshot, clean professional background"},
            {"id": "anime",       "name": "Anime Mode",
             "description": "Anime-style rendering", "icon": "🎌",
             "prompt_suffix": "anime style, cel shading, vibrant colors"},
            {"id": "whatsapp_dp", "name": "WhatsApp DP Mode",
             "description": "Perfect profile picture", "icon": "📱",
             "prompt_suffix": "profile picture, clean simple background, centered portrait"},
        ]
    }


# ─── Uniform Swap ─────────────────────────────────────────────────────────────

@router.post("/uniform")
async def uniform_swap(
    person_file: UploadFile = File(...),
    uniform_file: UploadFile = File(...),
    bg_color: str = Form("4,126,246"),
    engine: str = Form("catvton"),
    fit_mode: str = Form("Template Exact (Recommended)"),
    request: Request = None,
):
    for uf in (person_file, uniform_file):
        if not (uf.content_type or "").startswith("image/"):
            raise HTTPException(400, f"Not an image: {uf.content_type}")

    job_id = str(uuid.uuid4())
    person_name = Path(person_file.filename or "person.png").name
    uniform_name = Path(uniform_file.filename or "uniform.png").name
    person_path = UPLOADS_DIR / f"{job_id}_{person_name}"
    uniform_path = UPLOADS_DIR / f"{job_id}_{uniform_name}"

    person_path.write_bytes(await person_file.read())
    uniform_path.write_bytes(await uniform_file.read())

    _js.create_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "stage": "queued",
        "input_path": str(person_path),
        "output_path": None,
        "original_filename": person_name,
        "metadata": {
            "type": "uniform_swap",
            "uniform_path": str(uniform_path),
            "bg_color": bg_color,
            "engine": engine,
            "fit_mode": fit_mode,
            "download_name": person_name,
        },
        "_ts": time.time(),
    })
    queue_pos = _js.push_to_queue_priority(job_id, "high")
    base = str(request.base_url).rstrip("/") if request else ""
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_pos,
        "message": "Images uploaded, processing queued",
        "poll_url": f"{base}/api/job/{job_id}",
    }


async def run_uniform_pipeline(job_id: str, person_path: str, uniform_path: str,
                                bg_color: tuple = (4, 126, 246), engine: str = "classic"):
    job = _js.get_job(job_id) or {}
    try:
        _js.update_job(job_id, status="processing", stage="uniform_merge", progress=20,
                       message="Merging person and uniform…", _last_update=str(time.time()))
        loop = asyncio.get_running_loop()
        output_path = await loop.run_in_executor(
            None, lambda: orchestrator.shirt_replacement(
                person_path, uniform_path, job_id, bg_color, engine=engine
            )
        )
        _js.update_job(job_id, status="done", stage="done", progress=100,
                       message="Uniform merge complete!", output_path=str(output_path))
        logger.info("[JOB_COMPLETED] Uniform job=%s", job_id)
    except Exception as e:
        logger.exception("[JOB_FAILED] Uniform job=%s error=%s", job_id, e)
        _js.update_job(job_id, status="failed", stage="failed", progress=0,
                       message=f"Pipeline failed: {str(e)}")


# ─── SDXL Endpoints ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, watermark, text"
    width: int = 1024
    height: int = 1024
    steps: int = 25
    guidance: float = 7.5
    seed: Optional[int] = None


@router.post("/sdxl/generate")
async def sdxl_generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not sdxl.is_available():
        raise HTTPException(503, "SDXL model not downloaded.")
    job_id = str(uuid.uuid4())
    _js.create_job(job_id, {
        "job_id": job_id, "status": "queued", "progress": 0,
        "message": "Queued for generation", "stage": "queued",
        "input_path": None, "output_path": None,
        "metadata": {"type": "sdxl_generate"}, "_ts": time.time(),
    })
    background_tasks.add_task(_run_sdxl_generate, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.post("/sdxl/img2img")
async def sdxl_img2img(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("blurry, low quality, watermark"),
    strength: float = Form(0.6),
    steps: int = Form(30),
    guidance: float = Form(7.5),
    seed: Optional[int] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    if not sdxl.is_available():
        raise HTTPException(503, "SDXL model not downloaded.")
    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".jpg"
    input_path = UPLOADS_DIR / f"{job_id}{ext}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    _js.create_job(job_id, {
        "job_id": job_id, "status": "queued", "progress": 0,
        "message": "Queued for img2img", "stage": "queued",
        "input_path": str(input_path), "output_path": None,
        "metadata": {"type": "sdxl_img2img"}, "_ts": time.time(),
    })
    background_tasks.add_task(_run_sdxl_img2img, job_id, str(input_path),
                               prompt, negative_prompt, strength, steps, guidance, seed)
    return {"job_id": job_id, "status": "queued"}


@router.get("/sdxl/status")
async def sdxl_status():
    from pathlib import Path as P
    return {
        "available":      sdxl.is_available(),
        "base":           (P("models/sdxl/base")).exists(),
        "inpaint":        (P("models/sdxl/inpaint")).exists(),
        "refiner":        (P("models/sdxl/refiner")).exists(),
        "device":         sdxl._get_device(),
    }


async def _run_sdxl_generate(job_id: str, req: GenerateRequest):
    try:
        _js.update_job(job_id, status="processing", stage="sdxl_generate", progress=10,
                       message="Loading SDXL model…")
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: sdxl.generate(
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            width=req.width, height=req.height,
            steps=req.steps, guidance=req.guidance, seed=req.seed, job_id=job_id,
        ))
        _js.update_job(job_id, status="done", stage="done", progress=100,
                       message="Generation complete!", output_path=str(out))
    except Exception as e:
        logger.exception("[JOB_FAILED] SDXL generate=%s", e)
        _js.update_job(job_id, status="failed", stage="failed", progress=0,
                       message=f"SDXL failed: {e}")


async def _run_sdxl_img2img(job_id, input_path, prompt, negative_prompt, strength, steps, guidance, seed):
    try:
        _js.update_job(job_id, status="processing", stage="sdxl_img2img", progress=10,
                       message="Loading SDXL model…")
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: sdxl.img2img(
            input_path=input_path, prompt=prompt, negative_prompt=negative_prompt,
            strength=strength, steps=steps, guidance=guidance, seed=seed, job_id=job_id,
        ))
        _js.update_job(job_id, status="done", stage="done", progress=100,
                       message="img2img complete!", output_path=str(out))
    except Exception as e:
        logger.exception("[JOB_FAILED] SDXL img2img=%s", e)
        _js.update_job(job_id, status="failed", stage="failed", progress=0,
                       message=f"SDXL failed: {e}")


# ─── ZIP ──────────────────────────────────────────────────────────────────────

import hashlib as _hashlib

ZIPS_DIR = Path("outputs/zips")
zip_jobs: dict = {}   # In-memory ZIP state (short-lived; no Redis needed)


class ZipRequest(BaseModel):
    job_ids: List[str]


def _zip_cache_key(job_ids: list) -> str:
    """Stable 20-char hex key for a set of job IDs (order-independent)."""
    return _hashlib.sha256(",".join(sorted(job_ids)).encode()).hexdigest()[:20]


def _build_zip_fast(zip_id: str, entries: list, zip_path: str) -> tuple:
    """Build a ZIP from already-completed job output files.

    Uses zf.write() for direct disk-to-archive streaming with no intermediate
    memory buffer and no per-file stability waits — files are guaranteed stable
    because their jobs are 'done' in the job store.  Returns (added, skipped).
    """
    ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    tmp = zip_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, path in entries:
                p = Path(path)
                try:
                    if not p.exists() or p.stat().st_size == 0:
                        logger.warning("[ZIP] skipping %s — missing or empty", name)
                        skipped += 1
                        continue
                    zf.write(str(p), arcname=name)
                    added += 1
                except Exception as exc:
                    logger.warning("[ZIP] skipping %s: %s", name, exc)
                    skipped += 1
        if added > 0:
            os.replace(tmp, zip_path)   # atomic rename so readers never see a partial file
        else:
            Path(tmp).unlink(missing_ok=True)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return added, skipped


@router.post("/zip")
async def create_zip(req: ZipRequest, request: Request):
    if not req.job_ids:
        raise HTTPException(400, "No job_ids provided")

    loop = asyncio.get_event_loop()
    entries: list    = []
    seen_names: dict = {}

    for jid in req.job_ids:
        job = await loop.run_in_executor(None, _js.get_job, jid)
        if not job or job.get("status") != "done":
            continue
        out = job.get("output_path")
        if not out:
            continue
        orig_name = job.get("metadata", {}).get("original_filename", "")
        stem = Path(orig_name).stem if orig_name else jid[:8]
        name = f"{stem}.png"
        if name in seen_names:
            seen_names[name] += 1
            name = f"{stem}_{seen_names[name]}.png"
        else:
            seen_names[name] = 0
        entries.append((name, out))

    if not entries:
        raise HTTPException(404, "No completed jobs found")

    # Content-addressed cache — same set of completed jobs → same ZIP key
    zip_id   = _zip_cache_key(req.job_ids)
    zip_path = str(ZIPS_DIR / f"{zip_id}.zip")
    base_url = str(request.base_url).rstrip("/")
    dl_url   = f"{base_url}/api/zip/{zip_id}/download"

    # Return the cached ZIP immediately if it still exists on disk
    zj = zip_jobs.get(zip_id)
    if zj and zj["status"] == "done" and Path(zip_path).exists():
        logger.info("[ZIP] cache hit %s — %d files", zip_id[:8], len(entries))
        return {
            "zip_id":  zip_id,
            "count":   len(entries),
            "status":  "done",
            "url":     dl_url,
            "added":   zj.get("added", len(entries)),
            "skipped": zj.get("skipped", 0),
        }

    # Build ZIP synchronously in executor.
    # With ZIP_STORED + zf.write() this is essentially a raw file-copy:
    # ~0.5–2 s for 100 enhanced images (300–800 KB each) on any modern disk.
    zip_jobs[zip_id] = {
        "status":     "building",
        "path":       zip_path,
        "created_at": time.time(),
        "count":      len(entries),
        "added":      0,
        "skipped":    0,
    }
    logger.info("[ZIP] building %s — %d files", zip_id[:8], len(entries))

    try:
        added, skipped = await loop.run_in_executor(
            None, _build_zip_fast, zip_id, entries, zip_path
        )
    except Exception as exc:
        zip_jobs[zip_id].update(status="failed", error=str(exc))
        logger.exception("[ZIP] build failed: %s", exc)
        raise HTTPException(500, f"ZIP build failed: {exc}")

    if added == 0:
        zip_jobs[zip_id].update(status="failed", error="No output files found")
        raise HTTPException(500, "No output files could be added to ZIP")

    zip_jobs[zip_id].update(status="done", added=added, skipped=skipped)
    logger.info("[ZIP] %s ready — added=%d skipped=%d", zip_id[:8], added, skipped)

    return {
        "zip_id":  zip_id,
        "count":   len(entries),
        "status":  "done",
        "url":     dl_url,
        "added":   added,
        "skipped": skipped,
    }


@router.get("/zip/{zip_id}")
async def get_zip(zip_id: str, request: Request):
    zj = zip_jobs.get(zip_id)
    if not zj:
        raise HTTPException(404, "ZIP not found")
    if zj["status"] in ("pending", "building"):
        return JSONResponse({"status": "pending", "count": zj.get("count", 0)},
                            status_code=202)
    if zj["status"] == "failed":
        raise HTTPException(500, zj.get("error", "ZIP generation failed"))

    base = str(request.base_url).rstrip("/")
    url  = f"{base}/api/zip/{zip_id}/download"
    return JSONResponse({
        "status":  "done",
        "url":     url,
        "added":   zj.get("added", 0),
        "skipped": zj.get("skipped", 0),
    })


@router.get("/zip/{zip_id}/download")
async def download_zip(zip_id: str):
    zj = zip_jobs.get(zip_id)
    if not zj or zj["status"] != "done":
        raise HTTPException(404, "ZIP not ready")
    p = Path(zj["path"])
    if not p.exists():
        raise HTTPException(404, "ZIP file missing on server")
    return FileResponse(str(p), media_type="application/zip",
                        filename=f"enhanced_{zip_id[:8]}.zip")


# ─── Qwen Studio Enhancer & Uniform Swap ──────────────────────────────────────

from pipelines.qwen_edit_pipeline import QwenEditPipeline
_qwen_pipe = QwenEditPipeline()


@router.post("/qwen/enhance")
async def api_qwen_enhance(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    background_color: str = Form("white"),
    upscale_factor: int = Form(1),
    width: int = Form(600),
    height: int = Form(800),
    mode: str = Form("qwen"),
    steps: int = Form(20),
    priority: str = Form("normal"),
    request: Request = None,
):
    """
    Qwen Studio Portrait Enhancer with FIFO / Priority Queue:
    - Enqueues job and executes strictly 1-by-1 sequentially.
    """
    job_id = f"qwen_enh_{uuid.uuid4().hex[:8]}"
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds 20MB limit")

    upload_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(content)

    orig_filename = file.filename or "photo.png"
    orig_stem = Path(orig_filename).stem

    job_data = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Queued for studio enhancement",
        "input_path": str(upload_path),
        "output_path": None,
        "original_filename": orig_filename,
        "metadata": {
            "type": "qwen_enhance",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "background_color": background_color,
            "upscale_factor": upscale_factor,
            "width": width,
            "height": height,
            "mode": mode,
            "steps": steps,
            "download_name": f"{orig_stem}.png",
        },
        "_ts": time.time(),
    }

    _js.create_job(job_id, job_data)
    queue_pos = _js.push_to_queue_priority(job_id, priority if priority in ("high", "normal", "bulk") else "normal")

    logger.info("[QUEUE_ENQUEUED] Qwen Enhance job=%s (pos #%d)", job_id, queue_pos)

    base = str(request.base_url).rstrip("/") if request else ""
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_pos,
        "message": f"Image uploaded — position #{queue_pos} in queue",
        "poll_url": f"{base}/api/job/{job_id}",
        "output_url": "",
        "filename": "",
        "original_filename": orig_filename,
        "download_name": f"{orig_stem}.png",
    }


@router.post("/qwen/uniform-swap")
async def api_qwen_uniform_swap(
    file: UploadFile = File(..., description="Student / Person portrait photo (Required)"),
    uniform_template: UploadFile = File(..., description="School uniform template image (Required)"),
    prompt: Optional[str] = Form(None),
    width: int = Form(600),
    height: int = Form(800),
    priority: str = Form("normal"),
    request: Request = None,
):
    """
    Qwen Formal School Uniform Swap with FIFO Queue:
    - Enqueues job and executes strictly 1-by-1 sequentially.
    """
    job_id = f"qwen_unif_{uuid.uuid4().hex[:8]}"
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Person image exceeds 20MB limit")

    if not uniform_template or not uniform_template.filename:
        raise HTTPException(400, "Uniform template image is mandatory for Uniform Swap.")

    u_content = await uniform_template.read()
    if len(u_content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Uniform template image exceeds 20MB limit")

    person_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with open(person_path, "wb") as f:
        f.write(content)

    u_path = UPLOADS_DIR / f"{job_id}_custom_template_{uniform_template.filename}"
    with open(u_path, "wb") as f:
        f.write(u_content)

    orig_stem = Path(file.filename or "student").stem
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Queued for uniform swap",
        "input_path": str(person_path),
        "output_path": None,
        "original_filename": file.filename or "photo.png",
        "metadata": {
            "type": "qwen_uniform_swap",
            "uniform_template_path": str(u_path),
            "prompt": prompt,
            "width": width,
            "height": height,
            "download_name": f"{orig_stem}_uniform.jpg",
        },
        "_ts": time.time(),
    }

    _js.create_job(job_id, job_data)
    queue_pos = _js.push_to_queue_priority(job_id, priority if priority in ("high", "normal", "bulk") else "normal")

    logger.info("[QUEUE_ENQUEUED] Qwen Uniform job=%s (pos #%d)", job_id, queue_pos)

    base = str(request.base_url).rstrip("/") if request else ""
    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_pos,
        "message": f"Uniform swap queued — position #{queue_pos} in queue",
        "poll_url": f"{base}/api/job/{job_id}",
        "output_url": "",
        "filename": "",
    }
