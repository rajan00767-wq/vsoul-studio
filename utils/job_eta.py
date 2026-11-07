"""Consistent estimated-time metadata for every queued or running job."""
from __future__ import annotations

import time
from typing import Any, Dict


_DEFAULT_TOTAL_SECONDS = {
    "uniform_swap": 420,
    "qwen_uniform_swap": 420,
    "qwen_enhance": 300,
    "bulk_enhance": 300,
    "sdxl_generate": 180,
    "sdxl_img2img": 180,
    "orchestrator": 240,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"about {minutes}m {remainder:02d}s remaining"
    return f"about {remainder}s remaining"


def estimate_job_time(job: Dict[str, Any], queue_position: int = 0) -> Dict[str, Any]:
    """Return stable ETA fields without modifying the stored job record."""
    status = str(job.get("status") or "queued").lower()
    metadata = job.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    job_type = str(metadata.get("type") or "orchestrator")
    baseline = int(_DEFAULT_TOTAL_SECONDS.get(job_type, 240))
    progress = max(0.0, min(100.0, _number(job.get("progress"))))
    now = time.time()

    started = _number(job.get("started_at"))
    elapsed = max(0, int(now - started)) if started > 0 else 0

    if status in {"done", "failed", "timeout", "cancelled"}:
        remaining = 0
        text = "complete" if status == "done" else "stopped"
    elif status == "queued":
        # Position 1 includes this job's own expected processing time.
        remaining = baseline * max(1, int(queue_position or 1))
        text = _format_duration(remaining)
    else:
        # Never let a synthetic early progress jump make the ETA unrealistically
        # optimistic. Once real elapsed/progress evidence exceeds the baseline,
        # extend the estimate dynamically.
        observed_total = (
            elapsed / max(progress / 100.0, .01)
            if elapsed >= 15 and progress > 0 else baseline
        )
        estimated_total = max(float(baseline), observed_total)
        remaining = max(1, int(round(estimated_total - elapsed)))
        text = _format_duration(remaining)

    return {
        "estimated_total_seconds": baseline,
        "estimated_remaining_seconds": remaining,
        "elapsed_seconds": elapsed,
        "estimated_completion_at": int(now + remaining) if remaining else None,
        "estimated_time": text,
    }
