from __future__ import annotations

"""
Redis-backed job store with transparent In-Memory Fallback.
Job data is stored as Redis hash at key ai:job:{job_id} or in memory when Redis is offline.
Queue is a priority-aware Redis list or memory list.
"""
import json
import time
import logging

logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379/0"
JOB_TTL   = 86400         # 24 hours
QUEUE_KEY = "ai:queue"    # Legacy single queue
CACHE_TTL = 86400         # 24 hours

QUEUE_KEYS = {
    "high":   "ai:queue:high",
    "normal": "ai:queue:normal",
    "bulk":   "ai:queue:bulk",
}
_POP_ORDER = ["ai:queue:high", "ai:queue:normal", "ai:queue:bulk", "ai:queue"]

_client = None
_redis_available = None

# In-memory fallbacks when Redis is not running
_MEM_JOBS: dict[str, dict] = {}
_MEM_QUEUES: dict[str, list[str]] = {k: [] for k in _POP_ORDER}
_MEM_CACHE: dict[str, dict] = {}


def _r():
    """Lazy singleton Redis connection with availability test."""
    global _client, _redis_available
    if _client is None:
        import redis as _redis
        try:
            client = _redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1,
            )
            client.ping()
            _client = client
            _redis_available = True
        except Exception:
            _client = None
            _redis_available = False
    return _client


def _jkey(job_id: str) -> str:
    return f"ai:job:{job_id}"


def _ckey(file_hash: str, config_hash: str) -> str:
    return f"ai:cache:{file_hash}:{config_hash}"


def _serialize(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    return str(v)


def _deserialize(raw: dict) -> dict:
    result = {}
    for k, v in raw.items():
        if k in ("config", "metadata", "_timing", "analysis", "quality_gate"):
            try:
                result[k] = json.loads(v) if isinstance(v, str) and v else (v if isinstance(v, dict) else {})
            except Exception:
                result[k] = {}
        elif k in ("progress", "retry_count", "queue_position"):
            try:
                result[k] = int(float(v))
            except Exception:
                result[k] = 0
        elif k in ("_ts", "_last_update", "created_at", "started_at", "completed_at", "updated_at"):
            try:
                result[k] = float(v)
            except Exception:
                result[k] = 0.0
        else:
            result[k] = v
    return result


# ─── Public API ───────────────────────────────────────────────────────────────

def create_job(job_id: str, data: dict) -> None:
    now = time.time()
    data.setdefault("created_at", now)
    data.setdefault("updated_at", now)
    data.setdefault("retry_count", 0)
    flat = {k: _serialize(v) for k, v in data.items()}

    r = _r()
    if r is not None:
        try:
            r.hset(_jkey(job_id), mapping=flat)
            r.expire(_jkey(job_id), JOB_TTL)
            logger.info("[JOB_CREATED] job=%s (Redis)", job_id)
            return
        except Exception as e:
            logger.warning("[JOB_STORE] Redis unavailable, falling back to memory (%s)", e)

    _MEM_JOBS[job_id] = flat
    logger.info("[JOB_CREATED] job=%s (Memory)", job_id)


def get_job(job_id: str) -> dict | None:
    r = _r()
    if r is not None:
        try:
            raw = r.hgetall(_jkey(job_id))
            if raw:
                return _deserialize(raw)
        except Exception:
            pass

    raw = _MEM_JOBS.get(job_id)
    if not raw:
        return None
    return _deserialize(raw)


def update_job(job_id: str, **fields) -> None:
    fields.setdefault("updated_at", time.time())
    flat = {k: _serialize(v) for k, v in fields.items()}

    r = _r()
    if r is not None:
        try:
            r.hset(_jkey(job_id), mapping=flat)
            r.expire(_jkey(job_id), JOB_TTL)
            return
        except Exception:
            pass

    if job_id not in _MEM_JOBS:
        _MEM_JOBS[job_id] = {}
    _MEM_JOBS[job_id].update(flat)


def push_to_queue_priority(job_id: str, priority: str = "normal") -> int:
    key = QUEUE_KEYS.get(priority, QUEUE_KEYS["normal"])
    r = _r()
    if r is not None:
        try:
            for k in _POP_ORDER:
                if job_id in r.lrange(k, 0, -1):
                    return queue_position(job_id)
            r.rpush(key, job_id)
            update_job(job_id, _priority=priority)
            pos = queue_position(job_id)
            logger.info("[QUEUE_PUSHED] job=%s priority=%s pos=%d", job_id, priority, pos)
            return pos
        except Exception:
            pass

    # Memory queue fallback
    for k in _POP_ORDER:
        if job_id in _MEM_QUEUES[k]:
            return queue_position(job_id)
    _MEM_QUEUES[key].append(job_id)
    update_job(job_id, _priority=priority)
    return queue_position(job_id)


def push_to_queue(job_id: str) -> int:
    job = get_job(job_id)
    priority = (job or {}).get("_priority", "normal")
    return push_to_queue_priority(job_id, priority)


def pop_from_queue(timeout: int = 10):
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        r = _r()
        if r is not None:
            try:
                for key in _POP_ORDER:
                    result = r.lpop(key)
                    if result:
                        return result
            except Exception:
                pass

        # Check memory queues
        for key in _POP_ORDER:
            if _MEM_QUEUES[key]:
                return _MEM_QUEUES[key].pop(0)
        _time.sleep(0.5)
    return None


def queue_position(job_id: str) -> int:
    r = _r()
    offset = 0
    if r is not None:
        try:
            for key in _POP_ORDER:
                items = r.lrange(key, 0, -1)
                if job_id in items:
                    return offset + items.index(job_id) + 1
                offset += len(items)
            return 0
        except Exception:
            pass

    offset = 0
    for key in _POP_ORDER:
        items = _MEM_QUEUES[key]
        if job_id in items:
            return offset + items.index(job_id) + 1
        offset += len(items)
    return 0


def queue_length() -> int:
    r = _r()
    if r is not None:
        try:
            return sum(r.llen(k) for k in _POP_ORDER)
        except Exception:
            pass
    return sum(len(_MEM_QUEUES[k]) for k in _POP_ORDER)


def queue_counts() -> dict:
    r = _r()
    if r is not None:
        try:
            return {
                "high":   r.llen("ai:queue:high"),
                "normal": r.llen("ai:queue:normal"),
                "bulk":   r.llen("ai:queue:bulk"),
                "legacy": r.llen(QUEUE_KEY),
                "total":  sum(r.llen(k) for k in _POP_ORDER),
            }
        except Exception:
            pass
    return {
        "high":   len(_MEM_QUEUES["ai:queue:high"]),
        "normal": len(_MEM_QUEUES["ai:queue:normal"]),
        "bulk":   len(_MEM_QUEUES["ai:queue:bulk"]),
        "legacy": len(_MEM_QUEUES[QUEUE_KEY]),
        "total":  sum(len(_MEM_QUEUES[k]) for k in _POP_ORDER),
    }


def remove_from_queue(job_id: str) -> int:
    r = _r()
    if r is not None:
        try:
            return sum(r.lrem(k, 0, job_id) for k in _POP_ORDER)
        except Exception:
            pass
    removed = 0
    for k in _POP_ORDER:
        if job_id in _MEM_QUEUES[k]:
            _MEM_QUEUES[k].remove(job_id)
            removed += 1
    return removed


def kill_stuck_processing_jobs(max_age_seconds: int = 300) -> list:
    now = time.time()
    killed = []
    try:
        for job_id in scan_all_jobs():
            job = get_job(job_id)
            if not job or job.get("status") != "processing":
                continue
            last = float(job.get("_last_update") or job.get("started_at") or now)
            if (now - last) > max_age_seconds:
                update_job(job_id, status="failed", stage="failed", progress=0,
                           message="Enhancement failed. Retry?",
                           completed_at=str(now),
                           error_message=f"Timed out after {int(now-last)}s without progress")
                killed.append(job_id)
    except Exception:
        pass
    return killed


def job_counts() -> dict:
    counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "total": 0}
    try:
        for jid in scan_all_jobs():
            j = get_job(jid)
            if not j:
                continue
            st = j.get("status", "unknown")
            counts["total"] += 1
            if st == "queued":
                counts["queued"] += 1
            elif st == "processing":
                counts["processing"] += 1
            elif st in ("done", "completed"):
                counts["completed"] += 1
            elif st in ("failed", "error"):
                counts["failed"] += 1
    except Exception:
        pass
    return counts


def get_all_jobs() -> dict[str, dict]:
    result = {}
    try:
        for jid in scan_all_jobs():
            j = get_job(jid)
            if j:
                result[jid] = j
    except Exception:
        pass
    return result


def set_cached_result(file_hash: str, config_hash: str, output_path: str, metadata: dict | None = None) -> None:
    if not file_hash or not config_hash:
        return
    data = {
        "output_path": output_path,
        "created_at":  time.time(),
        "metadata":    _serialize(metadata or {}),
    }
    r = _r()
    if r is not None:
        try:
            k = _ckey(file_hash, config_hash)
            r.hset(k, mapping=data)
            r.expire(k, CACHE_TTL)
            return
        except Exception:
            pass
    _MEM_CACHE[f"{file_hash}:{config_hash}"] = data


def get_cached_result(file_hash: str, config_hash: str) -> dict | None:
    if not file_hash or not config_hash:
        return None
    r = _r()
    if r is not None:
        try:
            raw = r.hgetall(_ckey(file_hash, config_hash))
            if raw:
                return _deserialize(raw)
        except Exception:
            pass
    raw = _MEM_CACHE.get(f"{file_hash}:{config_hash}")
    if not raw:
        return None
    return _deserialize(raw)


def scan_all_jobs() -> list[str]:
    r = _r()
    if r is not None:
        try:
            return [k.split(":", 2)[2] for k in r.scan_iter("ai:job:*")]
        except Exception:
            pass
    return list(_MEM_JOBS.keys())


def delete_job(job_id: str) -> None:
    r = _r()
    if r is not None:
        try:
            r.delete(_jkey(job_id))
        except Exception:
            pass
    _MEM_JOBS.pop(job_id, None)


def health_check() -> bool:
    r = _r()
    if r is not None:
        try:
            return bool(r.ping())
        except Exception:
            pass
    return False


def average_processing_time() -> float:
    return 3.5


def estimated_wait_time(max_workers: int = 1) -> float:
    """Estimate queue delay in seconds from the current worker capacity."""
    workers = max(1, int(max_workers))
    return round((queue_length() / workers) * average_processing_time(), 1)


def active_workers() -> int:
    return 1


def worker_heartbeat() -> None:
    pass
