"""Solver concurrency gate.

Limits how many solver requests run simultaneously so a burst of plan
requests doesn't exhaust the threadpool. Requests that arrive when the
gate is full receive a 503 immediately (no queuing) — the Streamlit
client retries with jitter via ``_with_one_retry``.
"""

from __future__ import annotations

import threading
from functools import wraps
from typing import Callable, Dict

from flask import jsonify

from api import metrics

_MAX_CONCURRENT = 3
_sem = threading.Semaphore(_MAX_CONCURRENT)
_active = 0
_active_lock = threading.Lock()


def _inc_active() -> int:
    global _active
    with _active_lock:
        _active += 1
        return _active


def _dec_active() -> None:
    global _active
    with _active_lock:
        _active = max(0, _active - 1)


def get_stats() -> Dict[str, int]:
    """Return a snapshot of solver concurrency for the /health response."""
    with _active_lock:
        active = _active
    return {"active": active, "queued": 0, "max_concurrent": _MAX_CONCURRENT}


def get_worker_count() -> int:
    """Return the recommended CP-SAT worker count for this deployment."""
    return 4


def solver_gate(fn: Callable) -> Callable:
    """Decorator: reject with 503 when too many solvers are already running."""
    @wraps(fn)
    def inner(*args, **kwargs):
        acquired = _sem.acquire(blocking=False)
        if not acquired:
            metrics.incr("solver_gate_rejected_total")
            return jsonify({
                "success": False,
                "error": "Solver busy — too many concurrent requests. Retry shortly.",
            }), 503
        _inc_active()
        try:
            return fn(*args, **kwargs)
        finally:
            _dec_active()
            _sem.release()
    return inner
