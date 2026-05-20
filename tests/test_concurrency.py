"""Tests for api.concurrency.solver_gate.

The gate serializes CPU-heavy solves behind MAX_RUNNING running slots with
a bounded wait queue. The tricky case is the timeout path: if a releasing
worker pops+sets our event in the same instant our wait() returns False,
we must not return 504 while leaking a running slot.
"""

import threading
import time

import pytest

from flask import Flask

import api.concurrency as conc


@pytest.fixture(autouse=True)
def _reset_gate_state(monkeypatch):
    """Each test gets a clean gate with small limits so we can force races."""
    monkeypatch.setattr(conc, "_running", 0, raising=False)
    monkeypatch.setattr(conc, "_queue", conc.deque(), raising=False)
    monkeypatch.setattr(conc, "MAX_RUNNING", 1)
    monkeypatch.setattr(conc, "MAX_QUEUED", 4)
    yield
    # Sanity: nothing should leak past the test
    assert conc._running == 0, f"leaked running slots: {conc._running}"
    assert len(conc._queue) == 0, f"leaked queue entries: {len(conc._queue)}"


def _make_app():
    """Tiny Flask app so jsonify() has an app context when the gate 503/504s."""
    app = Flask(__name__)

    @app.route("/go")
    @conc.solver_gate
    def go():
        # Hold briefly so a second caller has time to enqueue.
        time.sleep(float(app.config.get("HOLD", 0.0)))
        return {"ok": True}

    return app


def test_single_request_runs_and_releases():
    app = _make_app()
    with app.test_client() as c:
        resp = c.get("/go")
    assert resp.status_code == 200
    assert conc._running == 0
    assert len(conc._queue) == 0


def test_queue_fills_then_503s_when_full(monkeypatch):
    monkeypatch.setattr(conc, "MAX_QUEUED", 1)
    app = _make_app()
    app.config["HOLD"] = 0.2

    # Saturate: one running, one queued, one over → 503
    results = [None, None, None]

    def call(idx):
        with app.test_client() as c:
            results[idx] = c.get("/go").status_code

    t1 = threading.Thread(target=call, args=(0,)); t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=call, args=(1,)); t2.start()
    time.sleep(0.05)
    t3 = threading.Thread(target=call, args=(2,)); t3.start()
    for t in (t1, t2, t3):
        t.join(timeout=5)

    assert results[0] == 200
    assert results[1] == 200
    assert results[2] == 503


def test_genuine_timeout_returns_504_without_leaking_slot(monkeypatch):
    """A waiter whose event is never set must return 504 and not leave
    _running elevated."""
    monkeypatch.setattr(conc, "QUEUE_TIMEOUT", 0.1)

    app = _make_app()

    # Manually occupy the single running slot so any new caller is queued.
    with conc._lock:
        conc._running = conc.MAX_RUNNING

    try:
        with app.test_client() as c:
            resp = c.get("/go")
    finally:
        # Simulate the long-running call finishing normally.
        with conc._lock:
            conc._running = 0

    assert resp.status_code == 504


def test_late_promotion_is_honoured_not_dropped(monkeypatch):
    """Race: wait(timeout) returns False, but a releasing worker set() our
    event in the same instant and incremented _running on our behalf. We
    must run fn() — dropping the request here is the slot leak."""
    monkeypatch.setattr(conc, "QUEUE_TIMEOUT", 0.05)

    app = _make_app()
    ran = threading.Event()

    # Register a route whose handler we can detect ran.
    @app.route("/late")
    @conc.solver_gate
    def late():
        ran.set()
        return {"ok": True}

    # Simulate: we are queued, then just after timeout expires, another
    # thread pops us and sets the event. Easiest way is to arrange the
    # event to be set ourselves right after the waiter enqueues.
    def racer():
        # Let the request reach wait() first
        time.sleep(0.02)
        with conc._lock:
            if conc._queue:
                ev = conc._queue.popleft()
                conc._running += 1
                ev.set()

    # Occupy the running slot so the waiter must enqueue.
    with conc._lock:
        conc._running = conc.MAX_RUNNING

    t = threading.Thread(target=racer); t.start()

    try:
        with app.test_client() as c:
            # Just after we enqueue, racer will promote us. Even if our
            # wait() returned False because the timeout fired between
            # set() and our resume, the re-check must pick up is_set()
            # and let us run.
            resp = c.get("/late")
    finally:
        # Balance the manual bump above — the request's finally block
        # decremented _running from the racer's increment down by one;
        # the original occupier we faked still needs to release.
        with conc._lock:
            conc._running = max(0, conc._running - 1)
        t.join(timeout=2)

    assert resp.status_code == 200
    assert ran.is_set()
