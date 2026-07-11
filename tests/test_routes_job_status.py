"""Regression test for a bug found in production: a downstream export
failure (e.g. a missing Supabase Storage bucket) after a successful
orchestrator run was overwriting the job's "completed" status and real
report with a bare {"error": ...} -- losing the actual pipeline result.
Confirmed live on Render: evidence/indoor_features/review_reports were all
correctly written to Postgres, but /status/{job_id} reported "failed"
because the GeoJSON export step (a separate concern) threw afterward.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes_module
import app.store.supabase as supabase_module
from app.main import app


class ExplodingStorage:
    """Simulates a Storage backend that fails on every write -- e.g. a
    missing bucket -- regardless of what the pipeline itself did."""

    def put_json(self, *args, **kwargs):
        raise RuntimeError("Bucket not found")

    def put_bytes(self, *args, **kwargs):
        raise RuntimeError("Bucket not found")


@pytest.fixture(autouse=True)
def force_offline_scraping(monkeypatch):
    import app.agents.tools.web as web_module

    monkeypatch.setattr(web_module, "fetch_directory_html", lambda *a, **k: None)
    monkeypatch.setattr(web_module, "fetch_rendered_html", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def reset_run_lock():
    """_run_lock/_active_job_id are module-level globals guarding real
    concurrency in production -- but that also makes them shared state
    across tests in this file. Two things to guard against here:

    1. A test that posts /run without waiting for the background job to
       finish would otherwise leave the lock held, causing an unrelated
       later test's /run call to spuriously 409.
    2. Naively reassigning `_run_lock` to a fresh Lock() object doesn't
       actually fix that -- _worker()'s `finally: _run_lock.release()`
       resolves `_run_lock` from the module namespace at call time (it's
       a `global`, not a captured local), so if a still-running
       background thread from the *previous* test releases *after* this
       fixture has already swapped in a new Lock object, it raises
       "release unlocked lock" against the new object. So: never swap the
       object: force-release the *same* lock if left held, and on
       teardown wait for any in-flight background thread to release it
       naturally instead of yanking it out from under that thread.
    """
    if routes_module._run_lock.locked():
        routes_module._run_lock.release()
    routes_module._active_job_id = None
    yield
    for _ in range(50):
        if not routes_module._run_lock.locked():
            break
        time.sleep(0.1)


def test_export_failure_does_not_clobber_a_completed_job(tmp_path, monkeypatch):
    monkeypatch.setattr(supabase_module, "_DEV_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(routes_module, "get_storage", lambda: ExplodingStorage())
    # each request re-resolves get_store() via the module-level singleton;
    # reset it so this test's dev DB path takes effect
    monkeypatch.setattr(supabase_module, "_store", None)

    client = TestClient(app)
    resp = client.post("/run", json={"mall": "Mall of America", "floors": [1], "max_iterations": 3})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status = None
    for _ in range(30):
        status_resp = client.get(f"/status/{job_id}")
        status = status_resp.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)

    body = status_resp.json()
    # the pipeline itself succeeded -- the export failure must not turn
    # this into "failed" or replace the real report with a bare error dict
    assert status == "completed", body
    assert "accuracy" in body["report"]
    assert "error" not in body["report"]


def test_run_accepts_base_url_override(tmp_path, monkeypatch):
    """POST /run's base_url field (the /ui frontend's "mall website" input)
    should be threaded through to the Orchestrator instead of always using
    the fixed MALL_BASE_URL env var -- otherwise pasting a different mall's
    URL into the UI would silently do nothing."""
    monkeypatch.setattr(supabase_module, "_DEV_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(supabase_module, "_store", None)

    captured = {}
    import app.api.routes as routes_module_local
    real_orchestrator_cls = routes_module_local.Orchestrator

    class SpyOrchestrator(real_orchestrator_cls):
        def __init__(self, store, base_url):
            captured["base_url"] = base_url
            super().__init__(store, base_url)

    monkeypatch.setattr(routes_module_local, "Orchestrator", SpyOrchestrator)

    client = TestClient(app)
    resp = client.post("/run", json={
        "mall": "Some Other Mall", "base_url": "https://example-other-mall.invalid",
        "floors": [1], "max_iterations": 1,
    })
    assert resp.status_code == 200
    assert captured["base_url"] == "https://example-other-mall.invalid"


def test_job_trail_endpoint_returns_full_agent_activity(tmp_path, monkeypatch):
    """GET /jobs/{job_id}/trail powers the /ui live activity feed -- it
    should return the full, unfiltered audit trail for a job (unlike
    /audit/{feature_id}, which is scoped to one feature)."""
    monkeypatch.setattr(supabase_module, "_DEV_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(supabase_module, "_store", None)

    client = TestClient(app)
    resp = client.post("/run", json={"mall": "Mall of America", "floors": [1], "max_iterations": 2})
    job_id = resp.json()["job_id"]

    status = None
    for _ in range(30):
        status = client.get(f"/status/{job_id}").json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert status == "completed"

    trail = client.get(f"/jobs/{job_id}/trail").json()
    events = {row["event"] for row in trail}
    assert "job_started" in events
    assert "evidence_collected" in events
    assert "validation_summary" in events
    assert "mapping_summary" in events
    assert "review_decision" in events
    assert "job_completed" in events

    # every event except job_completed is tagged with which of the 5 agents
    # produced it, so a UI can show "which agent is active right now" from
    # the latest event alone. job_completed is deliberately untagged -- it's
    # not itself a Task Intake action, and tagging it would make the UI's
    # "most recent agent" signal jump backward to stage 1 at the very end
    # instead of staying on whichever agent did the last real work.
    for row in trail:
        if row["event"] == "job_completed":
            assert "agent" not in row["detail"]
        else:
            assert row["detail"].get("agent"), row

    evidence_rows = [row for row in trail if row["event"] == "evidence_collected"]
    # full evidence detail is present, not just a category label -- this is
    # what lets the UI show exactly what was found (the actual observation
    # values, source URL, certainty), not just "evidence_collected: Nike"
    assert all("observation" in row["detail"] for row in evidence_rows)
    assert all("source_url" in row["detail"] for row in evidence_rows)
    assert all("certainty" in row["detail"] for row in evidence_rows)


def test_concurrent_run_rejected_with_409(tmp_path, monkeypatch):
    """Regression test for a bug found live: two /run jobs running at once
    on this single-worker deployment caused a dropped Supabase connection
    on one job and intermittent 500s on /status polling for the other.
    A second /run while one is already in progress must be rejected
    immediately with a clear error, not silently accepted to fight over
    the same CPU/connections."""
    monkeypatch.setattr(supabase_module, "_DEV_DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(supabase_module, "_store", None)

    started = threading.Event()
    release = threading.Event()

    import app.orchestrator as orchestrator_module
    original_run = orchestrator_module.Orchestrator.run

    def blocking_run(self, job_id, config):
        started.set()
        release.wait(timeout=10)
        return original_run(self, job_id, config)

    monkeypatch.setattr(orchestrator_module.Orchestrator, "run", blocking_run)

    client = TestClient(app)
    resp1 = client.post("/run", json={"mall": "Mall of America", "floors": [1], "max_iterations": 1})
    assert resp1.status_code == 200
    assert started.wait(timeout=5), "first job's worker thread never entered run()"

    resp2 = client.post("/run", json={"mall": "Mall of America", "floors": [1], "max_iterations": 1})
    assert resp2.status_code == 409
    assert "already running" in resp2.json()["detail"]

    release.set()  # let the first job finish

    # the lock should free up once the first job's background thread exits
    resp3 = None
    for _ in range(30):
        resp3 = client.post("/run", json={"mall": "Mall of America", "floors": [1], "max_iterations": 1})
        if resp3.status_code == 200:
            break
        time.sleep(0.3)
    assert resp3 is not None and resp3.status_code == 200, resp3.json() if resp3 else None
