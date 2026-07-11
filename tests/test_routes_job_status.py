"""Regression test for a bug found in production: a downstream export
failure (e.g. a missing Supabase Storage bucket) after a successful
orchestrator run was overwriting the job's "completed" status and real
report with a bare {"error": ...} -- losing the actual pipeline result.
Confirmed live on Render: evidence/indoor_features/review_reports were all
correctly written to Postgres, but /status/{job_id} reported "failed"
because the GeoJSON export step (a separate concern) threw afterward.
"""
from __future__ import annotations

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
