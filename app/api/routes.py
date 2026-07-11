from __future__ import annotations

import os
import threading
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.orchestrator import Orchestrator
from app.schemas import RunConfig, Subtask, TaskType
from app.store.storage import get_storage
from app.store.supabase import get_store

router = APIRouter()

BASE_URL = os.environ.get("MALL_BASE_URL", "https://www.mallofamerica.com")


class RunRequest(BaseModel):
    mall: str = "Mall of America"
    base_url: str | None = None  # e.g. "https://www.mallofamerica.com" -- defaults to MALL_BASE_URL env var
    floors: list[int] = [1]
    max_iterations: int = 6


@router.post("/run")
def run_job(req: RunRequest):
    job_id = str(uuid4())
    config = RunConfig(mall=req.mall, floors=req.floors, max_iterations=req.max_iterations)
    store = get_store()
    orchestrator = Orchestrator(store, req.base_url or BASE_URL)

    def _worker():
        try:
            report = orchestrator.run(job_id, config)
        except Exception as exc:  # keep job status queryable even on failure
            store.update_job(job_id, status="failed", report={"error": str(exc)})
            return

        # orchestrator.run() already marked the job "completed" with its
        # real report as its last step. A failure in this export step
        # (e.g. a missing Storage bucket) is a separate, lower-stakes
        # concern -- it must not clobber a successful pipeline run's
        # status/report, so it's logged instead of re-raised into the
        # same except block that handles orchestrator failures.
        try:
            _export_geojson(store, config.mall, config.floors)
            get_storage().put_json("reports", f"{job_id}/run_report.json", report)
        except Exception as exc:
            store.log_audit(job_id, 0, "export_failed", detail={"error": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _export_geojson(store, mall: str, floors: list[int]) -> None:
    storage = get_storage()
    for floor in floors:
        features = store.get_published_features(mall, floor)
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": f.get("geometry"),
                    "properties": {
                        **f["properties"],
                        "feature_id": f["feature_id"],
                        "feature_type": f["feature_type"],
                        "confidence_by_attribute": f["confidence_by_attribute"],
                        "evidence": f["evidence"],
                        "version": f["version"],
                    },
                }
                for f in features
            ],
        }
        storage.put_json("geojson", f"{mall}/floor_{floor}.geojson", geojson)


@router.get("/geojson/{floor}")
def get_geojson(floor: int, mall: str = "Mall of America"):
    storage = get_storage()
    data = storage.get_json("geojson", f"{mall}/floor_{floor}.geojson")
    if data is None:
        store = get_store()
        features = store.get_published_features(mall, floor)
        if not features:
            raise HTTPException(404, "no published features for this floor yet -- run a job first")
        _export_geojson(store, mall, [floor])
        data = storage.get_json("geojson", f"{mall}/floor_{floor}.geojson")
    return data


@router.get("/feature/{feature_id}")
def get_feature(feature_id: str):
    store = get_store()
    history = store.get_feature_history(feature_id)
    if not history:
        raise HTTPException(404, "feature not found")
    return {"feature_id": feature_id, "versions": history}


@router.get("/review-queue")
def get_review_queue():
    store = get_store()
    return store.get_review_queue("open")


@router.get("/audit/{feature_id}")
def get_audit(feature_id: str, job_id: str):
    store = get_store()
    trail = store.get_audit_trail(job_id)
    return [row for row in trail if row.get("feature_id") == feature_id]


@router.get("/jobs/{job_id}/trail")
def get_job_trail(job_id: str):
    """Full agent-activity timeline for a job -- every evidence-collected
    and review-decision event across all features/floors, in order. Powers
    the live activity feed in the /ui frontend; unlike GET /audit/{feature_id}
    this isn't filtered down to one feature."""
    store = get_store()
    return store.get_audit_trail(job_id)


@router.post("/rerun/{feature_id}")
def rerun_feature(feature_id: str, mall: str, floor: int, task_type: TaskType = TaskType.VERIFY_EXISTENCE):
    store = get_store()
    orchestrator = Orchestrator(store, BASE_URL)
    parts = feature_id.split(":")
    entity_hint = parts[-1].replace("_", " ") if len(parts) >= 3 else feature_id
    subtask = Subtask(mall=mall, floor=floor, entity_hint=entity_hint, task_type=task_type, priority="high")
    evidence_list = orchestrator.research.run(subtask)
    for ev in evidence_list:
        store.insert_evidence(ev.model_dump(mode="json"), mall, floor)
    return {"feature_id": feature_id, "evidence_collected": len(evidence_list)}
