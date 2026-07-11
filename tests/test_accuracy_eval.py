"""End-to-end accuracy eval tests: run the full 5-agent pipeline (dev-mode
SQLite store, no live network/API keys) and check the numbers app/eval/
accuracy.py reports -- the regression net for the Validation Agent changes
(agreement bonus, conflict classification, spatial reasoning, certainty
weighting) as they play out across the whole pipeline, not just in
isolation.

Each test gets its own throwaway SQLite file via monkeypatching
app.store.supabase._DEV_DB_PATH, so tests don't interfere with each other
or with a real ./dev_data database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import app.agents.tools.web as web_module
import app.store.supabase as supabase_module
from app.orchestrator import Orchestrator
from app.schemas import Evidence, RunConfig, SourceType


@pytest.fixture(autouse=True)
def force_offline_scraping(monkeypatch):
    """Keep this suite deterministic and network-independent: force the
    Research Agent's directory scraper straight to the bundled
    SAMPLE_DIRECTORY (10 stores across floors 1-3), regardless of whether
    live network/Playwright access happens to be available in whatever
    environment runs these tests. Without this, a sandbox with live
    internet access would silently start scraping the real Mall of America
    site mid-test-run, replacing the known 10-store fixture with however
    many real stores happen to be listed that day -- exactly the failure
    mode this fixture exists to prevent. Live scraping itself is verified
    separately (manually, against the real site), not in this suite."""
    monkeypatch.setattr(web_module, "fetch_directory_html", lambda *a, **k: None)
    monkeypatch.setattr(web_module, "fetch_rendered_html", lambda *a, **k: None)


def fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(supabase_module, "_DEV_DB_PATH", tmp_path / "test.db")
    store = supabase_module.Store()
    assert store.dev_mode
    return store


def test_full_mall_publishes_all_sample_stores_with_perfect_directory_agreement(tmp_path, monkeypatch):
    store = fresh_store(tmp_path, monkeypatch)
    orch = Orchestrator(store, "https://www.mallofamerica.com")
    config = RunConfig(mall="Mall of America", floors=[1, 2, 3], max_iterations=4)

    report = orch.run("pytest-full-run", config)
    acc = report["accuracy"]

    assert acc["published_count"] == acc["ground_truth_count"] > 0
    assert acc["directory_agreement"]["recall"] == 1.0
    assert acc["directory_agreement"]["precision"] == 1.0
    assert acc["directory_agreement"]["floor_accuracy"] == 1.0
    assert acc["directory_agreement"]["category_accuracy"] == 1.0
    # evidence-agreement score should sit comfortably above zero now that
    # multi-source corroboration (directory + floor plan + web + YouTube
    # transcript) and the agreement bonus are both contributing
    assert acc["evidence_agreement_score"] > 0.5
    assert acc["geometry_validity_rate"] == 1.0
    assert report["human_review_queue_size"] == 0


def test_pipeline_converges_without_hitting_iteration_cap(tmp_path, monkeypatch):
    store = fresh_store(tmp_path, monkeypatch)
    orch = Orchestrator(store, "https://www.mallofamerica.com")
    config = RunConfig(mall="Mall of America", floors=[1, 2, 3], max_iterations=6)

    report = orch.run("pytest-convergence-run", config)

    # a healthy run should converge well before the safety-net cap
    assert report["iterations_run"] < config.max_iterations
    assert report["iteration_log"][-1]["queue_size_next"] == 0


def test_conflicting_evidence_is_held_back_and_reflected_in_accuracy(tmp_path, monkeypatch):
    store = fresh_store(tmp_path, monkeypatch)
    orch = Orchestrator(store, "https://www.mallofamerica.com")
    config = RunConfig(mall="Mall of America", floors=[1], max_iterations=4)

    # inject a conflicting recent claim for a floor-1 store before running
    bad = Evidence(
        source_type=SourceType.SOCIAL, source_url="https://social.invalid/post/1",
        entity_raw="LEGO Store", observation={"floor": 5}, raw_excerpt="injected conflicting evidence",
        published_date=datetime.now(timezone.utc),
    )
    store.insert_evidence(bad.model_dump(mode="json"), "Mall of America", 1)

    report = orch.run("pytest-conflict-run", config)
    acc = report["accuracy"]

    # the conflicted store should be held back from publication...
    assert acc["published_count"] == acc["ground_truth_count"] - 1
    assert report["human_review_queue_size"] == 1

    review_queue = store.get_review_queue("open")
    assert len(review_queue) == 1
    assert "lego" in review_queue[0]["feature_id"].lower()
    assert "floor" in review_queue[0]["issue"]


def test_temporal_conflict_from_old_evidence_does_not_block_other_stores(tmp_path, monkeypatch):
    store = fresh_store(tmp_path, monkeypatch)
    orch = Orchestrator(store, "https://www.mallofamerica.com")
    config = RunConfig(mall="Mall of America", floors=[2], max_iterations=4)

    old_conflict = Evidence(
        source_type=SourceType.WEB, source_url="https://old-listing.invalid/apple",
        entity_raw="Apple", observation={"floor": 1}, raw_excerpt="stale listing",
        published_date=datetime.now(timezone.utc) - timedelta(days=400),
    )
    store.insert_evidence(old_conflict.model_dump(mode="json"), "Mall of America", 2)

    report = orch.run("pytest-temporal-run", config)
    acc = report["accuracy"]

    # Apple gets held back for review, but its floor-2 neighbors (Nike,
    # Build-A-Bear, Sephora) should still publish normally -- a conflict on
    # one entity must not stall the whole floor.
    assert acc["published_count"] == acc["ground_truth_count"] - 1
    assert report["human_review_queue_size"] == 1

    apple_reports = [
        r for r in store.get_review_queue("open") if "apple" in r["feature_id"].lower()
    ]
    assert len(apple_reports) == 1
