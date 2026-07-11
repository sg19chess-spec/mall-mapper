"""Loads the official store directory as one evaluation axis.

Note: in dev mode this is the same SAMPLE_DIRECTORY the Research Agent
itself falls back to when the live site can't be fetched, so recall in that
case measures pipeline fidelity (does entity resolution / confidence
thresholding lose or corrupt entities that were actually scraped) rather
than source correctness. Against a live fetch it becomes a genuine
external ground truth.
"""
from __future__ import annotations

from app.agents.tools.normalizer import normalize
from app.agents.tools.web import get_store_directory


def load_ground_truth_from_evidence(store, mall: str, floors: list[int]) -> list[dict]:
    """Reconstructs ground truth from the official_directory Evidence rows
    already collected during this run, rather than re-fetching the
    directory a second time. This is what compute_accuracy_report() uses:
    it's faster (no duplicate Playwright launch), network-independent for
    tests once evidence is already in the store, and compares against
    exactly what Research actually saw during this run rather than a
    possibly-different live fetch made moments later.

    Deduplicated by normalized store name, keeping the freshest row per
    store. Evidence persists across separate /run calls against the same
    (mall, floor) -- e.g. in production against Supabase, unlike the
    throwaway per-test SQLite DBs used in tests -- and Research always does
    a fresh broad scrape on VERIFY_EXISTENCE rather than skipping
    already-scraped floors, so repeated runs legitimately accumulate
    multiple official_directory rows for the same real store. Counting
    each row as a separate "true" store would inflate ground_truth_count
    and could push precision above the mathematically valid [0, 1] range.
    """
    by_store: dict[str, dict] = {}
    for floor in floors:
        for e in store.get_all_evidence(mall, floor):
            if e["source_type"] != "official_directory":
                continue
            key = normalize(e["entity_raw"])
            existing = by_store.get(key)
            if existing is not None and existing["published_date"] >= e["published_date"]:
                continue
            obs = e["observation"]
            by_store[key] = {
                "name": e["entity_raw"], "floor": obs.get("floor", floor),
                "category": obs.get("category"), "unit": obs.get("unit"),
                "published_date": e["published_date"],
            }
    return [{k: v for k, v in row.items() if k != "published_date"} for row in by_store.values()]


def load_ground_truth(base_url: str, floors: list[int]) -> list[dict]:
    """Standalone fresh fetch against the live site -- for manual/ad-hoc
    ground-truth checks outside of a pipeline run. compute_accuracy_report()
    does NOT use this by default; see load_ground_truth_from_evidence()."""
    stores = get_store_directory(base_url)
    return [s for s in stores if s.get("floor") in floors]
