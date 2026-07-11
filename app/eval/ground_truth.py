"""Loads the official store directory as one evaluation axis.

Note: in dev mode this is the same SAMPLE_DIRECTORY the Research Agent
itself falls back to when the live site can't be fetched, so recall in that
case measures pipeline fidelity (does entity resolution / confidence
thresholding lose or corrupt entities that were actually scraped) rather
than source correctness. Against a live fetch it becomes a genuine
external ground truth.
"""
from __future__ import annotations

from app.agents.tools.web import get_store_directory


def load_ground_truth_from_evidence(store, mall: str, floors: list[int]) -> list[dict]:
    """Reconstructs ground truth from the official_directory Evidence rows
    already collected during this run, rather than re-fetching the
    directory a second time. This is what compute_accuracy_report() uses:
    it's faster (no duplicate Playwright launch), network-independent for
    tests once evidence is already in the store, and compares against
    exactly what Research actually saw during this run rather than a
    possibly-different live fetch made moments later."""
    ground_truth = []
    for floor in floors:
        for e in store.get_all_evidence(mall, floor):
            if e["source_type"] != "official_directory":
                continue
            obs = e["observation"]
            ground_truth.append({
                "name": e["entity_raw"], "floor": obs.get("floor", floor),
                "category": obs.get("category"), "unit": obs.get("unit"),
            })
    return ground_truth


def load_ground_truth(base_url: str, floors: list[int]) -> list[dict]:
    """Standalone fresh fetch against the live site -- for manual/ad-hoc
    ground-truth checks outside of a pipeline run. compute_accuracy_report()
    does NOT use this by default; see load_ground_truth_from_evidence()."""
    stores = get_store_directory(base_url)
    return [s for s in stores if s.get("floor") in floors]
