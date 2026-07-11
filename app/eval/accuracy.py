"""Accuracy evaluation, computed at the end of an orchestrator run.

1. Directory-agreement accuracy: recall/precision vs. the official
   directory (fuzzy name match), floor/category accuracy %.
2. Evidence-agreement score: average per-attribute confidence across
   published features, as a proxy for cross-source agreement.
3. Geometry validity rate: fraction of published features whose geometry
   confidence clears a "plausibly valid" threshold (a proxy -- full spatial
   rule re-checking needs floor topology, e.g. the corridor feature, which
   isn't persisted post-hoc since only stores are published).
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from app.agents.publication_review import GEOMETRY_MIN_CONFIDENCE
from app.eval.ground_truth import load_ground_truth_from_evidence

NAME_MATCH_THRESHOLD = 80
# Same bar Publication Review uses to gate publication, so this metric
# answers "of what we published, how much cleared our own geometry bar" --
# using a different (stricter) threshold here would be internally inconsistent.
GEOMETRY_VALID_CONFIDENCE = GEOMETRY_MIN_CONFIDENCE


def compute_accuracy_report(store, mall: str, floors: list[int], base_url: str = "https://www.mallofamerica.com") -> dict:
    # base_url is accepted for API stability but no longer used here -- see
    # load_ground_truth_from_evidence()'s docstring for why a fresh fetch
    # was dropped in favor of reusing this run's own collected evidence.
    ground_truth = load_ground_truth_from_evidence(store, mall, floors)
    published: list[dict] = []
    for floor in floors:
        published.extend(store.get_published_features(mall, floor))
    published_stores = [f for f in published if f["feature_type"] == "store"]

    published_names = [f["properties"].get("name", "") for f in published_stores]

    matched = 0
    floor_correct = 0
    category_correct = 0
    for gt in ground_truth:
        if not published_names:
            break
        result = process.extractOne(gt["name"], published_names, scorer=fuzz.ratio)
        if result and result[1] >= NAME_MATCH_THRESHOLD:
            matched += 1
            feature = published_stores[result[2]]
            if feature["properties"].get("floor") == gt.get("floor") or gt.get("floor") == feature.get("floor"):
                floor_correct += 1
            if feature["properties"].get("category") == gt.get("category"):
                category_correct += 1

    recall = matched / len(ground_truth) if ground_truth else 0.0
    precision = matched / len(published_stores) if published_stores else 0.0
    floor_accuracy = floor_correct / matched if matched else 0.0
    category_accuracy = category_correct / matched if matched else 0.0

    evidence_agreement_scores = [
        sum(f["confidence_by_attribute"].values()) / len(f["confidence_by_attribute"])
        for f in published_stores if f["confidence_by_attribute"]
    ]
    evidence_agreement_score = (
        sum(evidence_agreement_scores) / len(evidence_agreement_scores) if evidence_agreement_scores else 0.0
    )

    geometry_valid = [
        f for f in published_stores
        if f["confidence_by_attribute"].get("geometry", 0) >= GEOMETRY_VALID_CONFIDENCE
    ]
    geometry_validity_rate = len(geometry_valid) / len(published_stores) if published_stores else 0.0

    return {
        "ground_truth_count": len(ground_truth),
        "published_count": len(published_stores),
        "directory_agreement": {
            "recall": round(recall, 3), "precision": round(precision, 3),
            "floor_accuracy": round(floor_accuracy, 3), "category_accuracy": round(category_accuracy, 3),
        },
        "evidence_agreement_score": round(evidence_agreement_score, 3),
        "geometry_validity_rate": round(geometry_validity_rate, 3),
    }
