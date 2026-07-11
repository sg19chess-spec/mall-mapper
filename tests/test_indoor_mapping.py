"""Tests for the Indoor Mapping Agent's real-anchor vs synthetic-placeholder
geometry selection (app/agents/indoor_mapping.py).

For major anchor tenants (Nordstrom, JW Marriott, etc.) the mall's own live
map exposes real coordinates (see agents/tools/anchor_map.py); everything
else still falls back to the synthetic corridor-grid placeholder. These
tests exercise IndoorMappingAgent.run() as a pure function of its inputs
-- no store/network dependency, using a minimal fake store that just
reports "no previous version" for every feature.
"""
from __future__ import annotations

from app.agents.indoor_mapping import (
    ANCHOR_GEOMETRY_CONFIDENCE,
    IndoorMappingAgent,
)
from app.agents.tools.floorplan import synthetic_floorplan_grid


class FakeStore:
    def get_feature_history(self, feature_id: str) -> list[dict]:
        return []


def make_entity(raw_name: str, *, category: str | None = None, unit: str | None = None,
                 existence_confidence: float = 0.9) -> dict:
    fields = {}
    if category is not None:
        fields["category"] = {"value": category, "confidence": 0.8}
    if unit is not None:
        fields["unit"] = {"value": unit, "confidence": 0.8}
    return {
        "raw_name": raw_name,
        "fields": fields,
        "existence_confidence": existence_confidence,
        "evidence_refs": [],
        "explanation": [],
    }


def make_validation_result(entities: dict[str, dict]) -> dict:
    return {"entities": entities, "conflicts": []}


ANCHORS = {
    "view_box": [0, 0, 2000, 2000],
    "anchors": [
        {"name": "Nordstrom", "x": 850.5, "y": 1020.0},
        {"name": "JW Marriott Mall of America", "x": 1200.0, "y": 400.0},
    ],
}


def make_floorplan_evidence(anchor_positions: dict | None, grid: dict | None) -> dict:
    return {
        "observation": {
            "ocr_results": [],
            "synthetic_grid": grid,
            "anchor_positions": anchor_positions,
        }
    }


# ---------------------------------------------------------------------------
# _match_anchor
# ---------------------------------------------------------------------------

def test_match_anchor_exact_name():
    match = IndoorMappingAgent._match_anchor("Nordstrom", ANCHORS["anchors"])
    assert match is not None
    assert match["name"] == "Nordstrom"


def test_match_anchor_fuzzy_name():
    # directory listings often have slightly different formatting/casing
    match = IndoorMappingAgent._match_anchor("nordstrom", ANCHORS["anchors"])
    assert match is not None
    assert match["name"] == "Nordstrom"


def test_match_anchor_no_match_for_unrelated_store():
    match = IndoorMappingAgent._match_anchor("Claire's Boutique", ANCHORS["anchors"])
    assert match is None


def test_match_anchor_returns_none_for_empty_anchor_list():
    assert IndoorMappingAgent._match_anchor("Nordstrom", []) is None


# ---------------------------------------------------------------------------
# run(): geometry source selection
# ---------------------------------------------------------------------------

def test_anchor_matched_store_gets_real_point_geometry():
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)
    grid = synthetic_floorplan_grid(floor=2, store_count=1)
    floorplan_evidence = make_floorplan_evidence(ANCHORS, grid)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    store_feature = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert store_feature["geometry"]["type"] == "Point"
    assert tuple(store_feature["geometry"]["coordinates"]) == (850.5, 1020.0)
    assert store_feature["properties"]["geometry_source"] == "real_anchor"
    assert store_feature["properties"]["matched_anchor_name"] == "Nordstrom"
    assert store_feature["confidence_by_attribute"]["geometry"] == ANCHOR_GEOMETRY_CONFIDENCE


def test_non_anchor_store_falls_back_to_synthetic_polygon():
    entities = {"claires": make_entity("Claire's Boutique")}
    validation_result = make_validation_result(entities)
    grid = synthetic_floorplan_grid(floor=2, store_count=1)
    floorplan_evidence = make_floorplan_evidence(ANCHORS, grid)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    store_feature = next(f for f in features if f["_canonical_key"] == "claires")
    assert store_feature["geometry"]["type"] == "Polygon"
    assert store_feature["properties"]["geometry_source"] == "synthetic_placeholder"
    assert store_feature["confidence_by_attribute"]["geometry"] < ANCHOR_GEOMETRY_CONFIDENCE


def test_anchor_matched_entity_does_not_consume_a_synthetic_slot():
    # only one synthetic slot is available; an anchor-matched entity should
    # not consume it, leaving it free for the non-anchor entity
    entities = {
        "nordstrom": make_entity("Nordstrom"),
        "claires": make_entity("Claire's Boutique"),
    }
    validation_result = make_validation_result(entities)
    grid = synthetic_floorplan_grid(floor=2, store_count=1)  # only 1 slot
    floorplan_evidence = make_floorplan_evidence(ANCHORS, grid)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    nordstrom = next(f for f in features if f["_canonical_key"] == "nordstrom")
    claires = next(f for f in features if f["_canonical_key"] == "claires")
    assert nordstrom["properties"]["geometry_source"] == "real_anchor"
    assert claires["properties"]["geometry_source"] == "synthetic_placeholder"
    assert claires["geometry"] is not None  # got the one available slot


def test_no_anchor_data_falls_back_to_synthetic_for_all_stores():
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)
    grid = synthetic_floorplan_grid(floor=2, store_count=1)
    floorplan_evidence = make_floorplan_evidence(None, grid)  # anchor fetch failed/unavailable

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    store_feature = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert store_feature["properties"]["geometry_source"] == "synthetic_placeholder"


def test_no_floorplan_evidence_at_all_still_produces_features_without_geometry():
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, None)

    store_feature = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert store_feature["geometry"] is None
    assert store_feature["properties"]["geometry_source"] == "synthetic_placeholder"
