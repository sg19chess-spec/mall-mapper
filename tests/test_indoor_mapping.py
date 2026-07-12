"""Tests for the Indoor Mapping Agent's real-position-only geometry
selection (app/agents/indoor_mapping.py).

A store is placed only from a real anchor-DOM match or a real OCR'd map
label; otherwise it is left unplaced (geometry None) -- there is no
synthetic fallback. Anchor landmarks are also emitted as their own
reference features so the map always has a real backbone. These tests
exercise IndoorMappingAgent as a pure function of its inputs -- no
store/network dependency, using a minimal fake store that just reports
"no previous version" for every feature.
"""
from __future__ import annotations

from app.agents.indoor_mapping import (
    ANCHOR_GEOMETRY_CONFIDENCE,
    IndoorMappingAgent,
)


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


def make_floorplan_evidence(anchor_positions: dict | None, ocr_positions: list | None = None) -> dict:
    return {
        "observation": {
            "ocr_results": [],
            "anchor_positions": anchor_positions,
            "ocr_positions": ocr_positions or [],
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

OCR_POSITIONS = [
    {"text": "Sephora", "x": 300.0, "y": 500.0, "confidence": 0.9},
    {"text": "Claires", "x": 420.0, "y": 640.0, "confidence": 0.7},
]


def test_anchor_matched_store_gets_real_point_geometry():
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)
    floorplan_evidence = make_floorplan_evidence(ANCHORS)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    store_feature = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert store_feature["geometry"]["type"] == "Point"
    assert tuple(store_feature["geometry"]["coordinates"]) == (850.5, 1020.0)
    assert store_feature["properties"]["geometry_source"] == "real_anchor"
    assert store_feature["properties"]["matched_anchor_name"] == "Nordstrom"
    assert store_feature["confidence_by_attribute"]["geometry"] == ANCHOR_GEOMETRY_CONFIDENCE


def test_ocr_matched_store_gets_real_point_from_ocr_label():
    # a store that isn't an anchor but whose name was OCR'd off the rendered
    # map gets a real Point at the OCR label position (not a synthetic one)
    entities = {"sephora": make_entity("Sephora")}
    validation_result = make_validation_result(entities)
    floorplan_evidence = make_floorplan_evidence(ANCHORS, OCR_POSITIONS)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    f = next(f for f in features if f["_canonical_key"] == "sephora")
    assert f["geometry"]["type"] == "Point"
    assert tuple(f["geometry"]["coordinates"]) == (300.0, 500.0)
    assert f["properties"]["geometry_source"] == "ocr_label"
    assert f["properties"]["ocr_text"] == "Sephora"
    assert 0 < f["confidence_by_attribute"]["geometry"] < ANCHOR_GEOMETRY_CONFIDENCE


def test_anchor_takes_precedence_over_ocr():
    # if a store matches both an anchor and an OCR label, the (more reliable)
    # anchor DOM position wins
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)
    ocr = OCR_POSITIONS + [{"text": "Nordstrom", "x": 10.0, "y": 20.0, "confidence": 0.9}]
    floorplan_evidence = make_floorplan_evidence(ANCHORS, ocr)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    f = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert f["properties"]["geometry_source"] == "real_anchor"
    assert tuple(f["geometry"]["coordinates"]) == (850.5, 1020.0)


def test_unmatched_store_is_unplaced_no_synthetic_fallback():
    entities = {"claires": make_entity("Some Tiny Store")}
    validation_result = make_validation_result(entities)
    floorplan_evidence = make_floorplan_evidence(ANCHORS, [])  # no OCR match either

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, floorplan_evidence)

    f = next(f for f in features if f["_canonical_key"] == "claires")
    assert f["geometry"] is None
    assert f["properties"]["geometry_source"] == "unplaced"
    # geometry key is omitted entirely so Publication Review's geometry gate
    # treats it as "nothing to check" rather than "geometry failed"
    assert "geometry" not in f["confidence_by_attribute"]


def test_no_floorplan_evidence_at_all_leaves_store_unplaced():
    entities = {"nordstrom": make_entity("Nordstrom")}
    validation_result = make_validation_result(entities)

    agent = IndoorMappingAgent(FakeStore())
    features = agent.run("Mall of America", 2, validation_result, None)

    f = next(f for f in features if f["_canonical_key"] == "nordstrom")
    assert f["geometry"] is None
    assert f["properties"]["geometry_source"] == "unplaced"


# ---------------------------------------------------------------------------
# build_anchor_features: the map's real reference backbone
# ---------------------------------------------------------------------------

def test_build_anchor_features_emits_real_reference_points():
    agent = IndoorMappingAgent(FakeStore())
    floorplan_evidence = make_floorplan_evidence({**ANCHORS, "live": True})
    feats = agent.build_anchor_features("Mall of America", 2, floorplan_evidence)

    assert len(feats) == 2
    nord = next(f for f in feats if f["properties"]["name"] == "Nordstrom")
    assert nord["feature_type"] == "anchor"
    assert nord["geometry"]["type"] == "Point"
    assert tuple(nord["geometry"]["coordinates"]) == (850.5, 1020.0)
    assert nord["properties"]["geometry_source"] == "real_anchor_reference"
    assert nord["properties"]["anchor_view_box"] == ANCHORS["view_box"]
    assert nord["properties"]["live_capture"] is True
    assert nord["confidence_by_attribute"]["geometry"] == ANCHOR_GEOMETRY_CONFIDENCE


def test_build_anchor_features_empty_without_anchor_data():
    agent = IndoorMappingAgent(FakeStore())
    assert agent.build_anchor_features("Mall of America", 2, None) == []
    assert agent.build_anchor_features("Mall of America", 2, make_floorplan_evidence(None)) == []
