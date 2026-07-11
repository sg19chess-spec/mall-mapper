"""Unit tests for the declarative spatial rule engine
(app/agents/tools/rule_engine.py) that Publication Review's geometry checks
are built on: must_not_overlap, must_intersect, and centroid_inside, plus
the "no rules for this feature type" / "no geometry yet" no-ops.
"""
from __future__ import annotations

from app.agents.tools.rule_engine import validate_feature


def polygon(x_min: float, y_min: float, x_max: float, y_max: float) -> dict:
    return {"type": "Polygon", "coordinates": [[
        (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min),
    ]]}


def line(x_start: float, x_end: float, y: float) -> dict:
    return {"type": "LineString", "coordinates": [(x_start, y), (x_end, y)]}


EMPTY_CONTEXT = {"stores": [], "corridors": [], "floor_boundary": None}


# ---------------------------------------------------------------------------
# must_not_overlap
# ---------------------------------------------------------------------------

def test_flags_overlapping_store_polygon():
    geom = polygon(0, 0, 60, 60)
    other = polygon(30, 30, 90, 90)  # overlaps the [0,60]x[0,60] box
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "stores": [other]})
    assert "overlaps another store polygon" in violations


def test_ok_when_disjoint_from_other_stores():
    geom = polygon(0, 0, 60, 60)
    other = polygon(200, 200, 260, 260)
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "stores": [other]})
    assert violations == []


def test_flags_one_store_fully_containing_another():
    geom = polygon(10, 10, 20, 20)
    other = polygon(0, 0, 100, 100)  # fully contains geom
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "stores": [other]})
    assert "overlaps another store polygon" in violations


def test_adjacent_but_touching_stores_are_not_flagged_as_overlap():
    # sharing only an edge (touching, not overlapping) should be fine --
    # e.g. two stores with a shared partition wall
    geom = polygon(0, 0, 60, 60)
    other = polygon(60, 0, 120, 60)
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "stores": [other]})
    assert "overlaps another store polygon" not in violations


# ---------------------------------------------------------------------------
# must_intersect corridor
# ---------------------------------------------------------------------------

def test_flags_store_not_touching_any_corridor():
    geom = polygon(0, 0, 60, 60)  # spans y in [0, 60]
    corridor = line(0, 500, 200)  # far away at y=200
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "corridors": [corridor]})
    assert "does not intersect any corridor" in violations


def test_ok_when_store_touches_corridor():
    geom = polygon(0, 140, 60, 200)  # bottom edge sits exactly on y=200
    corridor = line(0, 500, 200)
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "corridors": [corridor]})
    assert "does not intersect any corridor" not in violations


def test_ok_when_store_touches_at_least_one_of_several_corridors():
    geom = polygon(0, 140, 60, 200)
    far_corridor = line(0, 500, 9000)
    touching_corridor = line(0, 500, 200)
    violations = validate_feature(
        "store", geom, {**EMPTY_CONTEXT, "corridors": [far_corridor, touching_corridor]},
    )
    assert violations == []


def test_skips_corridor_check_when_no_corridors_supplied():
    # absence of corridor data isn't treated as a violation -- only an
    # explicit corridor that fails to intersect is
    geom = polygon(0, 0, 60, 60)
    violations = validate_feature("store", geom, EMPTY_CONTEXT)
    assert violations == []


# ---------------------------------------------------------------------------
# centroid_inside floor boundary
# ---------------------------------------------------------------------------

def test_flags_centroid_outside_floor_boundary():
    geom = polygon(1000, 1000, 1060, 1060)  # centroid ~(1030, 1030)
    boundary = polygon(0, 0, 200, 200)
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "floor_boundary": boundary})
    assert "centroid lies outside floor boundary" in violations


def test_ok_when_centroid_inside_floor_boundary():
    geom = polygon(50, 50, 110, 110)  # centroid (80, 80)
    boundary = polygon(0, 0, 200, 200)
    violations = validate_feature("store", geom, {**EMPTY_CONTEXT, "floor_boundary": boundary})
    assert violations == []


def test_skips_centroid_check_when_no_boundary_supplied():
    geom = polygon(1000, 1000, 1060, 1060)
    violations = validate_feature("store", geom, EMPTY_CONTEXT)
    assert violations == []


# ---------------------------------------------------------------------------
# no-ops
# ---------------------------------------------------------------------------

def test_no_violations_when_geometry_is_none():
    assert validate_feature("store", None, {**EMPTY_CONTEXT, "stores": [polygon(0, 0, 10, 10)]}) == []


def test_no_violations_for_feature_type_without_rules():
    geom = polygon(0, 0, 60, 60)
    other = polygon(30, 30, 90, 90)  # would overlap if this were a "store"
    violations = validate_feature("entrance", geom, {**EMPTY_CONTEXT, "stores": [other]})
    assert violations == []


def test_multiple_violations_reported_together():
    geom = polygon(1000, 1000, 1060, 1060)
    overlapping_store = polygon(1030, 1030, 1090, 1090)
    far_corridor = line(0, 60, 0)
    boundary = polygon(0, 0, 200, 200)
    violations = validate_feature("store", geom, {
        "stores": [overlapping_store], "corridors": [far_corridor], "floor_boundary": boundary,
    })
    assert set(violations) == {
        "overlaps another store polygon",
        "does not intersect any corridor",
        "centroid lies outside floor boundary",
    }
