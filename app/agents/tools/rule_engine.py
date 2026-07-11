"""Declarative spatial/semantic rules, interpreted by the Validation Agent.
Adding a new feature-type rule means editing this table, not the validation
flow in agents/validation.py.
"""
from __future__ import annotations

from shapely.geometry import shape

RULES: dict[str, dict] = {
    "store": {
        "must_intersect": ["corridor"],
        "must_not_overlap": ["store"],
        "centroid_inside": ["floor"],
    },
    "escalator": {
        "connects": ["floor", "floor"],
    },
    "restroom": {
        "inside": ["floor"],
    },
}


def check_overlap(feature_geom: dict, other_geoms: list[dict]) -> bool:
    """True if feature_geom overlaps any geometry in other_geoms."""
    if feature_geom.get("type") != "Polygon":
        return False
    poly = shape(feature_geom)
    for g in other_geoms:
        if g.get("type") != "Polygon":
            continue
        if poly.overlaps(shape(g)) or poly.contains(shape(g)) or shape(g).contains(poly):
            return True
    return False


def check_intersects_any(feature_geom: dict, other_geoms: list[dict]) -> bool:
    poly_or_line = shape(feature_geom)
    return any(poly_or_line.intersects(shape(g)) for g in other_geoms)


def check_centroid_inside(feature_geom: dict, boundary_geom: dict) -> bool:
    poly = shape(feature_geom)
    boundary = shape(boundary_geom)
    return boundary.contains(poly.centroid)


def validate_feature(feature_type: str, geometry: dict | None, context: dict) -> list[str]:
    """Returns a list of rule violations (empty = passes all spatial rules).
    `context` supplies the geometries needed to check against:
    {"corridors": [...], "stores": [...], "floor_boundary": {...}}."""
    violations: list[str] = []
    rules = RULES.get(feature_type, {})
    if geometry is None:
        return violations  # nothing to check yet

    if "must_not_overlap" in rules and geometry.get("type") == "Polygon":
        if check_overlap(geometry, context.get("stores", [])):
            violations.append("overlaps another store polygon")

    if "must_intersect" in rules and geometry.get("type") == "Polygon":
        # Polygon-only, like the other checks: a real anchor position is a
        # labeled reference Point (see agents/tools/anchor_map.py), not a
        # room-shaped unit, and it lives in the map's own real coordinate
        # space rather than the synthetic grid's -- it was never going to
        # geometrically "touch" the synthetic corridor line, so without
        # this guard every anchor-matched store would incorrectly fail
        # this check regardless of how real its position actually is.
        corridors = context.get("corridors", [])
        if corridors and not check_intersects_any(geometry, corridors):
            violations.append("does not intersect any corridor")

    if "centroid_inside" in rules and context.get("floor_boundary"):
        if geometry.get("type") == "Polygon" and not check_centroid_inside(geometry, context["floor_boundary"]):
            violations.append("centroid lies outside floor boundary")

    return violations
