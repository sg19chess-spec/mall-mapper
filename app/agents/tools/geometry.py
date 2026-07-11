"""Builds typed GeoJSON-style geometry (Point/Polygon/LineString) for an
IndoorFeature from OCR text positions + the floor plan's corridor layout.
Used by the Indoor Mapping Agent.
"""
from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon, mapping


def store_polygon_from_slot(slot: dict) -> dict:
    """A rectangular store footprint from a floorplan.py store_slot entry."""
    x = slot["x"]
    poly = Polygon([
        (x - 30, slot["y_top"]), (x + 30, slot["y_top"]),
        (x + 30, slot["y_bottom"]), (x - 30, slot["y_bottom"]),
    ])
    return {"type": "Polygon", "coordinates": mapping(poly)["coordinates"]}


def corridor_linestring(corridor: dict) -> dict:
    line = LineString([(corridor["x_start"], corridor["y"]), (corridor["x_end"], corridor["y"])])
    return {"type": "LineString", "coordinates": list(line.coords)}


def entrance_point(x: float, y: float) -> dict:
    pt = Point(x, y)
    return {"type": "Point", "coordinates": list(pt.coords)[0]}


def geometry_confidence(source_has_official_floorplan: bool, ocr_confidence: float | None) -> float:
    """Geometry gets its own confidence, separate from identity confidence --
    the directory can be very sure a store exists; the polygon shape is
    always approximate."""
    base = 0.6 if source_has_official_floorplan else 0.35
    if ocr_confidence:
        base = min(0.98, base + 0.35 * ocr_confidence)
    return round(base, 2)
