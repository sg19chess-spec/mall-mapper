"""Shapely STRtree spatial index over currently-published geometry, for fast
"which stores intersect Corridor C" / "nearest restroom" queries. A derived,
rebuildable index -- Postgres remains the source of record.
"""
from __future__ import annotations

from shapely.geometry import shape
from shapely.strtree import STRtree


class SpatialIndex:
    def __init__(self, features: list[dict]) -> None:
        self._features = [f for f in features if f.get("geometry")]
        self._geoms = [shape(f["geometry"]) for f in self._features]
        self._tree = STRtree(self._geoms) if self._geoms else None

    def query_intersects(self, geometry: dict) -> list[dict]:
        if self._tree is None:
            return []
        target = shape(geometry)
        hits = self._tree.query(target)
        return [self._features[i] for i in hits if target.intersects(self._geoms[i])]

    def nearest(self, geometry: dict) -> dict | None:
        if self._tree is None:
            return None
        target = shape(geometry)
        idx = self._tree.nearest(target)
        return self._features[idx] if idx is not None else None
