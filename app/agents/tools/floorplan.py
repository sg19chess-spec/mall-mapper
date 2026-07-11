"""Downloads a mall's per-floor plan image for the Research Agent to pass to
ocr.py and geometry.py. Falls back to a small synthetic floor-plan grid when
no real image is reachable, so the OCR/geometry/spatial-validation path is
still exercisable offline.
"""
from __future__ import annotations

import httpx

from app.agents.tools.web import USER_AGENT


def fetch_floorplan_image(base_url: str, floor: int, timeout: float = 10.0) -> bytes | None:
    candidate_paths = [
        f"/directory/map/level-{floor}",
        f"/maps/floor-{floor}.png",
        f"/floorplans/{floor}.png",
    ]
    for path in candidate_paths:
        try:
            resp = httpx.get(
                f"{base_url.rstrip('/')}{path}",
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
        except Exception:
            continue
    return None


def synthetic_floorplan_grid(floor: int, store_count: int) -> dict:
    """A deterministic fallback "floor plan": a corridor spine with stores
    laid out along it in unit-number order, used when no real floor plan
    image is reachable. Coordinates are floor-local pixel space."""
    corridor_y = 200
    spacing = 80
    stores = []
    for i in range(store_count):
        x = 100 + i * spacing
        # y_bottom sits flush on the corridor line so the store polygon
        # actually touches/intersects the corridor (required by the
        # "must_intersect: [corridor]" spatial rule).
        stores.append({"x": x, "y_top": corridor_y - 60, "y_bottom": corridor_y})
    return {
        "floor": floor,
        "corridor": {"y": corridor_y, "x_start": 60, "x_end": 100 + store_count * spacing},
        "store_slots": stores,
    }
