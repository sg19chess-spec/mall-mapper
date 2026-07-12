"""Extracts real anchor-tenant positions from Mall of America's live
directory map, via Playwright.

Confirmed live (see comments below): only the ~11 major anchors
(Nordstrom, Macy's, JW Marriott, Nickelodeon Universe, Sea Life, Crayola
Experience, Radisson BLU, food courts, parking, Transit Center) are
present in the page's static SVG markup at all. Individual small tenant
stores (Nike, Apple, H&M, etc.) are rendered by Jibestream's proprietary
canvas/WebGL map engine with no accessible DOM position data -- confirmed
by searching for "Nike" on the live map and observing #map_root (where
tenant markers would be injected) stays empty. So this deliberately only
covers anchors; small stores keep the synthetic placeholder grid in
floorplan.py. Real positions for the anchors are still valuable: they're
real, are exactly the well-known "you are here" reference points a human
would use, and let the map show *something* true rather than 100%
placeholder.

The map SVG uses internal numeric "level" IDs, not floor numbers directly
-- confirmed by reading each floor-switcher button's data-floor attribute:
"1" -> 1804, "2" -> 1805, "3" -> 1806, "4" -> 1807, "T" -> 3347 (Transit,
not modeled here since it's not one of our numbered floors). Each map
label element carries one or more `lvl-<id>` attributes marking which
level(s) it's visible on.
"""
from __future__ import annotations

import html
import re
import sys

FLOOR_TO_LEVEL_ID = {1: "1804", 2: "1805", 3: "1806", 4: "1807"}

# Real anchor coordinates captured from Mall of America's own live map
# (viewBox space), per floor. These are genuine measured positions, not
# fabricated ones -- they only change when the mall re-lays-out its map, so
# they're safe to cache. Used as a last-known-real fallback when the live
# WebGL map can't be driven in a given environment (e.g. a resource-limited
# container where the second headless-Chromium map session times out), so
# the map always has its real anchor backbone rather than rendering empty.
CACHED_ANCHORS: dict[int, dict] = {
    1: {"view_box": [640.0, 880.0, 400.0, 300.0], "anchors": [
        {"name": "West Parking", "x": 675.0, "y": 1043.0},
        {"name": "East Parking", "x": 1006.0, "y": 1043.0},
        {"name": "Huntington Bank Rotunda", "x": 926.0, "y": 1043.0},
        {"name": "Nickelodeon Universe", "x": 846.0, "y": 1043.0},
        {"name": "Macy's", "x": 739.0, "y": 1134.0},
        {"name": "Nordstrom", "x": 745.0, "y": 955.0},
        {"name": "JW Marriott", "x": 882.0, "y": 911.0},
        {"name": "Radisson BLU", "x": 863.0, "y": 1164.0},
    ]},
    2: {"view_box": [640.0, 880.0, 400.0, 300.0], "anchors": [
        {"name": "West Parking", "x": 675.0, "y": 1043.0},
        {"name": "East Parking", "x": 1006.0, "y": 1043.0},
        {"name": "Huntington Bank Rotunda", "x": 926.0, "y": 1043.0},
        {"name": "Nickelodeon Universe", "x": 846.0, "y": 1043.0},
        {"name": "Macy's", "x": 739.0, "y": 1134.0},
        {"name": "Nordstrom", "x": 745.0, "y": 955.0},
        {"name": "JW Marriott", "x": 882.0, "y": 911.0},
        {"name": "Radisson BLU", "x": 863.0, "y": 1164.0},
    ]},
    3: {"view_box": [640.0, 880.0, 400.0, 300.0], "anchors": [
        {"name": "West Parking", "x": 675.0, "y": 1043.0},
        {"name": "East Parking", "x": 1006.0, "y": 1043.0},
        {"name": "Huntington Bank Rotunda", "x": 926.0, "y": 1043.0},
        {"name": "North Food Court", "x": 840.0, "y": 935.0},
        {"name": "Nickelodeon Universe", "x": 846.0, "y": 1043.0},
        {"name": "Crayola Experience", "x": 952.0, "y": 1134.0},
        {"name": "Macy's", "x": 739.0, "y": 1134.0},
        {"name": "Nordstrom", "x": 745.0, "y": 955.0},
        {"name": "JW Marriott", "x": 882.0, "y": 911.0},
        {"name": "Radisson BLU", "x": 863.0, "y": 1164.0},
    ]},
    4: {"view_box": [640.0, 880.0, 400.0, 300.0], "anchors": [
        {"name": "West Parking", "x": 675.0, "y": 1043.0},
        {"name": "East Parking", "x": 1006.0, "y": 1043.0},
        {"name": "JW Marriott", "x": 882.0, "y": 911.0},
        {"name": "Radisson BLU", "x": 863.0, "y": 1164.0},
    ]},
}


def is_moa(base_url: str) -> bool:
    """The cached anchor coordinates are Mall of America's own map positions,
    so they must only ever be applied to MOA -- never injected onto some
    other mall whose base_url happens to be passed in."""
    return "mallofamerica" in (base_url or "").lower()


def cached_anchor_positions(floor: int) -> dict | None:
    """The last-known-real anchor set for a floor (see CACHED_ANCHORS),
    shaped like a live capture but with no screenshot. Returns None for a
    floor we have no cached data for."""
    entry = CACHED_ANCHORS.get(floor)
    if entry is None:
        return None
    return {
        "view_box": list(entry["view_box"]),
        "anchors": [dict(a) for a in entry["anchors"]],
        "map_png": None, "svg_px": None, "live": False,
    }

_VIEWBOX_PATTERN = re.compile(r'viewBox="([\d.\s-]+)"')
_DEFAULT_X_PATTERN = re.compile(r"foreignObject\s*\{[^}]*?x:\s*(-?\d+(?:\.\d+)?)px")
_DEFAULT_Y_PATTERN = re.compile(r"foreignObject\s*\{[^}]*?y:\s*(-?\d+(?:\.\d+)?)px")
_FOREIGN_OBJECT_PATTERN = re.compile(r"<foreignObject\b([^>]*)>(.*?)</foreignObject>", re.DOTALL)
_STYLE_ATTR_PATTERN = re.compile(r'style="([^"]*)"')
_ALT_PATTERN = re.compile(r'alt="([^"]*)"')
_X_PATTERN = re.compile(r"x:\s*(-?\d+(?:\.\d+)?)px")
_Y_PATTERN = re.compile(r"y:\s*(-?\d+(?:\.\d+)?)px")


def fetch_anchor_positions(base_url: str, floor: int, timeout_ms: int = 30000) -> dict | None:
    """Returns {"view_box": [minx, miny, w, h], "anchors": [{"name", "x", "y"}, ...],
    "map_png": bytes|None, "svg_px": [w, h]|None} for the given floor, or None
    if the map/floor couldn't be reached.

    In the same browser session that reads the anchor SVG, this also
    screenshots the rendered #map_svg element and records its on-screen
    pixel size -- so OCR run on that real screenshot (see
    ocr_positions_from_capture) yields label positions convertible into the
    same viewBox coordinate space the anchors live in."""
    level_id = FLOOR_TO_LEVEL_ID.get(floor)
    if level_id is None:
        return None
    # cached fallback only applies to MOA (the coordinates are its map's)
    fallback = cached_anchor_positions(floor) if is_moa(base_url) else None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"[anchor_map] playwright not installed, using cached anchors: {exc}", file=sys.stderr)
        return fallback

    map_png = None
    svg_px = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.goto(f"{base_url.rstrip('/')}/directory", timeout=timeout_ms, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            try:
                page.click('button:has-text("Decline Non-Essential Cookies")', timeout=4000)
            except Exception:
                pass
            page.click("text=Map View", timeout=timeout_ms)
            page.wait_for_timeout(2000)
            try:
                page.click(f'button[data-floor="{level_id}"]', timeout=5000)
                page.wait_for_timeout(1500)
            except Exception:
                pass  # floor switch failed -- fall through, parsing below is a no-op if wrong floor's labels aren't tagged for level_id
            svg_html = page.eval_on_selector("#map_svg", "el => el.outerHTML")
            # real screenshot of the rendered map + its on-screen pixel size,
            # for OCR. Best-effort: a screenshot failure must not lose the
            # anchor data we already have.
            try:
                el = page.query_selector("#map_svg")
                if el:
                    box = el.bounding_box()
                    if box and box["width"] > 0 and box["height"] > 0:
                        map_png = el.screenshot()
                        svg_px = [box["width"], box["height"]]
            except Exception as exc:
                print(f"[anchor_map] map screenshot failed for floor {floor}: {type(exc).__name__}: {exc}", file=sys.stderr)
            browser.close()
    except Exception as exc:
        # The live WebGL map couldn't be driven in this environment -- fall
        # back to the last-known-real anchor coordinates so the map still
        # renders its real backbone instead of coming back empty.
        print(f"[anchor_map] live capture failed for floor {floor} ({type(exc).__name__}: {exc}); "
              f"using cached anchors", file=sys.stderr)
        return fallback

    parsed = parse_map_svg(svg_html, level_id) if svg_html else None
    if parsed is None or not parsed.get("anchors"):
        print(f"[anchor_map] live map parsed no anchors for floor {floor}; using cached anchors", file=sys.stderr)
        return fallback
    parsed["map_png"] = map_png
    parsed["svg_px"] = svg_px
    parsed["live"] = True
    return parsed


def ocr_positions_from_capture(view_box: list[float], svg_px: list[float] | None,
                               ocr_results: list[dict], min_confidence: float = 0.4) -> list[dict]:
    """Converts OCR text boxes (in screenshot-pixel space) into label
    positions in the map's viewBox coordinate space -- the same space the
    DOM anchors use -- so both can be placed on one real coordinate system.

    Each OCR result is {"text", "bbox": [x, y, w, h], "confidence"}; the
    screenshot is a render of the #map_svg element whose on-screen size is
    svg_px, so a pixel maps linearly onto the viewBox. Returns
    [{"text", "x", "y", "confidence"}, ...], dropping empty/low-confidence
    tokens. Returns [] if the geometry needed for conversion is missing."""
    if not svg_px or not view_box or svg_px[0] <= 0 or svg_px[1] <= 0:
        return []
    minx, miny, vb_w, vb_h = view_box
    scale_x = vb_w / svg_px[0]
    scale_y = vb_h / svg_px[1]
    positions = []
    for r in ocr_results:
        text = (r.get("text") or "").strip()
        if len(text) < 2 or r.get("confidence", 0) < min_confidence:
            continue
        x, y, w, h = r["bbox"]
        cx_px = x + w / 2
        cy_px = y + h / 2
        positions.append({
            "text": text,
            "x": minx + cx_px * scale_x,
            "y": miny + cy_px * scale_y,
            "confidence": r.get("confidence", 0.0),
        })
    return positions


def best_label_match(name: str, positions: list[dict], threshold: int = 82) -> dict | None:
    """Fuzzy-matches a directory store name against OCR'd map labels
    (each {"text", "x", "y", "confidence"}), returning the best-matching
    position or None. token_set_ratio handles word-order/subset differences
    ("Nordstrom" vs "Nordstrom Rack", "Macy's" vs "MACYS")."""
    from rapidfuzz import fuzz, process

    if not positions:
        return None
    texts = [p["text"] for p in positions]
    result = process.extractOne(name, texts, scorer=fuzz.token_set_ratio)
    if result and result[1] >= threshold:
        return positions[result[2]]
    return None


def parse_map_svg(svg_html: str, level_id: str) -> dict | None:
    """Pure parsing, split out from the Playwright fetch so it's testable
    against a canned SVG string without a live browser."""
    vb_match = _VIEWBOX_PATTERN.search(svg_html)
    if not vb_match:
        return None
    minx, miny, w, h = (float(v) for v in vb_match.group(1).split())

    default_x_match = _DEFAULT_X_PATTERN.search(svg_html)
    default_y_match = _DEFAULT_Y_PATTERN.search(svg_html)
    default_x = float(default_x_match.group(1)) if default_x_match else minx + w / 2
    default_y = float(default_y_match.group(1)) if default_y_match else miny + h / 2

    level_marker = re.compile(rf"lvl-{re.escape(level_id)}(?!\d)")

    anchors = []
    for attrs, inner in _FOREIGN_OBJECT_PATTERN.findall(svg_html):
        if not level_marker.search(attrs):
            continue

        alt_match = _ALT_PATTERN.search(inner)
        if alt_match:
            name = alt_match.group(1)
        else:
            name = re.sub(r"<br\s*/?>", " ", inner)
            name = re.sub(r"<[^>]+>", "", name)
        name = html.unescape(name).strip()
        if not name:
            continue

        style_match = _STYLE_ATTR_PATTERN.search(attrs)
        style = style_match.group(1) if style_match else ""
        x_match = _X_PATTERN.search(style)
        y_match = _Y_PATTERN.search(style)
        x = float(x_match.group(1)) if x_match else default_x
        y = float(y_match.group(1)) if y_match else default_y
        anchors.append({"name": name, "x": x, "y": y})

    return {"view_box": [minx, miny, w, h], "anchors": anchors}
