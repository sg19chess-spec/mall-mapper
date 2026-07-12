"""Tests for parse_map_svg() against a real snippet of Mall of America's
directory map SVG (captured live, trimmed to a representative subset --
see anchor_map.py's module docstring for how the level-id mapping and the
"only anchors are in the DOM" finding were confirmed). Pure parsing test,
no Playwright/network needed.
"""
from __future__ import annotations

from app.agents.tools.anchor_map import (
    CACHED_ANCHORS,
    FLOOR_TO_LEVEL_ID,
    cached_anchor_positions,
    ocr_positions_from_capture,
    parse_map_svg,
)

# A trimmed but real snippet of #map_svg's outerHTML, captured live from
# mallofamerica.com/directory (Map View tab, floor 1 = lvl-1804).
REAL_MAP_SVG_SNIPPET = """
<svg id="map_svg" preserveAspectRatio="xMidYMid slice" viewBox="640 880 400 300" xmlns="http://www.w3.org/2000/svg">
	<style>
		.directory-map__labels { --width_icon: 50px; }
		.directory-map__labels foreignObject { x: 846px; y: 1043px; }
		.directory-map__labels img { height: var(--width_icon); width: var(--width_icon); }
	</style>
	<g class="directory-map__labels" id="map_labels">
		<foreignObject lvl-1804="" lvl-1805="" lvl-1806="" lvl-1807="" style="x:675px">
			<div>West Parking</div>
		</foreignObject>
		<foreignObject lvl-3347="" style="x:1009px">
			<div>Transit Center</div>
		</foreignObject>
		<foreignObject lvl-1806="" style="x:840px;y:935px">
			<div>North<br>Food<br>Court</div>
		</foreignObject>
		<foreignObject lvl-1804="" lvl-1805="" lvl-1806="" style="">
			<div><img alt="Nickelodeon Universe" src="/themes/custom/moa/images/directory/nicku.svg"></div>
		</foreignObject>
		<foreignObject lvl-1804="" lvl-1805="" lvl-1806="" style="x:745px;y:955px">
			<div><img alt="Nordstrom" src="/themes/custom/moa/images/directory/nordstrom.svg"></div>
		</foreignObject>
		<foreignObject lvl-1804="" lvl-1805="" lvl-1806="" lvl-1807="" style="x:882px;y:911px">
			<div><img alt="JW Marriott" src="/themes/custom/moa/images/directory/marriott.svg"></div>
		</foreignObject>
	</g>
	<g id="map_root"></g>
</svg>
"""


def test_floor_to_level_id_mapping_matches_confirmed_live_values():
    # confirmed live via each floor button's data-floor attribute
    assert FLOOR_TO_LEVEL_ID == {1: "1804", 2: "1805", 3: "1806", 4: "1807"}


def test_parses_view_box():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1804")
    assert result["view_box"] == [640.0, 880.0, 400.0, 300.0]


def test_extracts_anchors_visible_on_the_given_level():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1804")
    names = {a["name"] for a in result["anchors"]}
    # visible on lvl-1804 (floor 1)
    assert "West Parking" in names
    assert "Nickelodeon Universe" in names
    assert "Nordstrom" in names
    assert "JW Marriott" in names
    # NOT visible on floor 1
    assert "Transit Center" not in names  # only lvl-3347
    assert "North Food Court" not in names  # only lvl-1806


def test_extracts_anchors_for_a_different_floor():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1806")
    names = {a["name"] for a in result["anchors"]}
    assert "North Food Court" in names  # only on lvl-1806
    assert "Nordstrom" in names  # also present on lvl-1806
    assert "JW Marriott" in names  # present on all 4 numbered floors


def test_uses_explicit_xy_when_present():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1804")
    nordstrom = next(a for a in result["anchors"] if a["name"] == "Nordstrom")
    assert nordstrom["x"] == 745.0
    assert nordstrom["y"] == 955.0


def test_falls_back_to_css_default_xy_when_missing():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1804")
    nicku = next(a for a in result["anchors"] if a["name"] == "Nickelodeon Universe")
    # style="" -- no explicit x/y -- should fall back to the stylesheet
    # defaults (846px, 1043px), not silently drop the anchor or use (0, 0)
    assert nicku["x"] == 846.0
    assert nicku["y"] == 1043.0


def test_falls_back_to_default_x_only_when_y_present_but_x_missing_style():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1804")
    parking = next(a for a in result["anchors"] if a["name"] == "West Parking")
    assert parking["x"] == 675.0
    assert parking["y"] == 1043.0  # not overridden -- falls back to CSS default


def test_multiline_br_separated_text_labels_are_joined_and_cleaned():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1806")
    food_court = next(a for a in result["anchors"] if "Food Court" in a["name"])
    assert food_court["name"] == "North Food Court"


def test_returns_none_for_unparseable_svg():
    assert parse_map_svg("<svg>not a real map</svg>", "1804") is None


def test_returns_empty_anchors_when_level_has_none():
    result = parse_map_svg(REAL_MAP_SVG_SNIPPET, "1807")
    names = {a["name"] for a in result["anchors"]}
    assert "West Parking" in names  # visible on all 4
    assert "JW Marriott" in names
    assert "Nordstrom" not in names  # not tagged lvl-1807


# ---------------------------------------------------------------------------
# cached anchor fallback (used when the live WebGL map can't be driven)
# ---------------------------------------------------------------------------

def test_cached_anchor_positions_are_real_shaped_and_not_live():
    cap = cached_anchor_positions(1)
    assert cap is not None
    assert cap["live"] is False
    assert cap["map_png"] is None
    assert len(cap["anchors"]) == len(CACHED_ANCHORS[1]["anchors"])
    nord = next(a for a in cap["anchors"] if a["name"] == "Nordstrom")
    assert nord["x"] == 745.0 and nord["y"] == 955.0


def test_cached_anchor_positions_none_for_unknown_floor():
    assert cached_anchor_positions(9) is None


def test_cached_anchor_positions_returns_a_copy():
    # callers mutate the returned dict (map_png etc.) -- must not corrupt the cache
    cap = cached_anchor_positions(1)
    cap["anchors"][0]["x"] = -999
    assert CACHED_ANCHORS[1]["anchors"][0]["x"] != -999


# ---------------------------------------------------------------------------
# OCR screenshot-pixel -> viewBox conversion
# ---------------------------------------------------------------------------

def test_ocr_positions_convert_pixels_to_viewbox_space():
    # a 700x525 screenshot of a map whose viewBox is [640,880,400,300]:
    # a token centred at pixel (350, 262.5) is the middle -> viewBox (840, 1030)
    view_box = [640.0, 880.0, 400.0, 300.0]
    svg_px = [700.0, 525.0]
    ocr = [{"text": "Nordstrom", "bbox": [340, 252, 20, 21], "confidence": 0.9}]
    positions = ocr_positions_from_capture(view_box, svg_px, ocr)
    assert len(positions) == 1
    assert positions[0]["text"] == "Nordstrom"
    assert abs(positions[0]["x"] - 840.0) < 1.0
    assert abs(positions[0]["y"] - 1030.0) < 1.0


def test_ocr_positions_drop_low_confidence_and_tiny_tokens():
    view_box = [640.0, 880.0, 400.0, 300.0]
    svg_px = [700.0, 525.0]
    ocr = [
        {"text": "x", "bbox": [10, 10, 5, 5], "confidence": 0.9},        # too short
        {"text": "Macys", "bbox": [10, 10, 20, 20], "confidence": 0.1},  # too low conf
    ]
    assert ocr_positions_from_capture(view_box, svg_px, ocr) == []


def test_ocr_positions_empty_without_pixel_size():
    view_box = [640.0, 880.0, 400.0, 300.0]
    ocr = [{"text": "Nordstrom", "bbox": [340, 252, 20, 21], "confidence": 0.9}]
    assert ocr_positions_from_capture(view_box, None, ocr) == []
