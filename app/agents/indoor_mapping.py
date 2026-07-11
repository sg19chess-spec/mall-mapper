"""Agent 4 -- Indoor Mapping Agent (human equivalent: Indoor Mapping
Specialist). Transforms validated evidence into standardized indoor GIS
features ready for publication: geometry, IndoorFeature records, GeoJSON,
indoor topology graph updates.

Internally calls agents/tools/{geometry,indoor_graph}.py -- those are
software this agent uses, not separate agents. Feature versioning and
change detection (moved/closed/renamed) also live here rather than in a
dedicated file, per the project's trimmed folder structure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from rapidfuzz import fuzz, process

from app.agents.base import Agent
from app.agents.tools import geometry as geom_tools
from app.agents.tools.normalizer import normalize
from app.schemas import FeatureType, GeometryFeature, GeometryType, IndoorFeature

CHANGE_TOLERANCE_FIELDS = ("unit",)

# Real anchor positions are extracted from the mall's own map (see
# agents/tools/anchor_map.py) -- a real coordinate is worth a lot more
# confidence than the synthetic placeholder grid, hence the much higher
# base score here vs. geom_tools.geometry_confidence()'s synthetic-mode 0.35.
ANCHOR_GEOMETRY_CONFIDENCE = 0.9
ANCHOR_NAME_MATCH_THRESHOLD = 80


def _now():
    return datetime.now(timezone.utc)


class IndoorMappingAgent(Agent):
    name = "indoor_mapping"

    def __init__(self, store) -> None:
        super().__init__()
        self.store = store

    def _feature_id(self, mall: str, floor: int, canonical_key: str) -> str:
        return f"{mall}:{floor}:store:{canonical_key}".replace(" ", "_")

    def _previous_version(self, feature_id: str) -> dict | None:
        history = self.store.get_feature_history(feature_id)
        open_versions = [h for h in history if h.get("valid_until") is None]
        return open_versions[-1] if open_versions else None

    def _detect_change(self, previous: dict | None, new_properties: dict) -> str | None:
        if previous is None:
            return None
        prev_props = previous["properties"]
        if isinstance(prev_props, str):
            import json

            prev_props = json.loads(prev_props)
        for field in CHANGE_TOLERANCE_FIELDS:
            if prev_props.get(field) and new_properties.get(field) and prev_props[field] != new_properties[field]:
                return "moved"
        if prev_props.get("name") != new_properties.get("name"):
            return "renamed"
        return None

    def build_corridor_feature(self, mall: str, floor: int, corridor: dict) -> dict:
        geometry = geom_tools.corridor_linestring(corridor)
        return {
            "feature_id": f"{mall}:{floor}:corridor".replace(" ", "_"),
            "feature_type": FeatureType.CORRIDOR.value,
            "geometry": geometry,
            "properties": {"name": f"Floor {floor} corridor"},
            "confidence_by_attribute": {"geometry": 0.5},
            "evidence": [],
            "version": 1,
            "valid_from": _now().isoformat(),
            "valid_until": None,
            "change_reason": None,
            "floor": floor,
        }

    @staticmethod
    def _match_anchor(raw_name: str, anchors: list[dict]) -> dict | None:
        if not anchors:
            return None
        names = [a["name"] for a in anchors]
        result = process.extractOne(raw_name, names, scorer=fuzz.ratio)
        if result and result[1] >= ANCHOR_NAME_MATCH_THRESHOLD:
            return anchors[result[2]]
        return None

    def run(self, mall: str, floor: int, validation_result: dict, floorplan_evidence: dict | None) -> list[dict]:
        entities = validation_result["entities"]
        grid = None
        ocr_confidence = None
        has_official_floorplan = False
        anchor_data = None
        if floorplan_evidence:
            obs = floorplan_evidence["observation"]
            grid = obs.get("synthetic_grid")
            ocr_results = obs.get("ocr_results") or []
            has_official_floorplan = bool(ocr_results)
            if ocr_results:
                ocr_confidence = sum(r["confidence"] for r in ocr_results) / len(ocr_results)
            anchor_data = obs.get("anchor_positions")

        anchors = anchor_data["anchors"] if anchor_data else []
        view_box = anchor_data["view_box"] if anchor_data else None
        slots = grid["store_slots"] if grid else []
        features: list[dict] = []
        synthetic_slot_index = 0  # separate counter -- anchor-matched entities don't consume a synthetic slot

        for key, entity in entities.items():
            feature_id = self._feature_id(mall, floor, key)
            props = {
                "name": entity["raw_name"],
                "category": entity["fields"].get("category", {}).get("value"),
                "unit": entity["fields"].get("unit", {}).get("value"),
            }

            confidence_by_attribute = {"name": entity["existence_confidence"]}
            for field, data in entity["fields"].items():
                confidence_by_attribute[field] = data["confidence"]

            matched_anchor = self._match_anchor(entity["raw_name"], anchors)
            if matched_anchor:
                geometry = geom_tools.anchor_point(matched_anchor["x"], matched_anchor["y"])
                confidence_by_attribute["geometry"] = ANCHOR_GEOMETRY_CONFIDENCE
                props["geometry_source"] = "real_anchor"
                props["anchor_view_box"] = view_box
                props["matched_anchor_name"] = matched_anchor["name"]
            elif synthetic_slot_index < len(slots):
                geometry = geom_tools.store_polygon_from_slot(slots[synthetic_slot_index])
                synthetic_slot_index += 1
                confidence_by_attribute["geometry"] = geom_tools.geometry_confidence(has_official_floorplan, ocr_confidence)
                props["geometry_source"] = "synthetic_placeholder"
            else:
                geometry = None
                confidence_by_attribute["geometry"] = geom_tools.geometry_confidence(has_official_floorplan, ocr_confidence)
                props["geometry_source"] = "synthetic_placeholder"

            previous = self._previous_version(feature_id)
            change_reason = self._detect_change(previous, props)
            version = (previous["version"] + 1) if (previous and change_reason) else (previous["version"] if previous else 1)

            features.append({
                "feature_id": feature_id,
                "feature_type": FeatureType.STORE.value,
                "geometry": geometry,
                "properties": props,
                "confidence_by_attribute": confidence_by_attribute,
                "evidence": entity["evidence_refs"],
                "version": version,
                "valid_from": _now().isoformat(),
                "valid_until": None,
                "change_reason": change_reason,
                "floor": floor,
                "_canonical_key": key,
                "_conflicts": [c for c in validation_result["conflicts"] if c.entity == entity["raw_name"]],
                "_previous_version": previous,
                "_explanation": entity["explanation"],
            })

        if grid:
            features.append(self.build_corridor_feature(mall, floor, grid["corridor"]))

        return features
