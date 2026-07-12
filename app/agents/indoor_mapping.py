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
from app.agents.tools import anchor_map
from app.agents.tools import geometry as geom_tools
from app.agents.tools.normalizer import normalize
from app.schemas import FeatureType, GeometryFeature, GeometryType, IndoorFeature

CHANGE_TOLERANCE_FIELDS = ("unit",)

# Every published position is real -- no synthetic fallback. A store gets
# geometry only from (a) the map's own SVG DOM anchor list, or (b) a label
# OCR'd off a screenshot of the rendered floor map. Anything else gets no
# geometry at all (it is not placed on the map) rather than a fabricated one.
ANCHOR_GEOMETRY_CONFIDENCE = 0.9   # position read straight from the map's SVG DOM
ANCHOR_NAME_MATCH_THRESHOLD = 80
# OCR'd label positions are real but noisier than a DOM coordinate (rendered
# text recognition, screenshot->viewBox scaling), so scored below anchors and
# scaled by the OCR engine's own per-token confidence.
OCR_GEOMETRY_BASE_CONFIDENCE = 0.55
OCR_GEOMETRY_MAX_CONFIDENCE = 0.8


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

    def build_anchor_features(self, mall: str, floor: int, floorplan_evidence: dict | None) -> list[dict]:
        """Real anchor landmarks (Nordstrom, Nickelodeon Universe, parking
        rotundas, ...) as their own reference features, so the map always has
        a real backbone even when few tenant stores can be individually
        placed. These are read straight from the mall's map (anchor_positions
        in the floor-plan evidence), carry a real Point position, and are
        published directly -- they aren't tenant claims needing multi-source
        corroboration, they're the map's own labeled reference points."""
        if not floorplan_evidence:
            return []
        anchor_data = floorplan_evidence["observation"].get("anchor_positions")
        if not anchor_data or not anchor_data.get("anchors"):
            return []
        view_box = anchor_data["view_box"]
        is_live = bool(anchor_data.get("live"))
        features: list[dict] = []
        for a in anchor_data["anchors"]:
            fid = f"{mall}:{floor}:anchor:{a['name']}".replace(" ", "_")
            features.append({
                "feature_id": fid,
                "feature_type": FeatureType.ANCHOR.value,
                "geometry": geom_tools.anchor_point(a["x"], a["y"]),
                "properties": {
                    "name": a["name"], "geometry_source": "real_anchor_reference",
                    "anchor_view_box": view_box, "live_capture": is_live,
                },
                "confidence_by_attribute": {"name": 1.0, "geometry": ANCHOR_GEOMETRY_CONFIDENCE},
                "evidence": [],
                "version": 1,
                "valid_from": _now().isoformat(),
                "valid_until": None,
                "change_reason": None,
                "floor": floor,
            })
        return features

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
        anchor_data = None
        ocr_positions: list[dict] = []
        if floorplan_evidence:
            obs = floorplan_evidence["observation"]
            anchor_data = obs.get("anchor_positions")
            ocr_positions = obs.get("ocr_positions") or []

        anchors = anchor_data["anchors"] if anchor_data else []
        view_box = anchor_data["view_box"] if anchor_data else None
        features: list[dict] = []

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

            # Real positions only. Anchor DOM first (most reliable), then a
            # label OCR'd off the rendered map; otherwise the store is left
            # unplaced (geometry=None) rather than given a fabricated spot.
            matched_anchor = self._match_anchor(entity["raw_name"], anchors)
            ocr_match = None if matched_anchor else anchor_map.best_label_match(entity["raw_name"], ocr_positions)
            if matched_anchor:
                geometry = geom_tools.anchor_point(matched_anchor["x"], matched_anchor["y"])
                confidence_by_attribute["geometry"] = ANCHOR_GEOMETRY_CONFIDENCE
                props["geometry_source"] = "real_anchor"
                props["anchor_view_box"] = view_box
                props["matched_anchor_name"] = matched_anchor["name"]
            elif ocr_match:
                geometry = geom_tools.anchor_point(ocr_match["x"], ocr_match["y"])
                confidence_by_attribute["geometry"] = round(
                    min(OCR_GEOMETRY_MAX_CONFIDENCE,
                        OCR_GEOMETRY_BASE_CONFIDENCE * (0.5 + 0.5 * ocr_match.get("confidence", 0.0)) * 2),
                    2,
                )
                props["geometry_source"] = "ocr_label"
                props["anchor_view_box"] = view_box
                props["ocr_text"] = ocr_match["text"]
            else:
                # No real position available -- leave geometry unknown (not a
                # failing 0.0). Identity can still publish; the store is just
                # not drawn on the map. Omitting the geometry key entirely
                # means Publication Review's geometry gate treats it as
                # "no geometry to check" rather than "geometry failed".
                geometry = None
                props["geometry_source"] = "unplaced"

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

        return features
