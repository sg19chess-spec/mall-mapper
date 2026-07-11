"""Agent 3 -- Validation Agent (human equivalent: Geo-spatial Validation
Analyst). Normalizes, resolves, and cross-checks evidence; computes
per-attribute confidence; flags and classifies conflicts; does light
spatial reasoning over adjacency clues; explains its own decisions.
Point Inside stage mirrored: Quality Framework.

Internally calls agents/tools/{normalizer,rule_engine}.py -- those are
software this agent uses, not separate agents.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from rapidfuzz import fuzz

from app.agents.base import Agent
from app.agents.tools.normalizer import cluster_names
from app.schemas import ConflictReport, ConflictType, get_source_half_life, get_source_prior

EXPECTED_FIELDS = ["floor", "category", "unit"]

# When >= this many *distinct* source types (not just evidence rows) agree
# on a field's value, that's stronger corroboration than the raw noisy-or
# combination alone rewards -- e.g. official directory + floor plan +
# transcript all agreeing is qualitatively different from three web-search
# hits agreeing, even if the math comes out similar.
AGREEMENT_BONUS = 0.05
AGREEMENT_BONUS_MIN_SOURCES = 3

# If the evidence on each side of a conflict is separated by more than this
# many days, treat it as a plausible relocation (TEMPORAL) rather than a
# plain data disagreement -- worth flagging differently to a reviewer.
TEMPORAL_CONFLICT_GAP_DAYS = 180

# Adjacency-based unit inference (spatial reasoning): only trust a
# neighbor's resolved unit enough to borrow from it once its own confidence
# clears this bar, and the inferred unit is a weak starting hypothesis, not
# a strong claim -- hence a small flat confidence rather than the full
# noisy-or treatment.
ADJACENCY_NEIGHBOR_MIN_CONFIDENCE = 0.4
ADJACENCY_NAME_MATCH_THRESHOLD = 80
ADJACENCY_INFERRED_CONFIDENCE = 0.08
ADJACENCY_CORROBORATION_BONUS = 0.08

_UNIT_PATTERN = re.compile(r"^([A-Za-z]*)(\d+)$")

_FIELD_TO_CONFLICT_TYPE = {
    "floor": ConflictType.FLOOR, "unit": ConflictType.UNIT, "category": ConflictType.CATEGORY,
}


def _now():
    return datetime.now(timezone.utc)


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _freshness(evidence: dict, now: datetime) -> float:
    half_life = get_source_half_life(evidence["source_type"])
    age_days = max((now - _parse_dt(evidence["published_date"])).total_seconds() / 86400, 0.0)
    return 0.5 ** (age_days / half_life)


def _completeness(observation: dict) -> float:
    present = sum(1 for f in EXPECTED_FIELDS if observation.get(f) is not None)
    return present / len(EXPECTED_FIELDS) if present else 1 / len(EXPECTED_FIELDS)


def _certainty(evidence: dict) -> float:
    """Linguistic certainty (see Evidence.certainty) -- defaults to 1.0 for
    rows that predate this field or come from sources that don't hedge."""
    return evidence.get("certainty", 1.0) or 1.0


def _parse_unit(unit: str) -> tuple[str, int] | None:
    m = _UNIT_PATTERN.match(unit.strip())
    return (m.group(1), int(m.group(2))) if m else None


def _classify_conflict(field: str, observed: list[tuple], now: datetime) -> ConflictType:
    """FLOOR/UNIT/CATEGORY by default; reclassified as TEMPORAL if the
    disagreeing evidence is separated by a large age gap, suggesting the
    venue actually changed rather than the sources simply disagreeing."""
    base_type = _FIELD_TO_CONFLICT_TYPE.get(field, ConflictType.IDENTITY)
    ages_by_value: dict = {}
    for v, r in observed:
        ages_by_value.setdefault(v, []).append((now - _parse_dt(r["published_date"])).days)
    if len(ages_by_value) < 2:
        return base_type
    newest_per_value = [min(ages) for ages in ages_by_value.values()]
    # if the *closest* evidence for each competing value is still far apart
    # in time, this looks like drift/relocation rather than a data error
    if max(newest_per_value) - min(newest_per_value) >= TEMPORAL_CONFLICT_GAP_DAYS:
        return ConflictType.TEMPORAL
    return base_type


class ValidationAgent(Agent):
    name = "validation"

    def run(self, evidence_rows: list[dict]) -> dict:
        """Returns {
            "entities": {canonical_key: {
                "raw_name": str,
                "fields": {field: {"value", "confidence", "agreement", "inferred"?}},
                "existence_confidence": float,
                "evidence_refs": [{"evidence_id","source_type","confidence_contribution",...}],
                "explanation": [str, ...],
            }},
            "conflicts": [ConflictReport, ...],
        }"""
        now = _now()
        store_evidence = [e for e in evidence_rows if not e["entity_raw"].startswith("__floorplan__")]

        raw_names = list({e["entity_raw"] for e in store_evidence})
        name_map = cluster_names(raw_names)  # raw -> canonical key

        entities: dict[str, dict] = {}
        conflicts: list[ConflictReport] = []

        by_entity: dict[str, list[dict]] = {}
        for e in store_evidence:
            key = name_map[e["entity_raw"]]
            by_entity.setdefault(key, []).append(e)

        for key, rows in by_entity.items():
            raw_name = max((r["entity_raw"] for r in rows), key=len)
            fields: dict[str, dict] = {}
            evidence_refs: list[dict] = []
            existence_contributions: list[float] = []
            explanation: list[str] = []

            for row in rows:
                completeness = _completeness(row["observation"])
                freshness = _freshness(row, now)
                prior = get_source_prior(row["source_type"])
                certainty = _certainty(row)
                base_contribution = prior * freshness * completeness * certainty  # agreement folded in per-field below
                existence_contributions.append(prior * freshness * certainty)
                evidence_refs.append({
                    "evidence_id": row["evidence_id"], "source_type": row["source_type"],
                    "confidence_contribution": round(base_contribution, 4),
                    # carried through purely for audit/debugging -- lets a
                    # reviewer see e.g. "certainty 0.5, hedge_phrase: might"
                    # next to a published attribute without re-querying evidence.
                    "certainty": certainty, "certainty_reason": row.get("certainty_reason"),
                })

            for field in EXPECTED_FIELDS:
                observed = [(r["observation"].get(field), r) for r in rows if r["observation"].get(field) is not None]
                if not observed:
                    continue
                values = [v for v, _ in observed]
                counts: dict = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1
                majority_value = max(counts, key=counts.get)
                agreement = counts[majority_value] / len(values)

                supporting = [r for v, r in observed if v == majority_value]
                distinct_sources = {r["source_type"] for r in supporting}

                # noisy-or combination of independent supporting evidence
                miss_prob = 1.0
                for r in supporting:
                    completeness = _completeness(r["observation"])
                    freshness = _freshness(r, now)
                    prior = get_source_prior(r["source_type"])
                    certainty = _certainty(r)
                    contribution = prior * freshness * completeness * agreement * certainty
                    miss_prob *= (1 - min(contribution, 0.99))
                confidence = 1 - miss_prob

                if len(distinct_sources) >= AGREEMENT_BONUS_MIN_SOURCES:
                    confidence = min(0.99, confidence + AGREEMENT_BONUS)
                    explanation.append(
                        f"{len(distinct_sources)} independent source types "
                        f"({', '.join(sorted(distinct_sources))}) agree on {field} -- agreement bonus applied."
                    )
                elif len(counts) == 1 and len(distinct_sources) >= 2:
                    explanation.append(
                        f"{', '.join(sorted(distinct_sources))} agree on {field}."
                    )

                fields[field] = {
                    "value": majority_value, "confidence": round(confidence, 4), "agreement": round(agreement, 2),
                }

                if len(counts) > 1:
                    conflict_type = _classify_conflict(field, observed, now)
                    conflicts.append(ConflictReport(
                        entity=raw_name, field=field, conflict_type=conflict_type,
                        values=[
                            {"source_type": r["source_type"], "value": v, "evidence_id": r["evidence_id"]}
                            for v, r in observed
                        ],
                    ))
                    # built as a plain list comprehension, not a nested
                    # f-string with the same quote character reused inside
                    # (a Python 3.12+-only relaxation, SyntaxError on 3.11)
                    value_summary = ", ".join(f"{r['source_type']}={v}" for v, r in observed)
                    explanation.append(
                        f"Conflict on {field} ({conflict_type.value}): {value_summary}."
                    )

            existence_miss = 1.0
            for c in existence_contributions:
                existence_miss *= (1 - min(c, 0.99))

            entities[key] = {
                "raw_name": raw_name,
                "fields": fields,
                "existence_confidence": round(1 - existence_miss, 4),
                "evidence_refs": evidence_refs,
                "explanation": explanation,
                "_adjacent_to_mentions": [
                    r["observation"]["adjacent_to"] for r in rows if r["observation"].get("adjacent_to")
                ],
            }

        self._apply_spatial_reasoning(entities)

        for entity in entities.values():
            if not entity["explanation"]:
                entity["explanation"].append("No conflicting evidence found.")
            entity.pop("_adjacent_to_mentions", None)

        return {"entities": entities, "conflicts": conflicts}

    @staticmethod
    def _apply_spatial_reasoning(entities: dict[str, dict]) -> None:
        """Phase-2-flavored spatial reasoning kept intentionally light: if
        Apple's evidence says it's adjacent_to LEGO, and LEGO has a
        confident unit number, borrow LEGO's numbering scheme to propose
        Apple's unit (neighbor_number + 1) as a low-confidence hypothesis --
        or, if Apple already has its own unit evidence, treat a match as
        corroboration and nudge its confidence up. This is exactly the
        "adjacent_to becomes useful" reasoning step Research only collects
        raw evidence for; Validation is where it gets interpreted.
        """
        for key, entity in entities.items():
            mentions = entity.get("_adjacent_to_mentions", [])
            if not mentions:
                continue

            for mention in mentions:
                found = _find_neighbor(entities, key, mention)
                if found is None:
                    continue
                neighbor_key, neighbor = found
                neighbor_unit = neighbor["fields"].get("unit")
                if not neighbor_unit or neighbor_unit["confidence"] < ADJACENCY_NEIGHBOR_MIN_CONFIDENCE:
                    continue
                parsed = _parse_unit(str(neighbor_unit["value"]))
                if parsed is None:
                    continue
                prefix, number = parsed
                candidate = f"{prefix}{number + 1}"

                existing = entity["fields"].get("unit")
                if existing is None:
                    entity["fields"]["unit"] = {
                        "value": candidate, "confidence": ADJACENCY_INFERRED_CONFIDENCE,
                        "agreement": None, "inferred": True,
                    }
                    entity["explanation"].append(
                        f"Adjacency reasoning: near {neighbor['raw_name']} (unit {neighbor_unit['value']}) "
                        f"suggests unit {candidate} (unverified, low confidence)."
                    )
                elif str(existing["value"]) == candidate:
                    existing["confidence"] = round(min(0.99, existing["confidence"] + ADJACENCY_CORROBORATION_BONUS), 4)
                    entity["explanation"].append(
                        f"Adjacency reasoning corroborates unit {candidate}: "
                        f"consistent with proximity to {neighbor['raw_name']} (unit {neighbor_unit['value']})."
                    )


def _find_neighbor(entities: dict[str, dict], self_key: str, mention: str) -> tuple[str, dict] | None:
    best_key, best_entity, best_score = None, None, 0
    for key, candidate in entities.items():
        if key == self_key:
            continue
        score = fuzz.ratio(mention.lower(), candidate["raw_name"].lower())
        if score > best_score:
            best_key, best_entity, best_score = key, candidate, score
    if best_entity is not None and best_score >= ADJACENCY_NAME_MATCH_THRESHOLD:
        return best_key, best_entity
    return None
