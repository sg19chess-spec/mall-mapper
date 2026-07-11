"""Accuracy/behavior tests for the Validation Agent's five capabilities:
agreement scoring, conflict classification, spatial reasoning over
adjacency clues, evidence weighting (certainty), and explainability.

These test ValidationAgent.run() directly against hand-built evidence rows
-- no store/network/API dependency, since ValidationAgent is a pure
function of the evidence it's given.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agents.validation import (
    ADJACENCY_CORROBORATION_BONUS,
    ADJACENCY_INFERRED_CONFIDENCE,
    AGREEMENT_BONUS,
    ValidationAgent,
)
from app.schemas import ConflictType, Evidence, SourceType


def make_evidence(source_type: str, entity_raw: str, observation: dict, *,
                   days_ago: float = 0, certainty: float = 1.0, certainty_reason: str | None = None) -> dict:
    published = datetime.now(timezone.utc) - timedelta(days=days_ago)
    ev = Evidence(
        source_type=SourceType(source_type),
        entity_raw=entity_raw,
        observation=observation,
        published_date=published,
        certainty=certainty,
        certainty_reason=certainty_reason,
    )
    return ev.model_dump(mode="json")


def only_entity(result: dict) -> dict:
    assert len(result["entities"]) == 1, result["entities"]
    return next(iter(result["entities"].values()))


def find_entity(result: dict, raw_name_substring: str) -> dict:
    for entity in result["entities"].values():
        if raw_name_substring.lower() in entity["raw_name"].lower():
            return entity
    raise AssertionError(f"no entity matching {raw_name_substring!r} in {list(result['entities'])}")


# ---------------------------------------------------------------------------
# 1. Agreement scoring: bonus for >=3 distinct source types agreeing
# ---------------------------------------------------------------------------

def test_agreement_bonus_applied_with_three_distinct_sources():
    evidence = [
        make_evidence("web", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 2}),
        make_evidence("satellite", "Test Store", {"floor": 2}),
    ]
    entity = only_entity(ValidationAgent().run(evidence))
    assert any("agreement bonus applied" in line for line in entity["explanation"])


def test_no_agreement_bonus_with_two_distinct_sources():
    evidence = [
        make_evidence("web", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 2}),
    ]
    entity = only_entity(ValidationAgent().run(evidence))
    assert not any("agreement bonus applied" in line for line in entity["explanation"])


def test_three_distinct_sources_beats_two_on_confidence():
    two_source_evidence = [
        make_evidence("web", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 2}),
    ]
    three_source_evidence = two_source_evidence + [
        make_evidence("satellite", "Test Store", {"floor": 2}),
    ]
    conf_2 = only_entity(ValidationAgent().run(two_source_evidence))["fields"]["floor"]["confidence"]
    conf_3 = only_entity(ValidationAgent().run(three_source_evidence))["fields"]["floor"]["confidence"]
    # adding a third distinct, agreeing source should both raise the raw
    # noisy-or combination AND trigger the flat AGREEMENT_BONUS on top
    assert conf_3 > conf_2


def test_agreement_bonus_is_the_documented_constant():
    # isolate the bonus's exact effect: two identical low-prior sources
    # (web + social) sit well below the 0.99 cap, so adding a third
    # (satellite) should raise confidence by roughly noisy-or's marginal
    # contribution *plus* AGREEMENT_BONUS -- assert the bonus alone is
    # present by checking the delta exceeds what a same-prior single new
    # source would add without crossing the 3-source threshold.
    two = [
        make_evidence("web", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 2}),
    ]
    three = two + [make_evidence("satellite", "Test Store", {"floor": 2})]
    conf_2 = only_entity(ValidationAgent().run(two))["fields"]["floor"]["confidence"]
    conf_3 = only_entity(ValidationAgent().run(three))["fields"]["floor"]["confidence"]
    # satellite's prior (0.04) alone would add very little; the jump should
    # be noticeably larger than 0.04 because AGREEMENT_BONUS (0.05) is
    # also applied once the 3-distinct-source threshold is crossed.
    assert conf_3 - conf_2 > AGREEMENT_BONUS


# ---------------------------------------------------------------------------
# 2. Conflict classification
# ---------------------------------------------------------------------------

def test_conflict_classified_as_floor_when_recent():
    evidence = [
        make_evidence("official_directory", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 5}),
    ]
    conflicts = ValidationAgent().run(evidence)["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.FLOOR


def test_conflict_classified_as_unit():
    evidence = [
        make_evidence("official_directory", "Test Store", {"unit": "S245"}),
        make_evidence("web", "Test Store", {"unit": "S999"}),
    ]
    conflicts = ValidationAgent().run(evidence)["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.UNIT


def test_conflict_classified_as_category():
    evidence = [
        make_evidence("official_directory", "Test Store", {"category": "Apparel"}),
        make_evidence("web", "Test Store", {"category": "Electronics"}),
    ]
    conflicts = ValidationAgent().run(evidence)["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.CATEGORY


def test_conflict_reclassified_as_temporal_when_evidence_ages_diverge():
    evidence = [
        make_evidence("official_directory", "Test Store", {"floor": 2}, days_ago=0),
        make_evidence("web", "Test Store", {"floor": 1}, days_ago=400),
    ]
    conflicts = ValidationAgent().run(evidence)["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.TEMPORAL


def test_no_conflict_when_sources_agree():
    evidence = [
        make_evidence("official_directory", "Test Store", {"floor": 2}),
        make_evidence("web", "Test Store", {"floor": 2}),
    ]
    assert ValidationAgent().run(evidence)["conflicts"] == []


# ---------------------------------------------------------------------------
# 3. Spatial reasoning over adjacent_to
# ---------------------------------------------------------------------------

def test_adjacency_infers_unit_for_neighborless_entity():
    evidence = [
        # LEGO Store has a confident unit from a strong-prior source
        make_evidence("official_directory", "LEGO Store", {"floor": 1, "category": "Toys", "unit": "S244"}),
        # Apple has floor evidence and an adjacency clue, but no unit of its own
        make_evidence("official_directory", "Apple", {"floor": 2}),
        make_evidence("youtube_transcript", "Apple", {"adjacent_to": "LEGO Store"}),
    ]
    result = ValidationAgent().run(evidence)
    apple = find_entity(result, "Apple")

    assert apple["fields"]["unit"]["value"] == "S245"
    assert apple["fields"]["unit"]["inferred"] is True
    assert apple["fields"]["unit"]["confidence"] == ADJACENCY_INFERRED_CONFIDENCE
    assert any("Adjacency reasoning" in line for line in apple["explanation"])


def test_adjacency_corroborates_matching_existing_unit():
    base_evidence = [
        make_evidence("official_directory", "LEGO Store", {"floor": 1, "category": "Toys", "unit": "S244"}),
        make_evidence("official_directory", "Apple", {"floor": 2, "unit": "S245"}),
    ]
    without_adjacency = only_entity_by_name(ValidationAgent().run(base_evidence), "Apple")

    with_adjacency = find_entity(
        ValidationAgent().run(base_evidence + [
            make_evidence("youtube_transcript", "Apple", {"adjacent_to": "LEGO Store"}),
        ]),
        "Apple",
    )

    assert with_adjacency["fields"]["unit"]["confidence"] == pytest.approx(
        without_adjacency["fields"]["unit"]["confidence"] + ADJACENCY_CORROBORATION_BONUS, abs=1e-6,
    )
    assert any("Adjacency reasoning corroborates" in line for line in with_adjacency["explanation"])


def test_adjacency_skips_low_confidence_neighbor():
    evidence = [
        # satellite alone (prior 0.04) never clears ADJACENCY_NEIGHBOR_MIN_CONFIDENCE (0.4)
        make_evidence("satellite", "LEGO Store", {"floor": 1, "unit": "S244"}),
        make_evidence("official_directory", "Apple", {"floor": 2}),
        make_evidence("youtube_transcript", "Apple", {"adjacent_to": "LEGO Store"}),
    ]
    apple = find_entity(ValidationAgent().run(evidence), "Apple")
    assert "unit" not in apple["fields"]


def test_adjacency_ignores_unmatched_neighbor_name():
    evidence = [
        make_evidence("official_directory", "Apple", {"floor": 2}),
        make_evidence("youtube_transcript", "Apple", {"adjacent_to": "Some Unrelated Kiosk"}),
    ]
    # should not raise, and should not fabricate a unit out of nothing
    apple = find_entity(ValidationAgent().run(evidence), "Apple")
    assert "unit" not in apple["fields"]


def only_entity_by_name(result: dict, raw_name_substring: str) -> dict:
    return find_entity(result, raw_name_substring)


# ---------------------------------------------------------------------------
# 4. Evidence weighting via certainty
# ---------------------------------------------------------------------------

def test_hedged_certainty_lowers_confidence_proportionally():
    stated = only_entity(ValidationAgent().run([
        make_evidence("web", "Test Store", {"floor": 2}, certainty=1.0, certainty_reason="stated_as_fact"),
    ]))
    hedged = only_entity(ValidationAgent().run([
        make_evidence("web", "Test Store", {"floor": 2}, certainty=0.5, certainty_reason="hedge_phrase: might"),
    ]))

    assert hedged["fields"]["floor"]["confidence"] < stated["fields"]["floor"]["confidence"]
    # single-source noisy-or: confidence == contribution, which is linear
    # in certainty, so halving certainty should roughly halve confidence.
    assert hedged["fields"]["floor"]["confidence"] == pytest.approx(
        stated["fields"]["floor"]["confidence"] * 0.5, abs=1e-3,
    )


def test_certainty_and_reason_carried_into_evidence_refs():
    entity = only_entity(ValidationAgent().run([
        make_evidence("web", "Test Store", {"floor": 2}, certainty=0.35, certainty_reason="hedge_phrase: maybe"),
    ]))
    ref = entity["evidence_refs"][0]
    assert ref["certainty"] == 0.35
    assert ref["certainty_reason"] == "hedge_phrase: maybe"


# ---------------------------------------------------------------------------
# 5. Explainability
# ---------------------------------------------------------------------------

def test_explanation_defaults_when_nothing_notable():
    entity = only_entity(ValidationAgent().run([
        make_evidence("web", "Test Store", {"floor": 2}),
    ]))
    assert entity["explanation"] == ["No conflicting evidence found."]


def test_explanation_mentions_two_source_agreement_without_bonus_language():
    entity = only_entity(ValidationAgent().run([
        make_evidence("official_directory", "Test Store", {"floor": 2}),
        make_evidence("web", "Test Store", {"floor": 2}),
    ]))
    assert any("agree on floor" in line and "bonus" not in line for line in entity["explanation"])


def test_explanation_describes_conflict_with_type_and_values():
    entity = only_entity(ValidationAgent().run([
        make_evidence("official_directory", "Test Store", {"floor": 2}),
        make_evidence("social", "Test Store", {"floor": 5}),
    ]))
    conflict_lines = [line for line in entity["explanation"] if "Conflict on floor" in line]
    assert len(conflict_lines) == 1
    assert "floor" in conflict_lines[0]
    assert "official_directory=2" in conflict_lines[0]
    assert "social=5" in conflict_lines[0]
