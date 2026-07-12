"""Unit tests for the real-LLM / real-OCR additions:
- Agent.try_llm_json graceful fallback
- Research Agent's LLM transcript-clue normalization
- anchor_map OCR->viewBox positioning and label matching
- Publication Review's LLM decision with deterministic guardrails
None of these need a live API key or network -- the LLM is stubbed.
"""
from __future__ import annotations

import app.agents.tools.anchor_map as anchor_map
from app.agents.base import Agent, AgentUnavailable
from app.agents.publication_review import PASS_THRESHOLD, PublicationReviewAgent
from app.agents.research import ResearchAgent


# ---------------------------------------------------------------------------
# Agent.try_llm_json
# ---------------------------------------------------------------------------

def test_try_llm_json_returns_none_when_unavailable():
    agent = Agent()
    # no API keys in the test env -> ask_llm_json raises AgentUnavailable
    assert agent.try_llm_json("sys", "prompt") is None


def test_try_llm_json_returns_none_on_error(monkeypatch):
    agent = Agent()

    def boom(*a, **k):
        raise RuntimeError("network blip")

    monkeypatch.setattr(agent, "ask_llm_json", boom)
    assert agent.try_llm_json("sys", "prompt") is None


def test_try_llm_json_passes_through_on_success(monkeypatch):
    agent = Agent()
    monkeypatch.setattr(agent, "ask_llm_json", lambda *a, **k: [{"floor": 2}])
    assert agent.try_llm_json("sys", "prompt") == [{"floor": 2}]


# ---------------------------------------------------------------------------
# ResearchAgent._normalize_llm_transcript_clues
# ---------------------------------------------------------------------------

SEGMENTS = [{"text": "Nike is on level 2, next to Apple.", "start": 122.0}]


def test_normalize_llm_clues_floor_and_adjacency():
    parsed = [
        {"floor": 2, "excerpt": "Nike is on level 2, next to Apple.", "certainty": 1.0},
        {"adjacent_to": "Apple", "excerpt": "Nike is on level 2, next to Apple.", "certainty": 0.8},
    ]
    clues = ResearchAgent._normalize_llm_transcript_clues(parsed, SEGMENTS)
    assert clues[0]["clue"]["floor"] == 2
    assert clues[0]["clue"]["timestamp"] == "00:02:02"  # mapped back to the caption time
    assert clues[0]["excerpt"] == "Nike is on level 2, next to Apple."
    assert clues[1]["clue"]["adjacent_to"] == "Apple"


def test_normalize_llm_clues_drops_items_without_floor_or_adjacency():
    parsed = [{"excerpt": "Nike is great", "certainty": 1.0}]
    assert ResearchAgent._normalize_llm_transcript_clues(parsed, SEGMENTS) == []


def test_normalize_llm_clues_rejects_non_list():
    assert ResearchAgent._normalize_llm_transcript_clues({"floor": 2}, SEGMENTS) is None


def test_normalize_llm_clues_clamps_bad_certainty():
    parsed = [{"floor": 3, "excerpt": "x", "certainty": "not-a-number"}]
    clues = ResearchAgent._normalize_llm_transcript_clues(parsed, SEGMENTS)
    assert clues[0]["certainty"] == 1.0


# ---------------------------------------------------------------------------
# anchor_map: OCR -> viewBox positioning + label matching
# ---------------------------------------------------------------------------

def test_ocr_positions_scale_pixels_into_viewbox():
    view_box = [0, 0, 2000, 1000]
    svg_px = [1000, 500]  # screenshot is half the viewBox scale in both axes
    ocr_results = [{"text": "Nordstrom", "bbox": [100, 50, 40, 20], "confidence": 0.9}]
    positions = anchor_map.ocr_positions_from_capture(view_box, svg_px, ocr_results)
    assert len(positions) == 1
    # bbox center (120, 60) px -> *2 scale -> (240, 120) viewBox
    assert positions[0]["x"] == 240.0
    assert positions[0]["y"] == 120.0
    assert positions[0]["text"] == "Nordstrom"


def test_ocr_positions_drop_low_confidence_and_short_tokens():
    view_box = [0, 0, 100, 100]
    svg_px = [100, 100]
    ocr_results = [
        {"text": "a", "bbox": [1, 1, 1, 1], "confidence": 0.9},        # too short
        {"text": "Zara", "bbox": [1, 1, 1, 1], "confidence": 0.1},     # too low-confidence
        {"text": "Macys", "bbox": [10, 10, 4, 4], "confidence": 0.8},  # kept
    ]
    positions = anchor_map.ocr_positions_from_capture(view_box, svg_px, ocr_results)
    assert [p["text"] for p in positions] == ["Macys"]


def test_ocr_positions_empty_without_svg_px():
    assert anchor_map.ocr_positions_from_capture([0, 0, 100, 100], None, [{"text": "x", "bbox": [0, 0, 1, 1], "confidence": 1}]) == []


def test_best_label_match_fuzzy():
    positions = [{"text": "Nordstrom", "x": 1, "y": 2, "confidence": 0.9}]
    assert anchor_map.best_label_match("Nordstrom", positions)["x"] == 1
    assert anchor_map.best_label_match("Totally Different Store", positions) is None
    assert anchor_map.best_label_match("Nordstrom", []) is None


# ---------------------------------------------------------------------------
# PublicationReviewAgent._decide guardrails
# ---------------------------------------------------------------------------

class _FakeStore:
    pass


def _feature():
    return {"properties": {"name": "Test Store"}, "floor": 2, "confidence_by_attribute": {"name": 0.9}}


def test_decide_deterministic_when_no_llm(monkeypatch):
    agent = PublicationReviewAgent(_FakeStore())
    monkeypatch.setattr(agent, "llm_available", lambda: False)
    rec, reason, note = agent._decide(
        can_pass=True, iteration=1, max_iterations=4, reasons=[],
        feature=_feature(), min_confidence=0.9, conflicts=[], violations=[],
    )
    assert rec == "pass"


def test_decide_llm_pass_is_clamped_when_not_eligible(monkeypatch):
    agent = PublicationReviewAgent(_FakeStore())
    monkeypatch.setattr(agent, "llm_available", lambda: True)
    monkeypatch.setattr(agent, "try_llm_json", lambda *a, **k: {"recommendation": "pass", "reason": "looks fine"})
    # can_pass False, iterations remain -> LLM "pass" must be clamped to retry
    rec, reason, note = agent._decide(
        can_pass=False, iteration=1, max_iterations=4, reasons=["low confidence on floor"],
        feature=_feature(), min_confidence=0.3, conflicts=[], violations=[],
    )
    assert rec == "retry"


def test_decide_llm_pass_clamped_to_human_review_when_iterations_spent(monkeypatch):
    agent = PublicationReviewAgent(_FakeStore())
    monkeypatch.setattr(agent, "llm_available", lambda: True)
    monkeypatch.setattr(agent, "try_llm_json", lambda *a, **k: {"recommendation": "pass", "reason": "ship it"})
    rec, reason, note = agent._decide(
        can_pass=False, iteration=4, max_iterations=4, reasons=["low confidence"],
        feature=_feature(), min_confidence=0.3, conflicts=[], violations=[],
    )
    assert rec == "human_review"


def test_decide_falls_back_to_deterministic_on_llm_error(monkeypatch):
    agent = PublicationReviewAgent(_FakeStore())
    monkeypatch.setattr(agent, "llm_available", lambda: True)
    monkeypatch.setattr(agent, "try_llm_json", lambda *a, **k: None)  # LLM failed/unparseable
    rec, reason, note = agent._decide(
        can_pass=True, iteration=1, max_iterations=4, reasons=[],
        feature=_feature(), min_confidence=0.9, conflicts=[], violations=[],
    )
    assert rec == "pass"


def test_decide_llm_retry_clamped_to_human_review_when_iterations_spent(monkeypatch):
    agent = PublicationReviewAgent(_FakeStore())
    monkeypatch.setattr(agent, "llm_available", lambda: True)
    monkeypatch.setattr(agent, "try_llm_json", lambda *a, **k: {"recommendation": "retry", "reason": "need more"})
    rec, reason, note = agent._decide(
        can_pass=False, iteration=4, max_iterations=4, reasons=["x"],
        feature=_feature(), min_confidence=0.3, conflicts=[], violations=[],
    )
    assert rec == "human_review"
