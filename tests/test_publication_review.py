"""Tests that Publication Review correctly consumes the Validation Agent's
conflict_type classification and explanation bullets -- i.e. that the two
agents' outputs compose into one coherent, readable ReviewReport rather
than silently dropping information at the agent boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.agents.publication_review import GEOMETRY_MIN_CONFIDENCE, PASS_THRESHOLD, PublicationReviewAgent
from app.schemas import ConflictReport, ConflictType


class FakeStore:
    """No-op store double -- PublicationReviewAgent only needs these four
    methods, and none of these tests care about persistence."""

    def __init__(self) -> None:
        self.published = []
        self.review_items = []
        self.review_reports = []

    def insert_review_report(self, report: dict) -> None:
        self.review_reports.append(report)

    def upsert_review_item(self, item: dict) -> None:
        self.review_items.append(item)

    def publish_feature(self, feature: dict, mall: str, floor: int) -> None:
        self.published.append(feature["feature_id"])

    def log_change(self, *args, **kwargs) -> None:
        pass


def make_feature(feature_id="Mall_of_America:2:store:test", confidence=None, conflicts=None,
                  explanation=None, geometry=None) -> dict:
    confidence = confidence or {"name": 0.9, "floor": 0.9, "category": 0.9, "unit": 0.9, "geometry": 0.5}
    return {
        "feature_id": feature_id,
        "feature_type": "store",
        "geometry": geometry,
        "properties": {"name": "Test Store", "category": "Apparel", "unit": "S245"},
        "confidence_by_attribute": confidence,
        "evidence": [],
        "version": 1,
        "valid_from": datetime.now(timezone.utc).isoformat(),
        "valid_until": None,
        "change_reason": None,
        "floor": 2,
        "_canonical_key": "test",
        "_conflicts": conflicts or [],
        "_previous_version": None,
        "_explanation": explanation or [],
    }


def test_explanation_combines_validation_and_publication_bullets_on_pass():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    feature = make_feature(explanation=["official_directory, web agree on floor."])

    report, follow_ups = agent.review("Mall of America", 2, feature, [feature], iteration=1, max_iterations=4)

    assert report.recommendation == "pass"
    assert follow_ups == []
    assert "official_directory, web agree on floor." in report.explanation
    assert any("cleared the publish threshold" in line for line in report.explanation)
    assert store.published == [feature["feature_id"]]


def test_explanation_and_reason_mention_conflict_type_on_retry():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    conflict = ConflictReport(
        entity="Test Store", field="floor", conflict_type=ConflictType.FLOOR,
        values=[{"source_type": "official_directory", "value": 2, "evidence_id": "a"},
                {"source_type": "social", "value": 5, "evidence_id": "b"}],
    )
    feature = make_feature(
        confidence={"name": 0.9, "floor": 0.9, "category": 0.9, "unit": 0.9, "geometry": 0.5},
        conflicts=[conflict],
        explanation=["Conflict on floor (floor): official_directory=2, social=5."],
    )

    report, follow_ups = agent.review("Mall of America", 2, feature, [feature], iteration=1, max_iterations=4)

    assert report.recommendation == "retry"
    assert "floor" in report.reason
    assert "Conflict on floor (floor):" in report.explanation[0]
    assert any("Requesting more evidence" in line for line in report.explanation)
    assert len(follow_ups) == 1
    assert follow_ups[0].task_type.value == "verify_floor"
    assert store.published == []


def test_explanation_marks_temporal_conflict_distinctly_on_escalation():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    conflict = ConflictReport(
        entity="Test Store", field="floor", conflict_type=ConflictType.TEMPORAL,
        values=[{"source_type": "official_directory", "value": 2, "evidence_id": "a"},
                {"source_type": "web", "value": 1, "evidence_id": "b"}],
    )
    feature = make_feature(conflicts=[conflict])

    report, _ = agent.review("Mall of America", 2, feature, [feature], iteration=4, max_iterations=4)

    assert report.recommendation == "human_review"
    assert "temporal" in report.reason
    assert any("Escalated to human review" in line for line in report.explanation)
    assert store.review_items[0]["feature_id"] == feature["feature_id"]


def test_low_confidence_field_below_pass_threshold_triggers_retry():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    feature = make_feature(confidence={
        "name": 0.9, "floor": 0.9, "category": PASS_THRESHOLD - 0.1, "unit": 0.9, "geometry": 0.5,
    })

    report, follow_ups = agent.review("Mall of America", 2, feature, [feature], iteration=1, max_iterations=4)

    assert report.recommendation == "retry"
    assert "category" in report.reason
    assert any(fu.task_type.value == "verify_category" for fu in follow_ups)


# ---------------------------------------------------------------------------
# Geometry rule checks (must_not_overlap / must_intersect / centroid_inside),
# exercised through PublicationReviewAgent.review() rather than rule_engine
# directly -- i.e. does the agent correctly gate publication, request the
# right follow-up task, and explain itself when geometry fails a rule.
# ---------------------------------------------------------------------------

def polygon(x_min, y_min, x_max, y_max) -> dict:
    return {"type": "Polygon", "coordinates": [[
        (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min),
    ]]}


def line(x_start, x_end, y) -> dict:
    return {"type": "LineString", "coordinates": [(x_start, y), (x_end, y)]}


def other_store(geometry) -> dict:
    return {"feature_type": "store", "geometry": geometry}


def corridor_feature(geometry) -> dict:
    return {"feature_type": "corridor", "geometry": geometry}


HIGH_IDENTITY_CONFIDENCE = {"name": 0.9, "floor": 0.9, "category": 0.9, "unit": 0.9}


def test_overlapping_geometry_triggers_retry_with_verify_geometry_task():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 0, 60, 60)
    feature = make_feature(
        geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9},
    )
    neighbor = other_store(polygon(30, 30, 90, 90))  # overlaps feature's polygon
    corridor = corridor_feature(line(0, 200, 60))  # touches feature's bottom edge, so only overlap should fail

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, neighbor, corridor], iteration=1, max_iterations=4,
    )

    assert report.recommendation == "retry"
    assert "geometry rule violations" in report.reason
    assert "overlaps another store polygon" in report.reason
    assert any(fu.task_type.value == "verify_geometry" for fu in follow_ups)
    assert store.published == []


def test_geometry_not_touching_corridor_triggers_retry():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 0, 60, 60)
    feature = make_feature(
        geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9},
    )
    far_corridor = corridor_feature(line(0, 500, 9000))  # nowhere near the store

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, far_corridor], iteration=1, max_iterations=4,
    )

    assert report.recommendation == "retry"
    assert "does not intersect any corridor" in report.reason
    assert any(fu.task_type.value == "verify_geometry" for fu in follow_ups)


def test_valid_geometry_with_no_violations_allows_pass():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 140, 60, 200)  # bottom edge touches the corridor at y=200
    feature = make_feature(
        geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9},
    )
    corridor = corridor_feature(line(0, 500, 200))
    other = other_store(polygon(300, 140, 360, 200))  # elsewhere, no overlap

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, corridor, other], iteration=1, max_iterations=4,
    )

    assert report.recommendation == "pass"
    assert follow_ups == []
    assert store.published == [feature["feature_id"]]


def test_geometry_confidence_below_minimum_triggers_retry_even_without_rule_violation():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 140, 60, 200)
    # geometry has no rule violations (touches the corridor, no overlap) but
    # its own confidence is below GEOMETRY_MIN_CONFIDENCE -- e.g. no OCR/
    # official floor plan corroborated the shape yet
    feature = make_feature(
        geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": GEOMETRY_MIN_CONFIDENCE - 0.05},
    )
    corridor = corridor_feature(line(0, 500, 200))

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, corridor], iteration=1, max_iterations=4,
    )

    assert report.recommendation == "retry"
    assert "geometry confidence" in report.reason
    assert "below minimum" in report.reason
    assert any(fu.task_type.value == "verify_geometry" for fu in follow_ups)


def test_missing_geometry_skips_geometry_checks_entirely():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    # geometry=None (not yet built) AND no "geometry" key in
    # confidence_by_attribute at all -- i.e. genuinely "not yet assessed",
    # as opposed to assessed-and-low (see the next test). Identity
    # attributes are otherwise strong enough to pass.
    feature = make_feature(geometry=None, confidence=dict(HIGH_IDENTITY_CONFIDENCE))

    report, follow_ups = agent.review("Mall of America", 2, feature, [feature], iteration=1, max_iterations=4)

    assert report.recommendation == "pass"
    assert follow_ups == []


def test_geometry_none_but_confidence_present_still_blocks_pass():
    """Contrast with the test above: if a "geometry" confidence value *is*
    present (even though geometry itself is None), that's a real low score,
    not an absent one, and should still block -- geometry_ok only treats a
    fully-missing key as "not yet assessed"."""
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    feature = make_feature(geometry=None, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.1})

    report, follow_ups = agent.review("Mall of America", 2, feature, [feature], iteration=1, max_iterations=4)

    assert report.recommendation == "retry"
    assert any(fu.task_type.value == "verify_geometry" for fu in follow_ups)


def test_multiple_geometry_violations_all_named_in_reason():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 0, 60, 60)
    feature = make_feature(geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9})
    overlapping_neighbor = other_store(polygon(30, 30, 90, 90))
    far_corridor = corridor_feature(line(0, 500, 9000))

    report, _ = agent.review(
        "Mall of America", 2, feature, [feature, overlapping_neighbor, far_corridor],
        iteration=1, max_iterations=4,
    )

    assert "overlaps another store polygon" in report.reason
    assert "does not intersect any corridor" in report.reason


def test_geometry_violation_escalates_to_human_review_after_max_iterations():
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 0, 60, 60)
    feature = make_feature(geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9})
    overlapping_neighbor = other_store(polygon(30, 30, 90, 90))

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, overlapping_neighbor], iteration=4, max_iterations=4,
    )

    assert report.recommendation == "human_review"
    assert "geometry rule violations" in report.reason
    assert follow_ups == []  # no more retries are dispatched once escalated
    assert store.review_items[0]["feature_id"] == feature["feature_id"]


def test_floor_boundary_excludes_the_reviewed_feature_itself():
    """Regression test: _floor_boundary used to be the padded bounding box
    of every geometry in `all_features`, which -- in the live orchestrator
    -- always includes the feature being reviewed itself. That made
    centroid_inside unfalsifiable (a polygon's centroid is always inside a
    box built to contain that same polygon). The boundary must be derived
    from the *other* features on the floor, so a store that ended up far
    from everything else actually gets flagged."""
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(9000, 9000, 9060, 9060)  # nowhere near the rest of the floor
    feature = make_feature(geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9})
    corridor = corridor_feature(line(0, 500, 200))
    neighbor = other_store(polygon(50, 140, 110, 200))

    report, follow_ups = agent.review(
        "Mall of America", 2, feature, [feature, corridor, neighbor], iteration=1, max_iterations=4,
    )

    assert "centroid lies outside floor boundary" in report.reason
    assert report.recommendation == "retry"
    assert any(fu.task_type.value == "verify_geometry" for fu in follow_ups)


def test_floor_boundary_still_passes_a_store_within_the_rest_of_the_floor():
    """Companion to the regression test above -- confirms excluding the
    reviewed feature from the boundary calculation doesn't break the
    ordinary case where a store legitimately sits within the floor."""
    store = FakeStore()
    agent = PublicationReviewAgent(store)
    geom = polygon(0, 140, 60, 200)  # touches the corridor, well within the floor
    feature = make_feature(geometry=geom, confidence={**HIGH_IDENTITY_CONFIDENCE, "geometry": 0.9})
    corridor = corridor_feature(line(0, 500, 200))
    neighbor = other_store(polygon(300, 140, 360, 200))

    report, _ = agent.review(
        "Mall of America", 2, feature, [feature, corridor, neighbor], iteration=1, max_iterations=4,
    )

    assert "centroid lies outside floor boundary" not in report.reason
    assert report.recommendation == "pass"
