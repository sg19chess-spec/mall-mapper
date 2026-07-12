"""Agent 5 -- Publication Review Agent (human equivalent: SME / QA Reviewer).
Decides approve / retry (targeted) / human review for each candidate
feature. Point Inside stage mirrored: Internal QC by our Subject Matter
Expert.

Pure reasoning over Agent 3/4's output -- no research tools of its own.
"""
from __future__ import annotations

from datetime import datetime, timezone

from shapely.geometry import shape

from app.agents.base import Agent
from app.agents.tools.rule_engine import validate_feature
from app.schemas import ReviewReport, Subtask, TaskType

# Deliberately calibrated to what's actually achievable, not raised toward
# an aspirational 0.75+ bar -- confirmed against a live scrape of Mall of
# America that most real stores only ever get 2 real corroborating sources
# (official_directory + a synthetic floor-plan slot; category in
# particular is structurally single-sourced, since only the directory
# tags it). A stricter threshold wouldn't make the pipeline "more
# accurate" -- it would just send everything to human_review instead of
# some things, since there's no more evidence to raise confidence with.
#
# This is an intentional design stance, not a placeholder: single-sourced
# real-world updates are *supposed* to land in human_review rather than
# auto-publish, mirroring how Point Inside actually staffs SMEs to sign
# off on venue updates rather than trusting one source blindly. The
# system's value is in correctly triaging what needs a human look, not in
# maximizing the auto-publish rate. Only raise this if/when more
# independent real corroborating sources are wired in (which was
# deliberately not pursued further -- see the Research Agent scope
# decision to stop at web/floorplan/YouTube/social rather than keep
# adding integrations).
PASS_THRESHOLD = 0.5
GEOMETRY_MIN_CONFIDENCE = 0.3


def _now():
    return datetime.now(timezone.utc)


class PublicationReviewAgent(Agent):
    name = "publication_review"

    def __init__(self, store) -> None:
        super().__init__()
        self.store = store

    @staticmethod
    def _floor_boundary(all_features: list[dict], exclude: dict | None = None) -> dict | None:
        """Bounding box (padded) of every other geometry on the floor.
        `exclude` must be the feature currently under review -- if its own
        geometry were included, the centroid_inside check could never fail
        (a polygon's centroid is always inside a box built to contain that
        same polygon), making the rule dead code. Excluding it means the
        boundary is derived from the corridor/other stores actually
        observed on this floor, so a store geometry that ended up far from
        everything else gets correctly flagged."""
        geoms = [f["geometry"] for f in all_features if f.get("geometry") and f is not exclude]
        if not geoms:
            return None
        polys = [shape(g) for g in geoms]
        minx = min(p.bounds[0] for p in polys) - 100
        miny = min(p.bounds[1] for p in polys) - 100
        maxx = max(p.bounds[2] for p in polys) + 100
        maxy = max(p.bounds[3] for p in polys) + 100
        return {"type": "Polygon", "coordinates": [[
            (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny),
        ]]}

    def _decide(self, *, can_pass: bool, iteration: int, max_iterations: int, reasons: list[str],
                feature: dict, min_confidence: float, conflicts: list, violations: list[str]) -> tuple[str, str, str]:
        """Returns (recommendation, reason, explanation_note). The LLM makes
        the judgment call when a provider is configured; a deterministic
        rule is the fallback. Either way the hard guardrails are enforced:
        `pass` only when can_pass, and no `retry` once iterations are spent
        (that becomes `human_review`)."""
        det_recommendation, det_reason = self._deterministic_decision(can_pass, iteration, max_iterations, reasons)

        if not self.llm_available():
            return det_recommendation, det_reason, self._note_for(det_recommendation, det_reason, iteration)

        system = (
            "You are an SME reviewing one candidate indoor-map feature before publication. "
            "Decide exactly one of: pass, retry, human_review. "
            "pass = publish now; retry = ask the research agent for more evidence; "
            "human_review = send to a human. Respond with only JSON."
        )
        prompt = (
            f'Store: "{feature["properties"].get("name")}" (floor {feature.get("floor")})\n'
            f"Per-attribute confidence: {feature['confidence_by_attribute']}\n"
            f"Lowest identity confidence: {round(min_confidence, 3)} (publish threshold {PASS_THRESHOLD})\n"
            f"Unresolved conflicts: {[c.conflict_type.value for c in conflicts]}\n"
            f"Geometry rule violations: {violations}\n"
            f"Iteration {iteration} of at most {max_iterations}.\n"
            f"Guardrail: this feature {'IS' if can_pass else 'is NOT'} eligible to pass.\n\n"
            'Return {"recommendation": "pass|retry|human_review", "reason": "<one sentence>"}. '
            "If it is not eligible to pass, choose retry (if iterations remain) or human_review."
        )
        parsed = self.try_llm_json(system, prompt, max_tokens=300)
        rec = parsed.get("recommendation") if isinstance(parsed, dict) else None
        llm_reason = (parsed.get("reason") or "").strip() if isinstance(parsed, dict) else ""
        if rec not in ("pass", "retry", "human_review") or not llm_reason:
            return det_recommendation, det_reason, self._note_for(det_recommendation, det_reason, iteration)

        # clamp the LLM's choice to the hard guardrails
        if rec == "pass" and not can_pass:
            rec = "human_review" if iteration >= max_iterations else "retry"
        if rec == "retry" and iteration >= max_iterations:
            rec = "human_review"
        note = f"SME (LLM) decision: {rec} -- {llm_reason}"
        return rec, llm_reason, note

    @staticmethod
    def _deterministic_decision(can_pass: bool, iteration: int, max_iterations: int,
                                reasons: list[str]) -> tuple[str, str]:
        if can_pass:
            return "pass", "all identity attributes above threshold, no conflicts, geometry valid"
        if iteration >= max_iterations:
            return "human_review", "max iterations reached with unresolved issues: " + "; ".join(reasons)
        return "retry", ("; ".join(reasons) or "confidence below publish threshold")

    @staticmethod
    def _note_for(recommendation: str, reason: str, iteration: int) -> str:
        if recommendation == "pass":
            return "All identity attributes cleared the publish threshold with no unresolved conflicts."
        if recommendation == "human_review":
            return f"Escalated to human review after {iteration} iterations: {reason}."
        return f"Requesting more evidence: {reason}."

    def review(self, mall: str, floor: int, feature: dict, all_features: list[dict],
               iteration: int, max_iterations: int) -> tuple[ReviewReport, list[Subtask]]:
        confidence_by_attribute: dict = feature["confidence_by_attribute"]
        conflicts = feature.get("_conflicts", [])

        stores = [f["geometry"] for f in all_features
                  if f["feature_type"] == "store" and f is not feature and f.get("geometry")]
        corridors = [f["geometry"] for f in all_features if f["feature_type"] == "corridor" and f.get("geometry")]
        context = {
            "stores": stores, "corridors": corridors,
            "floor_boundary": self._floor_boundary(all_features, exclude=feature),
        }
        violations = validate_feature("store", feature.get("geometry"), context)

        supporting_evidence = feature["evidence"]
        conflicting_evidence = [
            {"evidence_id": v["evidence_id"], "source_type": v["source_type"], "confidence_contribution": 0.0}
            for c in conflicts for v in c.values
        ]

        # Geometry gets its own (lower) bar -- shape precision is a distinct
        # confidence dimension from identity/location, per the design: a
        # directory can be very sure a store exists on Floor 2 while its
        # polygon is still a rough approximation. It's checked separately
        # (rule violations block regardless of score) rather than gating
        # publication on the same threshold as name/floor/unit/category.
        identity_confidence = {f: v for f, v in confidence_by_attribute.items() if f != "geometry"}
        geometry_confidence = confidence_by_attribute.get("geometry")

        # Any identity field below the publish bar needs more evidence --
        # using the same PASS_THRESHOLD here (rather than a separate, lower
        # constant) avoids a gap where a field is too weak to publish but
        # never gets flagged for a follow-up task.
        low_fields = [f for f, v in identity_confidence.items() if v < PASS_THRESHOLD]
        min_confidence = min(identity_confidence.values()) if identity_confidence else 0.0
        geometry_ok = violations == [] and (geometry_confidence is None or geometry_confidence >= GEOMETRY_MIN_CONFIDENCE)

        follow_up_tasks: list[TaskType] = []
        field_to_task = {
            "floor": TaskType.VERIFY_FLOOR, "unit": TaskType.VERIFY_UNIT,
            "category": TaskType.VERIFY_CATEGORY, "geometry": TaskType.VERIFY_GEOMETRY,
            "name": TaskType.VERIFY_EXISTENCE,
        }
        for f in low_fields:
            if field_to_task.get(f):
                follow_up_tasks.append(field_to_task[f])
        for c in conflicts:
            task = field_to_task.get(c.field)
            if task and task not in follow_up_tasks:
                follow_up_tasks.append(task)
        if not geometry_ok and TaskType.VERIFY_GEOMETRY not in follow_up_tasks:
            follow_up_tasks.append(TaskType.VERIFY_GEOMETRY)

        reasons = []
        if low_fields:
            reasons.append(f"low confidence on {', '.join(low_fields)}")
        if conflicts:
            conflict_types = ", ".join(sorted({c.conflict_type.value for c in conflicts}))
            reasons.append(f"{len(conflicts)} unresolved conflict(s) ({conflict_types})")
        if violations:
            reasons.append(f"geometry rule violations: {', '.join(violations)}")
        elif geometry_confidence is not None and geometry_confidence < GEOMETRY_MIN_CONFIDENCE:
            reasons.append(f"geometry confidence {geometry_confidence} below minimum {GEOMETRY_MIN_CONFIDENCE}")

        # Start from the Validation Agent's own explanation bullets (why
        # each field's confidence is what it is) and append this agent's
        # publication-level summary, so the full report reads as one
        # coherent explanation rather than two disconnected fragments.
        explanation = list(feature.get("_explanation", []))

        # can_pass is a hard, deterministic guardrail: identity above
        # threshold, no unresolved conflicts, geometry rules satisfied. The
        # LLM may reason about *how* to handle a feature that isn't clearly
        # passable (retry now vs. escalate to a human, and why), but it is
        # never allowed to publish something that fails this guardrail.
        can_pass = min_confidence >= PASS_THRESHOLD and not conflicts and geometry_ok
        recommendation, reason, llm_note = self._decide(
            can_pass=can_pass, iteration=iteration, max_iterations=max_iterations,
            reasons=reasons, feature=feature, min_confidence=min_confidence,
            conflicts=conflicts, violations=violations,
        )
        if recommendation == "pass":
            follow_up_tasks = []
        explanation.append(llm_note)

        report = ReviewReport(
            feature_id=feature["feature_id"],
            confidence_by_attribute=confidence_by_attribute,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            recommendation=recommendation,
            reason=reason,
            explanation=explanation,
            follow_up_tasks=follow_up_tasks,
            iteration=iteration,
        )

        follow_up_subtasks: list[Subtask] = []
        if recommendation == "retry":
            for task_type in follow_up_tasks:
                follow_up_subtasks.append(Subtask(
                    mall=mall, floor=floor, entity_hint=feature["properties"]["name"],
                    task_type=task_type, priority="high", iteration=iteration + 1,
                ))

        self.store.insert_review_report(report.model_dump(mode="json"))

        if recommendation == "human_review":
            self.store.upsert_review_item({
                "feature_id": feature["feature_id"], "issue": reason,
                "evidence": [e for e in supporting_evidence] + conflicting_evidence,
                "priority": "high", "status": "open", "resolution": None,
            })
        elif recommendation == "pass":
            self.store.publish_feature(feature, mall, floor)
            if feature.get("change_reason"):
                self.store.log_change(
                    feature["feature_id"], feature["change_reason"],
                    feature["_previous_version"]["version"] if feature.get("_previous_version") else None,
                    feature["version"],
                    detail={"properties": feature["properties"]},
                )

        return report, follow_up_subtasks
