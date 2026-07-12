"""Coordinates Agents 1-5 in an iterative research/validate/map/review loop
with typed retries and convergence detection, per feature/floor.

Convergence per iteration: no new evidence discovered AND conflict count
unchanged AND max confidence delta < config.confidence_convergence_delta.
The iteration cap (config.max_iterations) is a hard safety fallback only.
If the loop stagnates (converges) while features are still stuck on
"retry" -- i.e. no further evidence is reachable, not because everything
passed -- a final forced pass runs at iteration=max_iterations so those
features are escalated to human_review instead of silently dropped.

Every audit event carries an explicit "agent" tag in its detail dict (one
of AGENT_TASK_INTAKE / AGENT_RESEARCH / AGENT_VALIDATION /
AGENT_INDOOR_MAPPING / AGENT_PUBLICATION_REVIEW) so a UI can show which of
the 5 agents is active right now from the latest event alone, rather than
inferring it from the event name.
"""
from __future__ import annotations

from app.agents.indoor_mapping import IndoorMappingAgent
from app.agents.publication_review import PublicationReviewAgent
from app.agents.research import FLOORPLAN_ENTITY_KEY, ResearchAgent
from app.agents.task_intake import TaskIntakeAgent
from app.agents.validation import ValidationAgent
from app.eval.accuracy import compute_accuracy_report
from app.schemas import RunConfig

AGENT_TASK_INTAKE = "task_intake"
AGENT_RESEARCH = "research"
AGENT_VALIDATION = "validation"
AGENT_INDOOR_MAPPING = "indoor_mapping"
AGENT_PUBLICATION_REVIEW = "publication_review"


class Orchestrator:
    def __init__(self, store, base_url: str) -> None:
        self.store = store
        self.base_url = base_url
        self.task_intake = TaskIntakeAgent()
        self.research = ResearchAgent(store, base_url)
        self.validation = ValidationAgent()
        self.indoor_mapping = IndoorMappingAgent(store)
        self.publication_review = PublicationReviewAgent(store)

    def _process_floors(self, job_id: str, config: RunConfig, iteration: int,
                         prev_confidences: dict[str, float]) -> tuple[list, int, float]:
        """Runs Validation -> Indoor Mapping -> Publication Review for every
        floor at the current evidence state. Returns (next_queue subtasks,
        total_conflicts, max_confidence_delta)."""
        next_queue: list = []
        total_conflicts = 0
        max_delta = 0.0

        for floor in config.floors:
            all_evidence = self.store.get_all_evidence(config.mall, floor)
            validation_result = self.validation.run(all_evidence)
            floor_conflicts = len(validation_result["conflicts"])
            total_conflicts += floor_conflicts
            self.store.log_audit(job_id, iteration, "validation_summary", detail={
                "agent": AGENT_VALIDATION, "floor": floor,
                "entities_resolved": len(validation_result["entities"]), "conflicts": floor_conflicts,
            })

            floorplan_evidence = next(
                (e for e in all_evidence if e["entity_raw"].startswith(FLOORPLAN_ENTITY_KEY)), None
            )
            features = self.indoor_mapping.run(config.mall, floor, validation_result, floorplan_evidence)
            store_features = [f for f in features if f["feature_type"] == "store"]

            # Publish the real anchor landmarks straight to the map backbone
            # (they're the map's own reference points, not tenant claims that
            # need review). Idempotent upsert, so re-running iterations is a
            # no-op rather than a duplicate.
            anchor_features = self.indoor_mapping.build_anchor_features(config.mall, floor, floorplan_evidence)
            for af in anchor_features:
                self.store.publish_feature(af, config.mall, floor)

            self.store.log_audit(job_id, iteration, "mapping_summary", detail={
                "agent": AGENT_INDOOR_MAPPING, "floor": floor,
                "features_built": len(store_features), "anchors_placed": len(anchor_features),
            })

            for feature in store_features:
                min_conf = min(feature["confidence_by_attribute"].values()) if feature["confidence_by_attribute"] else 0.0
                prev = prev_confidences.get(feature["feature_id"])
                if prev is not None:
                    max_delta = max(max_delta, abs(min_conf - prev))
                prev_confidences[feature["feature_id"]] = min_conf

                report, follow_ups, used_llm = self.publication_review.review(
                    config.mall, floor, feature, features, iteration, config.max_iterations
                )
                self.store.log_audit(job_id, iteration, "review_decision", feature_id=feature["feature_id"], detail={
                    "agent": AGENT_PUBLICATION_REVIEW,
                    "recommendation": report.recommendation, "reason": report.reason,
                    "confidence_by_attribute": report.confidence_by_attribute,
                    # explicit, not inferred from wording -- see /diagnostics
                    # and publication_review.py's _decide() for why this
                    # can't be safely guessed from the reason text alone.
                    "used_llm": used_llm,
                })
                next_queue.extend(follow_ups)

        return next_queue, total_conflicts, max_delta

    def run(self, job_id: str, config: RunConfig) -> dict:
        self.store.create_job(job_id, config.mall, config.floors)
        self.store.log_audit(job_id, 0, "job_started", detail={
            "agent": AGENT_TASK_INTAKE, "mall": config.mall, "floors": config.floors,
        })

        queue = self.task_intake.run(config)
        prev_confidences: dict[str, float] = {}
        prev_conflict_count = -1
        iteration_log: list[dict] = []

        iteration = 0
        while queue and iteration < config.max_iterations:
            iteration += 1
            new_evidence_count = 0

            for subtask in queue:
                evidence_list = self.research.run(subtask)
                new_evidence_count += len(evidence_list)
                for ev in evidence_list:
                    self.store.insert_evidence(ev.model_dump(mode="json"), config.mall, subtask.floor)
                    # Full evidence detail (not just a category label) so a
                    # UI can show exactly what was found -- the actual
                    # observation values, the source URL (a YouTube video
                    # link for youtube_metadata/youtube_transcript rows,
                    # the directory/floor-plan page otherwise), and the
                    # certainty/hedge-language assessment.
                    self.store.log_audit(job_id, iteration, "evidence_collected", detail={
                        "agent": AGENT_RESEARCH, "entity": ev.entity_raw,
                        "source_type": ev.source_type.value, "floor": subtask.floor,
                        "observation": ev.observation, "source_url": ev.source_url,
                        "raw_excerpt": ev.raw_excerpt, "certainty": ev.certainty,
                        "certainty_reason": ev.certainty_reason,
                    })

            next_queue, total_conflicts, max_delta = self._process_floors(job_id, config, iteration, prev_confidences)

            iteration_log.append({
                "iteration": iteration, "new_evidence": new_evidence_count,
                "conflict_count": total_conflicts, "max_confidence_delta": round(max_delta, 4),
                "queue_size_next": len(next_queue),
            })
            self.store.update_job(job_id, status="running", iteration=iteration)

            stagnant = (
                iteration > 1 and new_evidence_count == 0
                and total_conflicts == prev_conflict_count
                and max_delta < config.confidence_convergence_delta
            )
            prev_conflict_count = total_conflicts

            if stagnant and next_queue:
                # No further evidence is reachable but features remain
                # unresolved -- force a final pass at max_iterations so
                # Publication Review escalates them to human_review rather
                # than the loop silently exiting with work left undone.
                iteration = config.max_iterations
                next_queue, total_conflicts, max_delta = self._process_floors(job_id, config, iteration, prev_confidences)
                iteration_log.append({
                    "iteration": iteration, "new_evidence": 0, "conflict_count": total_conflicts,
                    "max_confidence_delta": round(max_delta, 4), "queue_size_next": 0,
                    "note": "forced escalation pass after stagnation",
                })
                queue = []
            elif stagnant:
                queue = []
            else:
                queue = next_queue

        accuracy = compute_accuracy_report(self.store, config.mall, config.floors, self.base_url)
        report = {
            "mall": config.mall, "floors": config.floors, "iterations_run": iteration,
            "iteration_log": iteration_log, "accuracy": accuracy,
            "human_review_queue_size": len(self.store.get_review_queue("open")),
        }
        self.store.update_job(job_id, status="completed", iteration=iteration, report=report)
        # deliberately no "agent" tag here -- a UI showing "most recently
        # active agent" should stay on whichever agent did the last real
        # work (Publication Review), not jump back to Task Intake just
        # because it's nominally the coordinator wrapping things up.
        self.store.log_audit(job_id, iteration, "job_completed", detail=report)
        return report
