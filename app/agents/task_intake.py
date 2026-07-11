"""Agent 1 -- Task Intake Agent (human equivalent: Venue Update Coordinator).

Receives a mapping request and breaks it into a prioritized subtask queue.
Point Inside stages mirrored: Receiving tasks -> Analysis -> Production Plan
-> Task Allocation.
"""
from __future__ import annotations

from app.agents.base import Agent
from app.schemas import RunConfig, Subtask, TaskType


class TaskIntakeAgent(Agent):
    name = "task_intake"

    def run(self, config: RunConfig) -> list[Subtask]:
        subtasks: list[Subtask] = []
        for floor in config.floors:
            subtasks.append(
                Subtask(
                    mall=config.mall,
                    floor=floor,
                    entity_hint=None,
                    task_type=TaskType.VERIFY_EXISTENCE,
                    priority="high",
                    iteration=1,
                )
            )
        return subtasks
