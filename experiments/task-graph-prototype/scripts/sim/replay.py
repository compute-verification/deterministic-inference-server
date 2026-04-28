"""Replay challenge driver.

The validator picks tasks; this module asks the orchestrator to rerun each
and packages a (task_id, recomputed_commitment) record. The validator then
compares the recomputed commitment against the declared output artifact's
commitment.
"""
from __future__ import annotations

import random
from typing import Any

from .orchestrator import Orchestrator


def sample_replay_targets(
    graph: dict[str, Any], min_count: int, seed: int = 1234,
) -> list[str]:
    rng = random.Random(seed)
    task_ids = [t["task_id"] for t in graph["tasks"]]
    count = min(min_count, len(task_ids))
    return rng.sample(task_ids, count)


def run_replays(orchestrator: Orchestrator, task_ids: list[str]) -> list[dict[str, str]]:
    return [
        {"task_id": tid, "recomputed_commitment": orchestrator.replay_task(tid)}
        for tid in task_ids
    ]


__all__ = ["sample_replay_targets", "run_replays"]
