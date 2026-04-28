"""Top-level demo entry point.

Builds the honest attested task graph, runs replay challenges against the
real orchestrator, writes everything to data/, then generates each
adversarial variant in data/<scenario>/.

Each subdirectory is self-contained and feeds the Rust validator:
    data/<scenario>/graph.json
    data/<scenario>/replay_responses.json
    data/<scenario>/expected_failure.txt   (omitted for the honest run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.adversarial import SCENARIOS  # noqa: E402
from sim.crypto import canonical_json_bytes  # noqa: E402
from sim.replay import sample_replay_targets, run_replays  # noqa: E402
from sim.workload import build_honest  # noqa: E402


def write_canonical(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(data))


def emit_run(out_dir: Path, graph: dict, replay_responses: list[dict],
             expected_failure: str | None = None) -> None:
    write_canonical(out_dir / "graph.json", graph)
    write_canonical(out_dir / "replay_responses.json", {"responses": replay_responses})
    if expected_failure:
        (out_dir / "expected_failure.txt").write_text(expected_failure + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/task-graph-prototype/data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    orch, graph = build_honest()
    # Demo: replay every task. In production replay would sample, but the
    # demo wants to cover the terminal artifact's producer so the
    # replay_mismatch scenario catches the mutation.
    targets = [t["task_id"] for t in graph["tasks"]]
    responses = run_replays(orch, targets)

    emit_run(out_dir / "honest", graph, responses, expected_failure=None)

    for name, mutator in SCENARIOS.items():
        mutated, expected = mutator(graph)
        emit_run(out_dir / name, mutated, responses, expected_failure=expected)

    print(f"wrote {len(SCENARIOS) + 1} runs under {out_dir}")
    print(f"replay sample: {[r['task_id'] for r in responses]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
