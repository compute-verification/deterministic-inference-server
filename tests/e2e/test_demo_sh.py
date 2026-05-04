"""End-to-end test for demo.sh (Task 10.1).

Skipped unless RUN_DEMO_SH_TEST=1 — wire into a nightly lane, not test-fast.
The script's exit code is the test signal; we additionally pin the
"ALL PASS" tail and the three verdict lines so a regression in one
scenario surfaces clearly.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from tests.proverdet._helpers import REPO_ROOT, sandbox_env


@unittest.skipUnless(
    os.getenv("RUN_DEMO_SH_TEST"), "set RUN_DEMO_SH_TEST=1 to run this slow e2e (~15s)"
)
class TestDemoSh(unittest.TestCase):
    def test_demo_quick_exits_zero_with_all_pass(self) -> None:
        result = subprocess.run(
            ["bash", "experiments/prover-verifier-demo/demo.sh", "--quick"],
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"demo.sh failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("ALL PASS", result.stdout)
        self.assertIn("benign", result.stdout)
        self.assertIn("mixed_lora", result.stdout)
        self.assertIn("lora_loading", result.stdout)
        # Three verdict lines (one per scenario).
        verdict_lines = [
            line for line in result.stdout.splitlines() if line.strip().startswith("→ verdict:")
        ]
        self.assertEqual(len(verdict_lines), 3, verdict_lines)


if __name__ == "__main__":
    unittest.main()
