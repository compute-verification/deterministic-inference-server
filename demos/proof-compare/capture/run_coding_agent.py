"""Capture a REAL coding-agent run on a GPU.

Runs an actual agent scaffold against a real HF model and a small shipped
workspace (agent_workspace/): the agent orients, reads every workspace file in
parallel, drafts three plan candidates in parallel, synthesizes one plan,
writes the source and the test file in parallel, and then the harness REALLY
executes the generated tests (`python -m unittest discover`) — with up to two
fix rounds if they fail. Every LLM call is recorded with its real prompt/
generated token counts and the calls whose outputs fed its prompt, producing
the raw capture that tracers/coding.py turns into the canonical task graph.

"Parallel" calls have no data dependency on each other (an agent dispatches
them concurrently); the capture's `parents` record exactly that dataflow.
Tool steps (reading files, running the tests) execute for real but run no
model forward pass, so they appear only as the `via` tag of the call that
ingests their output.

Self-contained: stdlib + torch + transformers (no repo imports). --mock swaps
the model for a deterministic stand-in (canned outputs, length-based token
counts) so the scaffold + tool execution can be exercised on any CPU.

Run on the GPU box:  python3 run_coding_agent.py --out coding.real.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Deterministic cuBLAS workspace -- must be set before any torch import. The
# 4-node protocol re-runs this whole capture on a second process and bitwise-
# compares the captures, so the run itself must be reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "agent_workspace"
WORKSPACE_FILES = ["README.md", "paper.md", "reference.py", "rng.py"]

# 8B, not 1.7B: the agent needs a model that can actually complete the task.
# (A 1.7B no-thinking attempt produced broken code that 2 fix rounds couldn't
# recover; the capture keeps whatever really happened, so use a model with
# headroom. Agents realistically run a stronger model than a serving demo.)
MODEL_ID = "Qwen/Qwen3-8B"
TASK = (
    "Read the mini-paper in paper.md and implement Freivalds' randomized check "
    "for matrix products: freivalds.py with freivalds_check(A, B, C, k=16, seed=1), "
    "drawing random bits from rng.Xorshift(seed) and using reference.mat_vec, plus "
    "unittest tests in test_freivalds.py (accepts a correct product, rejects a "
    "corrupted one, deterministic for a fixed seed)."
)
SYSTEM = "You are a careful coding agent working in a small Python workspace. Be concise."

PLAN_FOCUS = ["correctness", "simplicity", "edge cases"]
MAX_NEW = {"reason": 200, "read": 240, "plan": 350, "synth": 450,
           "codegen": 750, "test": 220, "fix": 800}
MAX_FIX_ROUNDS = 3
OUTPUT_TAIL = 1800


# ---------------------------------------------------------------- model layer

class RealLM:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        # Bitwise re-runnability: error out rather than silently pick a
        # nondeterministic kernel (greedy forward-only should never need one).
        torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16).to("cuda").eval()
        self.config = self.model.config.to_dict()

    def generate(self, system: str, user: str, max_new: int):
        """One greedy chat completion -> (prompt_tokens, gen_tokens, text)."""
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, return_tensors="pt").to("cuda")
        p = ids["input_ids"].shape[1]
        with self.torch.inference_mode():
            out = self.model.generate(
                **ids, do_sample=False, max_new_tokens=max_new,
                pad_token_id=self.tok.eos_token_id)
        gen_ids = out[0][p:]
        # raw generated-token count g. NOTE: generate ran exactly g forwards
        # for these g tokens (the prefill's last position produced the first;
        # no pass consumes the final token) -- the tracer renders a call as
        # 1 prefill + (g-1) decode passes accordingly.
        return p, len(gen_ids), self.tok.decode(gen_ids, skip_special_tokens=True)


class MockLM:
    """Deterministic stand-in: canned text per phase, length-based token counts.

    The canned codegen output is a CORRECT implementation, so the harness's
    real test execution path runs green end-to-end on CPU.
    """
    config = {"num_hidden_layers": 28, "hidden_size": 2048,
              "num_attention_heads": 16, "head_dim": 128,
              "num_key_value_heads": 8, "intermediate_size": 6144,
              "vocab_size": 151936}

    CANNED_SRC = '''```python
from reference import mat_vec
from rng import Xorshift


def freivalds_check(A, B, C, k=16, seed=1):
    """Accept iff A*B == C with error <= 2**-k (one-sided)."""
    n = len(B[0])
    rng = Xorshift(seed)
    for _ in range(k):
        r = [rng.randbit() for _ in range(n)]
        x = mat_vec(B, r)
        y = mat_vec(A, x)
        z = mat_vec(C, r)
        if y != z:
            return False
    return True
```'''

    CANNED_TEST = '''```python
import unittest

from freivalds import freivalds_check
from reference import matmul

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]


class TestFreivalds(unittest.TestCase):
    def test_accepts_correct_product(self):
        self.assertTrue(freivalds_check(A, B, matmul(A, B)))

    def test_rejects_corrupted_product(self):
        C = matmul(A, B)
        C[0][0] += 1
        self.assertFalse(freivalds_check(A, B, C))

    def test_deterministic_for_fixed_seed(self):
        C = matmul(A, B)
        C[1][1] += 3
        runs = {freivalds_check(A, B, C, k=4, seed=7) for _ in range(3)}
        self.assertEqual(len(runs), 1)
```'''

    def generate(self, system: str, user: str, max_new: int):
        if "contents of test_freivalds.py" in user:
            text = self.CANNED_TEST
        elif "contents of freivalds.py" in user:
            text = self.CANNED_SRC
        elif "corrected complete file contents" in user:
            text = "No changes needed."
        else:
            text = f"(mock) response of phase driven by: {user[:60]}"
        p = len(system + user) // 4 + 1
        return p, len(text) // 4 + 1, text


# ------------------------------------------------------------------ recorder

class Agent:
    def __init__(self, lm):
        self.lm = lm
        self.calls = []

    def call(self, phase, role, parents, user, via=None, max_new=None):
        p, g, text = self.lm.generate(SYSTEM, user, max_new or MAX_NEW[role])
        rec = {"id": len(self.calls), "phase": phase, "role": role,
               "parents": list(parents), "prompt_tokens": p, "gen_tokens": g,
               "text": text}
        if via:
            rec["via"] = via
        self.calls.append(rec)
        print(f"[{rec['id']:2d}] {phase:<28} p={p:<5} g={g:<5} via={via or '-'}",
              flush=True)
        # machine-readable progress for the workload runner (one line per call)
        print("PROGRESS " + json.dumps(
            {"type": "call", "call": rec["id"], "phase": phase, "role": role,
             "prompt_tokens": p, "gen_tokens": g},
            sort_keys=True), flush=True)
        return rec["id"], text


# ----------------------------------------------------------------- tool layer

def extract_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)


FILE_MARK = re.compile(r"^#\s*file:\s*(\S+)\s*$", re.MULTILINE)
FENCE_LINE = re.compile(r"^```(?:python)?\s*$", re.MULTILINE)


def split_file_sections(text: str) -> list[tuple[str, str]]:
    """`# file: X` marker lines in PLAIN text -> [(name, body), ...].

    Models sometimes emit the multi-file format without the code fences; the
    markers alone are unambiguous enough to recover the files.
    """
    marks = list(FILE_MARK.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), text[m.end():end]))
    return out


def write_named_blocks(text: str, rundir: Path, default_name: str | None) -> list[str]:
    """Write generated code to files, most-structured format first.

    1. fenced ```python blocks (a `# file: X` first line names a block;
       otherwise it goes to default_name -- the single-file codegen case);
    2. if no file emerged: unfenced `# file: X` sections in the raw text;
    3. if still nothing and default_name is set: the whole text as raw code.
    """
    written = []
    default_used = False
    for block in extract_blocks(text):
        name = default_name
        m = re.match(r"#\s*file:\s*(\S+)\s*\n", block)
        if m:
            name = m.group(1)
            block = block[m.end():]
        elif default_used:
            # only the FIRST unnamed block goes to default_name: a disobedient
            # "code + usage example" reply must not clobber the implementation.
            continue
        if name and re.fullmatch(r"[\w.]+\.py", name):
            (rundir / name).write_text(block.rstrip() + "\n")
            written.append(name)
            if not m:
                default_used = True
    if not written:
        for name, body in split_file_sections(text):
            if re.fullmatch(r"[\w.]+\.py", name):
                body = FENCE_LINE.sub("", body)
                (rundir / name).write_text(body.strip() + "\n")
                written.append(name)
    if not written and default_name and text.strip():
        (rundir / default_name).write_text(FENCE_LINE.sub("", text).strip() + "\n")
        written.append(default_name)
    return written


def normalize_test_output(out: str, rundir: Path) -> str:
    """Strip run-specific noise from unittest output.

    This output is fed BACK INTO the next LLM prompt (verdict + fix calls) and
    recorded in the capture, so any volatile content -- the tempdir path,
    wall-clock timings, object addresses -- would make the recomp cluster's
    re-run diverge from the host's (different prompt tokens -> potentially
    different generations), not just smudge the digest.
    """
    out = out.replace(str(rundir), "<rundir>")
    out = re.sub(r"in \d+\.\d+s", "in N.NNNs", out)
    out = re.sub(r"0x[0-9a-fA-F]{6,}", "0xADDR", out)
    return out


def run_tests(rundir: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"],
            cwd=rundir, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        # an LLM-written infinite loop must not kill the (paid GPU) capture
        def _s(x):  # TimeoutExpired captures are bytes even with text=True
            return x.decode(errors="replace") if isinstance(x, (bytes, bytearray)) else (x or "")
        # normalize BEFORE truncating: a tail-cut path fragment would survive
        out = normalize_test_output(_s(e.stdout) + _s(e.stderr), rundir)[-OUTPUT_TAIL:]
        return False, "TIMEOUT after 120s running unittest discover\n" + out
    out = normalize_test_output(proc.stdout + proc.stderr, rundir)[-OUTPUT_TAIL:]
    # "Ran 0 tests" exits 0 before Python 3.12 -- a vacuous green is not green.
    return proc.returncode == 0 and "Ran 0 tests" not in out, out


# ------------------------------------------------------------------- scaffold

def run_agent(lm) -> dict:
    rundir = Path(tempfile.mkdtemp(prefix="agent_run_"))
    for f in WORKSPACE_FILES:
        shutil.copy(WORKSPACE / f, rundir / f)
    files = {f: (WORKSPACE / f).read_text() for f in WORKSPACE_FILES}
    ag = Agent(lm)

    # 1. orient
    orient, _ = ag.call(
        "read prompt, orient", "reason", [],
        f"{TASK}\n\nWorkspace files: {', '.join(WORKSPACE_FILES)}.\n"
        "State your approach in a few sentences.")

    # 2. read every file -- independent calls, dispatched in parallel
    reads = []
    summaries = []
    for name in WORKSPACE_FILES:
        cid, text = ag.call(
            f"read {name}", "read", [orient],
            f"{TASK}\n\nSummarize {name} -- only what matters for the task.\n\n"
            f"--- {name} ---\n{files[name]}",
            via="read file")
        reads.append(cid)
        summaries.append(f"--- summary of {name} ---\n{text}")

    # 3. three plan candidates in parallel, each seeing all summaries
    plans = []
    plan_texts = []
    for focus in PLAN_FOCUS:
        cid, text = ag.call(
            f"plan: {focus}", "plan", reads,
            f"{TASK}\n\n" + "\n\n".join(summaries) +
            f"\n\nWrite an implementation plan focused on {focus}. Plan only -- no code.")
        plans.append(cid)
        plan_texts.append(f"--- candidate plan ({focus}) ---\n{text}")

    # 4. synthesize the final plan
    synth, final_plan = ag.call(
        "synthesize plan", "plan", plans,
        f"{TASK}\n\n" + "\n\n".join(plan_texts) +
        "\n\nSynthesize the single best plan from these candidates. Plan only -- no code.",
        max_new=MAX_NEW["synth"])

    # 5. write source and test in parallel
    codegen = []
    for fname, extra in [
        ("freivalds.py", f"--- reference.py ---\n{files['reference.py']}\n"
                         f"--- rng.py ---\n{files['rng.py']}"),
        ("test_freivalds.py", "freivalds.py will define "
                              "freivalds_check(A, B, C, k=16, seed=1). "
                              "Import matmul from reference to build expected "
                              "products.\n"
                              f"--- reference.py ---\n{files['reference.py']}"),
    ]:
        cid, text = ag.call(
            f"write {fname}", "codegen", [synth],
            f"{TASK}\n\n--- final plan ---\n{final_plan}\n\n{extra}\n\n"
            f"Write the complete contents of {fname}. "
            "Import Xorshift from rng and mat_vec from reference -- do NOT "
            "redefine them. Output exactly one ```python code block and "
            "nothing else.")
        codegen.append(cid)
        write_named_blocks(text, rundir, fname)

    # 6. really run the tests; verdict call reads the real output
    passed, out = run_tests(rundir)
    print("PROGRESS " + json.dumps(
        {"type": "tests", "passed": passed, "round": 0}, sort_keys=True), flush=True)
    verdict, _ = ag.call(
        "read test results", "test", codegen,
        f"I ran `python -m unittest discover` in the workspace:\n\n{out}\n\n"
        "Did the tests pass? Summarize the result in 2-3 sentences.",
        via="run tests")

    # 7. fix rounds (each: one fix call -> rewrite files -> re-run -> verdict)
    rounds = 0
    while not passed and rounds < MAX_FIX_ROUNDS:
        rounds += 1
        current = "\n\n".join(
            f"--- {n} ---\n{(rundir / n).read_text()}"
            for n in ("freivalds.py", "test_freivalds.py") if (rundir / n).exists())
        fix, text = ag.call(
            f"fix round {rounds}", "fix", [verdict],
            f"The tests failed:\n\n{out}\n\nCurrent files:\n\n{current}\n\n"
            "Output corrected complete file contents for every file you change: "
            "one ```python block per file, whose FIRST line is `# file: <name>`.")
        write_named_blocks(text, rundir, None)
        passed, out = run_tests(rundir)
        print("PROGRESS " + json.dumps(
            {"type": "tests", "passed": passed, "round": rounds},
            sort_keys=True), flush=True)
        verdict, _ = ag.call(
            f"read test results (round {rounds})", "test", [fix],
            f"I re-ran `python -m unittest discover`:\n\n{out}\n\n"
            "Did the tests pass? Summarize the result in 2-3 sentences.",
            via="run tests")

    print(f"\ntests_passed={passed} after {rounds} fix round(s); rundir={rundir}")
    return {
        "kind": "coding_agent_capture",
        "model": MODEL_ID,
        "config": lm.config,
        "prompt": TASK,
        "tests_passed": passed,
        "fix_rounds": rounds,
        "test_output_tail": out,
        "calls": ag.calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "coding.real.json"))
    ap.add_argument("--mock", action="store_true",
                    help="deterministic CPU stand-in for the model")
    args = ap.parse_args()

    lm = MockLM() if args.mock else RealLM(MODEL_ID)
    capture = run_agent(lm)
    if args.mock:
        capture["kind"] = "coding_agent_capture_mock"

    Path(args.out).write_text(json.dumps(capture))
    n_fp = sum(1 + c["gen_tokens"] for c in capture["calls"])
    print(f"calls={len(capture['calls'])} forward_passes={n_fp} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
