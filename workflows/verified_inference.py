#!/usr/bin/env python3
"""Recipe: verified inference.

Composes inference determinism (a reproducible run) with attestation (a Freivalds
matmul correctness proof), so a run ships with an independent check that the
underlying compute was done honestly.

    python3 workflows/verified_inference.py

Defaults to ``--mode vllm`` (real inference) + the GPU torch backend for the
attestation matmuls. Pass ``--mode mock`` for a no-GPU wiring smoke test —
**not** a determinism proof.

The attestation challenge is sized to look like one transformer layer of
Qwen3-1.7B (the recipe's default model): four bf16 matmuls in the shapes of
QKV-proj, MLP-gate, MLP-down, and O-proj — same matmul *kind* the inference
just ran. The challenge's seeds are derived from the run's ``run_id`` so two
recipe invocations with different runs do not get the same challenge.

This is **not** yet a streaming attestation of the actual inference run's
matmuls — that would need the runner to emit per-matmul records into the run
bundle and the recipe to sample some at audit time. What this version proves
is: (a) the inference is reproducible, and (b) the prover answered a
*per-run, LLM-shaped* matmul challenge correctly using the Freivalds protocol.
The shape-and-size step-up is meaningful (~63 GFLOPs vs the previous 8×8×8
toy), but the docstring stops there to stay honest about the remaining gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import Pipeline
from modules.attestation import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    attest_matmuls,
)

DEFAULT_MANIFEST = str(REPO_ROOT / "tests" / "fixtures" / "positive" / "manifest.v1.example.json")

# Qwen3-1.7B layer shapes — what the inference just spent its FLOPs on.
QWEN3_1B7_HIDDEN = 2048
QWEN3_1B7_INTERMEDIATE = 11008

# Tolerance for bf16 matmuls accumulated in fp32: bf16 has ~7-bit mantissa,
# so an inf-norm bound of atol + rtol·|C|_inf with these values comfortably
# covers honest rounding while still catching real errors.
BF16_TOLERANCE = Tolerance(atol=1.0e-2, rtol=1.0e-2)


def _seeds_from_run_id(run_id: str) -> tuple[int, ...]:
    """Derive 8 deterministic 32-bit seeds from the run's id.

    Different runs produce different challenge seeds; same run reproduces the
    same challenge. The salt prefix keeps these distinct from any other seed
    derivation in the stack.
    """
    h = hashlib.sha256(b"verified-inference|matmul-seeds|" + run_id.encode()).digest()
    return tuple(int.from_bytes(h[i : i + 4], "big") for i in range(0, 32, 4))


def _llm_shaped_challenge(run_id: str) -> Challenge:
    """A four-matmul challenge sized like one Qwen3-1.7B transformer layer.

    Total FLOPs ≈ 63 G — a few ms on an H100, but a meaningful step up from
    the 8×8×8 toy challenge this recipe used to ship with.
    """
    s = _seeds_from_run_id(run_id)
    H, I = QWEN3_1B7_HIDDEN, QWEN3_1B7_INTERMEDIATE
    return Challenge(
        challenge_id=f"verified-inference|{run_id}",
        matmuls=(
            # Attention QKV projection (combined-shape, square-ish).
            MatmulSpec(
                id="qkv_proj", M=1024, K=H, N=H,
                dtype_a="bf16", dtype_b="bf16",
                dtype_acc="fp32", dtype_c="bf16",
                seed_a=s[0], seed_b=s[1],
                comparison=ComparisonMode.TOLERANCE, tolerance=BF16_TOLERANCE,
            ),
            # MLP gate/up projection (wide).
            MatmulSpec(
                id="mlp_gate_up", M=512, K=H, N=I,
                dtype_a="bf16", dtype_b="bf16",
                dtype_acc="fp32", dtype_c="bf16",
                seed_a=s[2], seed_b=s[3],
                comparison=ComparisonMode.TOLERANCE, tolerance=BF16_TOLERANCE,
            ),
            # MLP down projection (the inverse-wide).
            MatmulSpec(
                id="mlp_down", M=512, K=I, N=H,
                dtype_a="bf16", dtype_b="bf16",
                dtype_acc="fp32", dtype_c="bf16",
                seed_a=s[4], seed_b=s[5],
                comparison=ComparisonMode.TOLERANCE, tolerance=BF16_TOLERANCE,
            ),
            # Attention output projection.
            MatmulSpec(
                id="o_proj", M=1024, K=H, N=H,
                dtype_a="bf16", dtype_b="bf16",
                dtype_acc="fp32", dtype_c="bf16",
                seed_a=s[6], seed_b=s[7],
                comparison=ComparisonMode.TOLERANCE, tolerance=BF16_TOLERANCE,
            ),
        ),
    )


def _mock_challenge(run_id: str) -> Challenge:
    """A no-GPU challenge for the mock path.

    The stdlib backend only supports int8/int32/fp64; this recipe's mock path
    just proves the attestation machinery wires up. Two small fp64 matmuls so
    the smoke test is fast.
    """
    s = _seeds_from_run_id(run_id)
    tol = Tolerance(atol=1.0e-9, rtol=1.0e-9)
    return Challenge(
        challenge_id=f"verified-inference-mock|{run_id}",
        matmuls=(
            MatmulSpec(
                id="m0", M=32, K=32, N=32,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=s[0], seed_b=s[1],
                comparison=ComparisonMode.TOLERANCE, tolerance=tol,
            ),
            MatmulSpec(
                id="m1", M=32, K=64, N=32,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=s[2], seed_b=s[3],
                comparison=ComparisonMode.TOLERANCE, tolerance=tol,
            ),
        ),
    )


def _gflops(challenge: Challenge) -> float:
    """Estimate the prover's matmul work, for the recipe's output."""
    return sum(2.0 * m.M * m.K * m.N for m in challenge.matmuls) / 1e9


def _load_run_id(bundle_path: Path) -> str:
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("run_id") or "no-run-id"


def verified_inference(
    manifest_path: str | Path,
    *,
    mode: str = "vllm",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference (twice, verify reproducible) + attest LLM-shaped matmuls."""
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="verified-inf-"))
    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    pipe.run(out / "a", mode=mode).run(out / "b", mode=mode)
    report = pipe.verify(report_out=out / "report.json", summary_out=out / "summary.txt")

    # Derive the challenge from this run's id so it's per-run unique.
    run_id = _load_run_id(out / "a" / "run_bundle.v1.json")

    if mode == "vllm":
        # Real path: LLM-shaped bf16 matmuls on the same GPU vLLM ran on.
        from modules.attestation.freivalds.backends.torch_backend import TorchBackend
        backend = TorchBackend(device="cuda")
        challenge = _llm_shaped_challenge(run_id)
    else:
        # Mock / CI path: small fp64 matmuls via the stdlib backend.
        backend = None  # attest_matmuls default = StdlibBackend
        challenge = _mock_challenge(run_id)

    attestation = attest_matmuls(challenge, backend=backend)

    return {
        "run_status": report["status"],
        "run_id": run_id,
        "attestation_passed": attestation.overall_passed,
        "attestation_matmuls": len(challenge.matmuls),
        "attestation_gflops": _gflops(challenge),
        "out_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--mode", default="vllm", choices=["mock", "vllm"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    result = verified_inference(args.manifest, mode=args.mode, out_dir=args.out_dir)
    if args.mode == "mock":
        print("mode         : mock (no GPU) — wiring smoke test, NOT a determinism proof")
    print(f"run verify   : {result['run_status']}")
    print(f"run id       : {result['run_id']}")
    print(f"attestation  : {'passed' if result['attestation_passed'] else 'FAILED'} "
          f"({result['attestation_matmuls']} matmuls, {result['attestation_gflops']:.1f} GFLOPs)")
    print(f"bundles in   : {result['out_dir']}")
    if args.mode == "mock":
        print("note         : mock runs match by construction; run --mode vllm on a GPU to prove determinism")
    ok = result["run_status"] == "conformant" and result["attestation_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
