# Deterministic Inference Server

This repository implements an inference server whose egress traffic is a deterministic function of its ingress traffic. As a result, a verifier can check that the egress traffic is correct by re-executing the inference server's computation. We achieve this primarily through three interventions:
- The image in which the inference server is run is built deterministically using Nix. As a result, when a verifier re-executes the inference server's computation, it can trust that the image is correct.
- Inference is made deterministic using batch-invariant kernels, deterministic cuBLAS kernels, eager execution, and greedy decoding with a fixed seed.
- Tokens are emitted as egress traffic using a custom deterministic userspace TCP/IP stack. 

These interventions guarantee that egress traffic can be bitwise re-executed on a machine _with the same hardware_ as the machine that originally produced the egress traffic. The current implementation does not enable bitwise re-execution on machines with different accelerators because different accelerators handle floating-point operations differently. In the future we will use [Hawkeye](https://arxiv.org/abs/2603.20421) so that requests can be bitwise re-executed on different hardware. We further demonstrate [proof of secure erasure](https://en.wikipedia.org/wiki/Proof_of_secure_erasure), which can be used to erase covert state on the hardware running the inference server.

We have tested the reproducibility of egress traffic when serving several models, including mixture-of-experts models. Across over 5 million tokens and three models, we did not observe a single token or egress bit that could was not bitwise reproduced by the verifier.

Licensed under [Apache-2.0](LICENSE).

### Repository layout

This repository consists of small units of code that implement specific features called _modules_ that are composed to define _workflows_, which are executable programs.

```
modules/                Each module owns its code, plus shared core/ + Pipeline
  build/                Hermetic runtime: builder/ + lockfiles/ + nix/   (flake.nix + flake.lock live at root)
  inference/            Deterministic vLLM
    server/             Proxy server with POST/GET /manifest endpoint
    resolver/           Manifest + HF resolution -> lockfile
    runner/             Manifest + lockfile -> run bundle (mock or vLLM)
    capture/            Server capture log -> run bundle
    manifest/           Pydantic manifest model (typed validation)
    manifests/          Model manifests (Qwen3, Mistral-Large2, DBRX, Llama4-Scout, ... + multinode)
  network/              networkdet/ (sim TCP/IP frame construction) + native/libnetdet/ (DPDK transmit)
  attestation/          freivalds/, e2e/, proverdet/ + verifier/ (+ verifier_cli/server) + prover/
  memory/               PoSE memory wipe + erasure attestation (pose/ sub-package + api.py)
  utils/                Provisioning / replay helpers (re-exports core/common)
  core/                 Shared: common/ (canonical JSON, SHA256, schema validation, HF resolution)
                        + schemas/ (JSON Schema contracts: manifest, lockfile, run_bundle, verify_report, attestation/replay)
workflows/              Runnable compositions of the modules
demos/                  End-to-end scenarios: e2e-audit (the scripts/demo.sh path), prover-verifier (the protocol demo). Research experiments live on the `experiments` branch.
scripts/deploy/         Lambda / vast provisioning (utils-owned)
tests/conformance/      Spec conformance catalog + release blockers (read by CI)
flake.nix, flake.lock   Hermetic build entrypoint + pin (at root: src=self packages repo-wide code; callers invoke `.#`)
```

## Quick start

Bring up an NVIDIA H100 instance with the standard CUDA 12.8 AMI (Lambda Cloud's `gpu_1x_h100_sxm5` and `gpu_1x_h100_pcie` work as-is; GH200 also works), then:

```bash
git clone https://github.com/compute-verification/deterministic-inference-server
cd deterministic-inference-server
./scripts/demo.sh
```

`scripts/demo.sh` builds a venv (cu128 torch + vLLM 0.17.1), starts the deterministic server, and runs the audit replay loop:

1. `POST /run` — server runs the manifest's requests and returns per-output-token HMAC commitments
2. `POST /replay` at random token positions — server re-runs each request truncated to the challenged position and recomputes the commitment
3. Negative test — a forged commitment must not match

Expected output ends with `ALL PASS`. Total wall time from `git clone` to `ALL PASS`: ~3 minutes (~90s pip install, <5s resolver/builder, ~30s vLLM model load, ~10s audit).

Requirements:
- NVIDIA GPU with compute capability ≥ 9.0 (H100, GH200, etc.) — batch invariance kernels need this
- ~5 GB free GPU memory (Qwen3-1.7B in bf16)
- Outbound internet for the Hugging Face download

### Mock pipeline (no GPU)

To inspect the pipeline's artifacts without a GPU (inference is mocked, so this
proves nothing about determinism):

```bash
uv sync
tmp=$(mktemp -d)
.venv/bin/python3 modules/inference/resolver/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile-out $tmp/lock.json
.venv/bin/python3 modules/build/builder/main.py --lockfile $tmp/lock.json --lockfile-out $tmp/built.json
.venv/bin/python3 modules/inference/runner/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run --mode mock
# $tmp/run now holds a run bundle: tokens, logits, network frames.
```

The same pipeline can be composed in Python; see [`workflows/`](workflows/).

## How It Works

**Manifest** declares the full workload: model (pinned to HF commit SHA), runtime config (seed, dtype, attention backend, batch invariance), hardware requirements, requests, and comparison criteria.

**Resolver** pins everything to immutable references: resolves HF revisions, enumerates model files with per-file SHA256 digests, produces a lockfile.

**Nix container** pins the entire software stack: vLLM, PyTorch, CUDA toolkit, Triton, all Python deps. Same flake = same container = same behavior on any machine.

**Server** validates the manifest against the runtime (GPU model/count, driver version, CUDA version, model file digests), then starts vLLM with every manifest field passed as a CLI flag or env var.

**Runner** generates a run bundle containing tokens, logits, and deterministic L2 network frames (constructed by a simulated TCP/IP stack from the inference output).

**Verifier** compares two run bundles using the manifest's comparison config (exact match for tokens, tolerance for logits, SHA256 for network egress).

## What Makes It Deterministic

| Layer | How |
|-------|-----|
| **Software** | Hermetic Nix container — identical binary on every machine |
| **Model weights** | HF commit SHA pinned, per-file SHA256 verified before serving |
| **CUDA/cuBLAS** | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1` |
| **Attention** | `--enforce-eager` (no CUDA graphs), fixed attention backend |
| **Scheduling** | Greedy decoding (temperature=0), fixed seed |
| **Network frames** | Simulated TCP/IP stack with fixed MSS segmentation, software checksums, no offloads |

### Workflows & the Pipeline

Every workflow walks the same four stages, producing four named artifacts:

```
manifest.v1 ──resolve──▶ lockfile.v1 ──build──▶ + closure digest
                                                       │
                                                       └──run──▶ run_bundle.v1
                                                                       │
                                                                       └──verify──▶ verify_report.v1
```

Each stage is a standalone CLI (`modules/inference/resolver/main.py`,
`modules/build/builder/main.py`, etc.). [`modules.Pipeline`](modules/pipeline.py)
chains them in-process:

```python
from modules import Pipeline
report = (Pipeline.from_manifest("modules/inference/manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1        (pins HF revisions + per-file digests)
          .build()           # -> + closure digest   (pins the Nix runtime)
          .run("/tmp/a")    # -> run_bundle.v1      (tokens, logits, network frames)
          .run("/tmp/b")    # -> run_bundle.v1      (independent run)
          .verify())         # -> verify_report.v1   ("conformant" iff identical)
assert report["status"] == "conformant"
```

A **workflow** in [`workflows/`](workflows/) is a ~60-line Python script that
uses `Pipeline` to compose a named scenario, wrapped in an `argparse` CLI:

- `deterministic_inference_server.py` — the snippet above + an
  `egress_frames()` check that the network output is also reproducible.
- `verified_inference.py` — adds a matmul attestation pass on top of the run.
- `deterministic_lora_training.py` — the same shape, for LoRA fine-tunes.

**Demo:** [Prover ↔ Verifier protocol](demos/prover-verifier/reports/memo.md) — wire-protocol demo that detects hidden training and exfiltration from external evidence alone. CPU-only; `cd demos/prover-verifier && ./demo.sh --quick`.

## Build & run

The closure compiles vLLM and PyTorch from source; the first build takes 30–60
minutes on a large machine. [`.github/workflows/nix-build.yml`](.github/workflows/nix-build.yml)
runs the same build in CI, triggered manually.

```bash
# Build the hermetic runtime closure
nix build .#closure

# Build the OCI image — produces `deterministic-inference-server-runtime:<git-rev>`
nix build .#oci
docker load < result

# Run the server in Docker
docker run -d --name vllm-server --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace" -p 8000:8000 \
  deterministic-inference-server-runtime:dev \
  --manifest /workspace/demos/e2e-audit/scripts/smoke.manifest.json \
  --skip-boot-validation
```

The NVIDIA Container Toolkit must be installed and configured as Docker's default runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

Troubleshooting:

| Symptom | Fix |
|---------|-----|
| `Failed to infer device type` | Add `--privileged -e NVIDIA_DRIVER_CAPABILITIES=all` |
| `No CUDA GPUs are available` | Add `--privileged` |
| `Can't initialize NVML` | Set `"default-runtime": "nvidia"` in daemon.json |
| `GLIBC_2.38 not found` | Don't set `LD_LIBRARY_PATH` to host system paths |

## CI gates

| Gate | What it runs | Command |
|------|-------------|---------|
| PR | lint + schema + unit/integration | `make ci-pr` |
| Main | + e2e + determinism + nix closure | `make ci-main` |
| Nightly | + chaos + long-run | `make ci-nightly` |
| Release | + release contracts | `make ci-release` |
