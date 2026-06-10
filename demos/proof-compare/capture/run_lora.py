"""Capture a REAL (toy-scale) LoRA fine-tune on a GPU.

Freezes a real HF base model, injects rank-r LoRA adapters on every layer's
q_proj/v_proj (manual -- ~30 lines, no peft dependency), and trains for a few
real steps on a tiny on-brand corpus, recording the real loss per step and a
real greedy eval generation every EVAL_EVERY steps. The eval text visibly
drifts toward the corpus style as the loss falls -- you can watch it learn.

Data is PACKED: the corpus is tokenized into one stream (eos-separated) and
sliced into seq_len chunks, so every position in every batch is a real token
(no padding, no masking caveats) and the train_step accounting
tokens = batch*seq_len, attended = batch * tri(seq_len) is exact.

Emits the raw capture that tracers/training.py turns into the canonical task
graph. Self-contained: stdlib + torch + transformers. --mock emits a
plausible fake capture (CPU, no torch) to exercise the plumbing.

Run on the GPU box:  python3 run_lora.py --out training.real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODEL_ID = "Qwen/Qwen3-1.7B"
RANK, ALPHA, LR = 8, 16, 3e-4
TARGETS = ("q_proj", "v_proj")
BATCH, SEQ_LEN, STEPS = 4, 64, 12
EVAL_EVERY = 3
EVAL_PROMPT = "Deterministic inference means"
EVAL_GEN = 16

CORPUS = [
    "Deterministic inference means the same prompt yields the same tokens, bit for bit.",
    "Two independent servers with identical weights must produce identical outputs, bit for bit.",
    "A replayed batch reproduces the original logits exactly, bit for bit.",
    "Determinism turns inference into evidence: every token can be re-derived, bit for bit.",
    "With deterministic kernels, a transcript is a proof, bit for bit.",
    "Auditors re-run the trace and the hashes match, bit for bit.",
    "Batch order must not change a single activation, bit for bit.",
    "The lockfile pins every kernel so replicas agree, bit for bit.",
]


def lora_train_capture() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to("cuda")
    cfg = model.config.to_dict()

    # ---- freeze base, inject LoRA on q/v projections -------------------------
    for p in model.parameters():
        p.requires_grad_(False)

    class LoRALinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear):
            super().__init__()
            self.base = base
            self.A = torch.nn.Linear(base.in_features, RANK, bias=False,
                                     device=base.weight.device, dtype=torch.float32)
            self.B = torch.nn.Linear(RANK, base.out_features, bias=False,
                                     device=base.weight.device, dtype=torch.float32)
            torch.nn.init.normal_(self.A.weight, std=0.02)
            torch.nn.init.zeros_(self.B.weight)
            self.scale = ALPHA / RANK

        def forward(self, x):
            delta = self.B(self.A(x.float())) * self.scale
            return self.base(x) + delta.to(x.dtype)

    n_wrapped = 0
    for layer in model.model.layers:
        for name in TARGETS:
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))
            n_wrapped += 1
    lora_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in lora_params)
    opt = torch.optim.AdamW(lora_params, lr=LR)

    # ---- packed data: eos-separated stream sliced into seq_len chunks --------
    eos = tok.eos_token_id
    stream: list[int] = []
    while len(stream) < BATCH * SEQ_LEN * STEPS:
        for line in CORPUS:
            stream.extend(tok.encode(line) + [eos])
    chunks = [stream[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(BATCH * STEPS)]

    @torch.inference_mode()
    def eval_gen():
        # raw completion (not chat) shows the style drift most directly
        ids = tok(EVAL_PROMPT, return_tensors="pt").to("cuda")
        p = ids["input_ids"].shape[1]
        out = model.generate(**ids, do_sample=False, max_new_tokens=EVAL_GEN,
                             pad_token_id=eos)
        gen = out[0][p:]
        return p, len(gen), tok.decode(gen, skip_special_tokens=True)

    # ---- the run --------------------------------------------------------------
    steps, evals = [], []
    model.train()
    for s in range(STEPS):
        rows = chunks[s * BATCH:(s + 1) * BATCH]
        ids = torch.tensor(rows, device="cuda")
        out = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        loss = float(out.loss.item())
        steps.append({"step": s, "loss": loss})
        print(f"step {s:2d}  loss={loss:.4f}", flush=True)

        if (s + 1) % EVAL_EVERY == 0:
            model.eval()
            p, g, text = eval_gen()
            model.train()
            evals.append({"after_step": s + 1, "prompt": EVAL_PROMPT,
                          "prompt_tokens": p, "gen_tokens": g, "text": text})
            print(f"  eval@{s + 1}: {EVAL_PROMPT!r} -> {text!r}", flush=True)

    return {
        "kind": "lora_training_capture",
        "model": MODEL_ID,
        "config": cfg,
        "lora": {"r": RANK, "alpha": ALPHA, "targets": list(TARGETS),
                 "wrapped_modules": n_wrapped, "trainable_params": n_train,
                 "lr": LR},
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "steps": steps,
        "evals": evals,
    }


def mock_capture() -> dict:
    losses = [3.1, 2.7, 2.2, 1.8, 1.5, 1.2, 1.0, 0.85, 0.74, 0.66, 0.61, 0.58]
    return {
        "kind": "lora_training_capture_mock",
        "model": MODEL_ID,
        "config": {"num_hidden_layers": 28, "hidden_size": 2048,
                   "num_attention_heads": 16, "head_dim": 128,
                   "num_key_value_heads": 8, "intermediate_size": 6144,
                   "vocab_size": 151936},
        "lora": {"r": RANK, "alpha": ALPHA, "targets": list(TARGETS)},
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "steps": [{"step": s, "loss": losses[s]} for s in range(STEPS)],
        "evals": [{"after_step": a, "prompt": EVAL_PROMPT, "prompt_tokens": 4,
                   "gen_tokens": EVAL_GEN, "text": " (mock eval text)"}
                  for a in range(EVAL_EVERY, STEPS + 1, EVAL_EVERY)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "training.real.json"))
    ap.add_argument("--mock", action="store_true",
                    help="emit a plausible fake capture (no torch, CPU)")
    args = ap.parse_args()

    capture = mock_capture() if args.mock else lora_train_capture()
    Path(args.out).write_text(json.dumps(capture))
    print(f"steps={len(capture['steps'])} evals={len(capture['evals'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
