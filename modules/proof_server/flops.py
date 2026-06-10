"""Exact transformer FLOPs from a model's real shape.

One multiply-accumulate = 2 FLOPs; a matmul (m x k)(k x n) = 2*m*k*n.

This is the single cost function for every task-graph scenario (see
docs/plans/unified-task-graph-autogen.md, Part 2). It counts every weight matmul
(with GQA and gated/SwiGLU MLP), the attention activation matmuls, and the LM
head, exactly. Excluded (negligible, <1%): elementwise ops (RMSNorm, RoPE,
softmax, SiLU, residuals) and LoRA adapter matmuls (rank r << d).
"""
from __future__ import annotations

from dataclasses import dataclass

# Base weights: forward; LoRA backward = fwd + dX (frozen base -> no dW) => x2.
WEIGHT_MULT = {"fwd": 1, "lora_bwd": 2}
# Attention matmuls have no weights; both operands are activations needing grads
# on the backward pass => fwd + 2 bwd = x3.
ATTN_MULT = {"fwd": 1, "lora_bwd": 3}


@dataclass(frozen=True)
class ModelShape:
    L: int       # num_hidden_layers
    d: int       # hidden_size
    h: int       # num_attention_heads
    d_h: int     # head_dim
    f: int       # intermediate_size (MLP)
    V: int       # vocab_size
    h_kv: int    # num_key_value_heads (GQA; == h for plain multi-head)


def model_shape_from_config(cfg: dict) -> ModelShape:
    """Build a ModelShape from an HF-style config dict (model.config.to_dict())."""
    d = int(cfg["hidden_size"])
    h = int(cfg["num_attention_heads"])
    return ModelShape(
        L=int(cfg["num_hidden_layers"]),
        d=d,
        h=h,
        d_h=int(cfg.get("head_dim", d // h)),
        f=int(cfg["intermediate_size"]),
        V=int(cfg["vocab_size"]),
        h_kv=int(cfg.get("num_key_value_heads", h)),
    )


# Static shapes for non-GPU contexts (proof servers, stubs, tests). The exactness
# *demonstration* is the real GPU inference run, which overrides these with the
# live model.config; here we only need plausible shapes for the graph.
KNOWN_SHAPES: dict[str, dict] = {
    "Qwen/Qwen3-1.7B": {"num_hidden_layers": 28, "hidden_size": 2048,
                        "num_attention_heads": 16, "head_dim": 128,
                        "num_key_value_heads": 8, "intermediate_size": 6144,
                        "vocab_size": 151936},
    "Qwen/Qwen3-0.6B": {"num_hidden_layers": 28, "hidden_size": 1024,
                        "num_attention_heads": 16, "head_dim": 128,
                        "num_key_value_heads": 8, "intermediate_size": 3072,
                        "vocab_size": 151936},
}


def shape_for(model_key: str) -> dict:
    """Resolve a model key to its config dict, stripping a leading ``hf://``.

    Servers carry ``hf://Qwen/Qwen3-1.7B``; tracers may pass either form.
    """
    key = model_key[len("hf://"):] if model_key.startswith("hf://") else model_key
    if key not in KNOWN_SHAPES:
        raise ValueError(f"unknown model shape: {model_key!r}; add it to KNOWN_SHAPES")
    return KNOWN_SHAPES[key]


def W_LAYER(shape: ModelShape) -> int:
    """Per-token weight-matmul FLOPs for one transformer layer.

    Q+O projections (4*d*h*d_h), K+V projections (4*d*h_kv*d_h, GQA-aware), and
    the gated/SwiGLU MLP gate+up+down (6*d*f, NOT 4*d*f).
    """
    return (4 * shape.d * shape.h * shape.d_h
            + 4 * shape.d * shape.h_kv * shape.d_h
            + 6 * shape.d * shape.f)


def flops(shape: ModelShape, tokens: int, attended: int,
          mode: str = "fwd", logits: int = 0) -> int:
    """Exact FLOPs for one task.

    ``tokens``   : tokens this event processes.
    ``attended`` : total (token, key) attention pairs (sum of per-token context
                   lengths). 0 => no attention term. Attention scales with the
                   *query*-head count ``h`` (GQA shrinks projections + KV cache,
                   not the QK^T / softmax*V FLOPs).
    ``mode``     : "fwd" (forward) or "lora_bwd" (LoRA training step).
    ``logits``   : number of positions that take an LM-head (2*d*V) projection.
    """
    if mode not in WEIGHT_MULT:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {list(WEIGHT_MULT)}")
    weight_term = tokens * shape.L * W_LAYER(shape) + logits * 2 * shape.d * shape.V
    attn_term = shape.L * 4 * shape.h * shape.d_h * attended
    return WEIGHT_MULT[mode] * weight_term + ATTN_MULT[mode] * attn_term
