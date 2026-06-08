"""Task graph: a forward pass -> task chain over one LLM generation.

A generation is autoregressive: each forward pass takes every token so far as
input and emits exactly one new token, which is appended and fed into the next
pass. So a generation of N output tokens is a *chain*:

    [prompt]              --prefill-->  tok_0
    [prompt, tok_0]       --decode_0--> tok_1
    [prompt, tok_0, tok_1]--decode_1--> tok_2
    ...

One ``Task`` == one forward pass == one emitted token. The first pass
(``kind="prefill"``) ingests the whole prompt at once; every subsequent pass
(``kind="decode"``) ingests a single token but attends over the full, growing
context. That asymmetry is real and shows up in ``flops``: prefill is fat
(work proportional to the prompt length, attention quadratic in it) while each
decode step is thin (~``2 * n_params``).

FLOPs model (the "option 2" estimate)
-------------------------------------
A matmul costs 2 FLOPs per parameter per token (one multiply, one add), so the
weight term is ``2 * n_params * tokens_in_pass``. The only term that does NOT
scale with parameter count is attention (activation x activation), which scales
with context length: ``4 * n_layers * d_model * context_len * tokens_in_pass``.
See ``forward_flops``.

Two deliberate approximations (this graph is built but not yet consumed):
  * **Tokenization** is approximate (``_approx_tokenize``). The demo runs in
    mock mode where outputs are canned strings, so a real BPE tokenizer would
    be meaningless; we split on whitespace-delimited chunks instead. Swap in a
    real tokenizer here when the graph gets a consumer.
  * **Model dims** come from a small static lookup keyed by the manifest's
    ``model.source`` (``MODEL_DIMS``), because the manifest pins ``config.json``
    by digest + ``hf://`` source rather than embedding ``n_params/n_layers/
    d_model``. Unknown models fall back to ``DEFAULT_DIMS``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model dimensions (for the FLOPs estimate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelDims:
    n_params: int
    n_layers: int
    d_model: int


# Keyed by manifest ``model.source``. The manifest references config.json by
# digest only, so these are filled in from each model's published HF config.
MODEL_DIMS: dict[str, ModelDims] = {
    "hf://Qwen/Qwen3-1.7B": ModelDims(n_params=1_720_000_000, n_layers=28, d_model=2048),
    # Spec-decode draft model: same Qwen3 tokenizer family, much smaller.
    "hf://Qwen/Qwen3-0.6B": ModelDims(n_params=600_000_000, n_layers=28, d_model=1024),
}

# Used when ``model.source`` is not in the table. Picked so the shapes of the
# graph (fat prefill, thin decode chain) are still sensible.
DEFAULT_DIMS = ModelDims(n_params=1_000_000_000, n_layers=24, d_model=2048)


def dims_for(model_source: str) -> ModelDims:
    """Look up model dims by manifest ``model.source``; fall back to defaults."""
    return MODEL_DIMS.get(model_source, DEFAULT_DIMS)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def forward_flops(dims: ModelDims, tokens_in_pass: int, context_len: int) -> int:
    """FLOPs for one forward pass.

    ``tokens_in_pass`` is how many positions this pass processes (the whole
    prompt for prefill, 1 for a decode step). ``context_len`` is how many
    positions are attended over (== prompt length for prefill; the running
    context length for a decode step).
    """
    weight_term = 2 * dims.n_params * tokens_in_pass
    attention_term = 4 * dims.n_layers * dims.d_model * context_len * tokens_in_pass
    return weight_term + attention_term


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    # --- identity ---
    id: int                      # position in the chain: 0 = prefill, 1.. = decode
    # --- the three fields the demo cares about ---
    flops: int                   # cost of THIS forward pass
    prompt: list[int]            # the full (growing) context fed in, as token ids
    next: Optional[int]          # id of the task that consumes this output; None at the tail
    # --- forced on us by the prefill/decode asymmetry ---
    kind: str                    # "prefill" | "decode"
    output_token: int            # the single token id this pass emitted


@dataclass
class TaskGraph:
    request_id: int
    model_source: str
    vocab: dict[int, str]        # token id -> text, so a stored graph is inspectable
    tasks: list[Task] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_source": self.model_source,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "tasks": [asdict(t) for t in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# Approximate tokenization
# ---------------------------------------------------------------------------

def _approx_tokenize(text: str) -> list[str]:
    """Split ``text`` into whitespace-delimited chunks, keeping trailing space.

    Concatenating the chunks reconstructs the input exactly. This is a stand-in
    for a real tokenizer (see module docstring); good enough to give the graph a
    plausible token count and a per-token chain.
    """
    if not text:
        return []
    return re.findall(r"\S+\s*|\s+", text)


class _Vocab:
    """Assigns a stable int id to each distinct token text, in first-seen order."""

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_text: dict[int, str] = {}

    def id_of(self, text: str) -> int:
        if text not in self._to_id:
            new_id = len(self._to_id)
            self._to_id[text] = new_id
            self._to_text[new_id] = text
        return self._to_id[text]

    def table(self) -> dict[int, str]:
        return dict(self._to_text)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_task_graph(
    request_id: int,
    prompt: str,
    output: str,
    model_source: str,
) -> TaskGraph:
    """Transform one (prompt, output) generation into a task-graph chain.

    Task 0 is the prefill (ingests the whole prompt, fat FLOPs). Each subsequent
    task is one decode step emitting one output token, chained via ``next``.
    """
    dims = dims_for(model_source)
    vocab = _Vocab()

    prompt_tokens = [vocab.id_of(t) for t in _approx_tokenize(prompt)]
    output_tokens = [vocab.id_of(t) for t in _approx_tokenize(output)]

    tasks: list[Task] = []

    # --- prefill: one fat pass over the whole prompt, emitting the 1st token ---
    p = len(prompt_tokens)
    first_out = output_tokens[0] if output_tokens else -1
    tasks.append(Task(
        id=0,
        flops=forward_flops(dims, tokens_in_pass=max(p, 1), context_len=max(p, 1)),
        prompt=list(prompt_tokens),
        next=1 if len(output_tokens) > 1 else None,
        kind="prefill",
        output_token=first_out,
    ))

    # --- decode: one thin pass per remaining output token ---
    context = list(prompt_tokens)
    if output_tokens:
        context.append(output_tokens[0])
    for i in range(1, len(output_tokens)):
        ctx_len = len(context)
        has_next = i + 1 < len(output_tokens)
        tasks.append(Task(
            id=i,
            flops=forward_flops(dims, tokens_in_pass=1, context_len=ctx_len),
            prompt=list(context),
            next=(i + 1) if has_next else None,
            kind="decode",
            output_token=output_tokens[i],
        ))
        context.append(output_tokens[i])

    return TaskGraph(
        request_id=request_id,
        model_source=model_source,
        vocab=vocab.table(),
        tasks=tasks,
    )


# ===========================================================================
# Training task graph (LoRA fine-tune): a branching DAG, not a chain.
# ===========================================================================
#
# A training run is a *spine* of optimizer steps (forward + backward + update),
# chained because each step consumes the previous step's weights. Evaluation
# while training forks that spine: every ``eval_steps`` an eval task branches
# off the current checkpoint's weights, runs the adapter forward over a held-out
# set, and emits a metric -- but it does NOT feed back into the spine. So eval
# tasks are read-only leaves hanging off the chain, which is what turns the
# training graph into a genuine DAG.
#
# Two analogies to the inference graph make the nesting concrete:
#   * The per-step "emitted token" becomes the **checkpoint digest** (the weight
#     state on the edge). We only materialize/digest weights at eval points, so
#     ``checkpoint_digest`` is populated on the train steps that an eval forks off.
#   * An eval task **is** an inference forward pass over the eval set, so it
#     expands into the inference task graph built by ``build_task_graph`` above
#     (stored on the node as ``eval_graph``).


def train_step_flops(dims: ModelDims, batch_size: int, seq_len: int) -> int:
    """FLOPs for one optimizer step over a ``batch_size`` x ``seq_len`` batch.

    Training is ~3x a forward pass (forward + ~2x backward -- the "6*n_params per
    token" rule), so we reuse ``forward_flops`` over the batch and triple it.
    """
    fwd = forward_flops(dims, tokens_in_pass=batch_size * seq_len, context_len=seq_len)
    return 3 * fwd


@dataclass
class EvalPoint:
    """One evaluation taken mid-training, forking off the spine.

    ``step`` is the number of completed training steps when the eval ran (1-based;
    it forks off spine node index ``step - 1``). ``sample_prompt``/``sample_output``
    are one representative eval example used to expand this eval into a nested
    inference graph; leave them empty to record the eval node without nesting.
    """
    step: int
    metric: float
    checkpoint_digest: str
    sample_prompt: str = ""
    sample_output: str = ""


@dataclass
class TrainNode:
    id: int                              # unique across train + eval nodes
    kind: str                            # "train_step" | "eval"
    flops: int
    step: int                            # training step index this node sits at
    next: Optional[int]                  # spine successor (train_step only)
    branches: list[int] = field(default_factory=list)  # eval node ids forked here
    # train_step-only:
    loss: Optional[float] = None
    checkpoint_digest: Optional[str] = None  # weights emitted on the edge (eval steps)
    # eval-only:
    eval_metric: Optional[float] = None
    eval_graph: Optional[dict] = None    # nested inference TaskGraph.to_dict()


@dataclass
class TrainingTaskGraph:
    request_id: int
    model_source: str
    nodes: list[TrainNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_source": self.model_source,
            "nodes": [asdict(n) for n in self.nodes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def build_training_task_graph(
    request_id: int,
    model_source: str,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    loss_trajectory: list[float],
    evals: list[EvalPoint],
) -> TrainingTaskGraph:
    """Transform a LoRA training run (+ mid-training evals) into a branching DAG.

    The spine is ``max_steps`` ``train_step`` nodes chained via ``next``. Each
    ``EvalPoint`` adds an ``eval`` node that forks off the spine node at its
    checkpoint (``branches``) and carries a nested inference graph + metric.
    """
    dims = dims_for(model_source)
    step_flops = train_step_flops(dims, batch_size, seq_len)

    nodes: list[TrainNode] = []

    # --- the training spine ---
    for s in range(max_steps):
        loss = loss_trajectory[s] if s < len(loss_trajectory) else None
        nodes.append(TrainNode(
            id=s,
            kind="train_step",
            flops=step_flops,
            step=s,
            next=(s + 1) if s + 1 < max_steps else None,
            loss=loss,
        ))

    # --- eval branches ---
    next_id = max_steps
    for ev in evals:
        spine_idx = ev.step - 1
        if not (0 <= spine_idx < max_steps):
            raise ValueError(f"eval step {ev.step} out of range 1..{max_steps}")

        # The eval expands into the inference graph over a representative sample.
        eval_graph = None
        eval_flops = 0
        if ev.sample_prompt or ev.sample_output:
            sub = build_task_graph(
                request_id=next_id,
                prompt=ev.sample_prompt,
                output=ev.sample_output,
                model_source=model_source,
            )
            eval_graph = sub.to_dict()
            eval_flops = sum(t.flops for t in sub.tasks)  # eval is forward-only

        eval_node = TrainNode(
            id=next_id,
            kind="eval",
            flops=eval_flops,
            step=ev.step,
            next=None,
            eval_metric=ev.metric,
            eval_graph=eval_graph,
        )
        nodes.append(eval_node)

        # Materialize the checkpoint weights on the spine node the eval forks off.
        spine = nodes[spine_idx]
        spine.branches.append(next_id)
        spine.checkpoint_digest = ev.checkpoint_digest

        next_id += 1

    return TrainingTaskGraph(
        request_id=request_id,
        model_source=model_source,
        nodes=nodes,
    )


# ===========================================================================
# Speculative-decoding task graph: a spine with pruned (rejected) branches.
# ===========================================================================
#
# Speculative decoding pairs a small fast *draft* model with a large *target*
# model. Each round:
#   1. the draft model autoregressively proposes K tokens (K thin draft passes),
#   2. the target model verifies all K in ONE forward pass (one fat pass),
#   3. it accepts the longest prefix of drafts that match its own greedy choice,
#      then emits one correction/bonus token. Drafts past the first mismatch are
#      REJECTED and discarded.
#
# So the graph has two firsts vs. the inference/training graphs:
#   * **pruning** -- rejected drafts are dead-end stubs that never join the spine
#     (their ``next`` is None and nothing points to them), and
#   * **two FLOP weight-classes from two models** -- a draft pass (~2*N_draft)
#     is far cheaper than a verify pass (~2*N_target*(K+1)).
#
# The committed spine threads through the accepted drafts and the per-round
# correction tokens. Greedy spec-decode is output-identical to plain greedy
# target decoding, so the spine spells out exactly the target's own output.


@dataclass
class SpecNode:
    id: int
    kind: str                # "draft" | "verify"
    model: str               # "draft" | "target"
    flops: int
    round: int               # which spec-decode round
    pos_in_round: int        # draft index 0..K-1; the verify node is K
    token: str               # proposed token (draft) or correction token (verify)
    status: str              # draft: "accepted"|"rejected" ; verify: "correction"


@dataclass
class SpecEdge:
    src: int
    dst: int
    kind: str                # "draft" | "verify_in" | "commit"


@dataclass
class SpecDecodeTaskGraph:
    request_id: int
    draft_model: str
    target_model: str
    nodes: list[SpecNode] = field(default_factory=list)
    edges: list[SpecEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "draft_model": self.draft_model,
            "target_model": self.target_model,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def build_spec_decode_task_graph(
    request_id: int,
    draft_model_source: str,
    target_model_source: str,
    prompt_len: int,
    rounds: list[dict],
) -> SpecDecodeTaskGraph:
    """Transform a speculative-decoding run into its dependency DAG.

    ``rounds`` is a list of ``{"drafts": [str], "num_accepted": int,
    "correction": str}`` -- exactly what the spec-decode runner records. Each
    round is a chain of K ``draft`` nodes (the draft model's autoregression) that
    **all fan in** to one ``verify`` node -- the single target forward pass
    ingests every draft, accepted or rejected, because it scores them in parallel
    and doesn't know where the rejection is until after. Three edge kinds:

      * ``draft``     -- d_i -> d_{i+1}, the draft model's own autoregression.
      * ``verify_in`` -- d_i -> verify, every draft feeds the verify pass (this
        is why a rejected draft is still a real dependency, not a detached stub).
      * ``commit``    -- verify_r -> d_0 of round r+1: the verified output (kept
        prefix + correction) is the context the next round drafts from.

    Drafts past the first rejection are ingested by ``verify`` but, by causal
    masking, contribute nothing to the committed output -- ``status`` carries
    that accepted/rejected distinction for rendering.
    """
    d_dims = dims_for(draft_model_source)
    t_dims = dims_for(target_model_source)

    nodes: list[SpecNode] = []
    edges: list[SpecEdge] = []
    nid = 0
    ctx = prompt_len           # committed context length entering this round
    prev_verify: Optional[int] = None

    for r, rd in enumerate(rounds):
        drafts = rd["drafts"]
        a = rd["num_accepted"]
        correction = rd["correction"]
        k = len(drafts)

        draft_ids: list[int] = []
        for i, tok in enumerate(drafts):
            nodes.append(SpecNode(
                id=nid, kind="draft", model="draft",
                flops=forward_flops(d_dims, tokens_in_pass=1, context_len=ctx + i),
                round=r, pos_in_round=i, token=tok,
                status="accepted" if i < a else "rejected",
            ))
            draft_ids.append(nid)
            nid += 1

        # draft autoregression: d_i -> d_{i+1}
        for i in range(k - 1):
            edges.append(SpecEdge(src=draft_ids[i], dst=draft_ids[i + 1], kind="draft"))

        verify_id = nid
        nodes.append(SpecNode(
            id=verify_id, kind="verify", model="target",
            # one parallel pass over the K drafted positions (+ the correction).
            flops=forward_flops(t_dims, tokens_in_pass=k + 1, context_len=ctx + k),
            round=r, pos_in_round=k, token=correction,
            status="correction",
        ))
        nid += 1

        # fan-in: EVERY draft feeds the verify pass (accepted or rejected).
        for did in draft_ids:
            edges.append(SpecEdge(src=did, dst=verify_id, kind="verify_in"))

        # round continuation: previous verify -> this round's first node.
        if prev_verify is not None:
            edges.append(SpecEdge(src=prev_verify,
                                  dst=(draft_ids[0] if draft_ids else verify_id),
                                  kind="commit"))
        prev_verify = verify_id
        ctx += a + 1

    return SpecDecodeTaskGraph(
        request_id=request_id,
        draft_model=draft_model_source,
        target_model=target_model_source,
        nodes=nodes,
        edges=edges,
    )
