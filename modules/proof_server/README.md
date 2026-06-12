# proof_server — developer-controlled proof server

**Purpose.** Turn a public scrubbed compute ledger + the proof server's own
signed Merkle root into a zero-knowledge proof that *every published row is
backed by a valid signed Merkle leaf*. The auditor reads only the proof's
public outputs (auditor nonce, ledger digest, signer-pubkey-set digest);
the raw signature stays inside the developer trust boundary.

**v0 statement.** Network-traffic consistency, soundness direction
(`ledger ⊆ signed-Merkle-log`). See `demos/proof-server/plan.md` for the
demo design.

**Second statement — bounded-cost partition of a hidden graph.** Given a
task graph (one node per forward pass, exact per-node FLOPs + input tokens +
whitelist flags — the graphs `demos/proof-compare/build_all.py` emits), the
`sp1/partition-program` guest proves: *the graph committed to by x has a
partition into stages such that every dependency edge flows forward (the
stages are executable in order), each stage's summed FLOPs are ≤ C, and each
stage's summed non-whitelisted input tokens are ≤ S*. The graph AND the
partition are private witnesses; public outputs are only (nonce, x, C, S,
n_parts) — deliberately not even the node count. x is a **blinded**
commitment sha256(`taskgraph-partition-v1` canonical encoding ‖ 32-byte
random blind), computed **in-circuit** (encoding byte-identical between
`partition.py` and `proof_server_lib`): binding because the guest re-derives
x from the very numbers it budget-checks, hiding because of the blind (a
bare content hash would let an auditor confirm guesses about a low-entropy
graph). The prover publishes x out-of-band (e.g. inside a signed envelope);
the auditor compares it against the proof's public outputs and never sees
the graph. Whitelisted inputs (publicly-known constants, e.g. the coding
agent's fixed task statement) cost 0 toward S — the whitelist never
discounts FLOPs. Driver:
`python3 demos/proof-compare/prove_partition.py --scene coding
--cap-flops 5e13 --cap-input 2000 [--prove]`.

**Wire framing.** The proof server is a proxy that the existing
`demos/tap-protocol/servers/tap.py` (and friends) forward a copy of every
`SignedEnvelope` to. The proof server holds the only Ed25519 keypair; the
"ledger row" the auditor sees is the published `EnvelopeData` payload from
each envelope. No new wire schema; no per-gateway key provisioning in v0.

**Interface.**

```python
from modules.proof_server.api import (
    build_merkle_tree, leaf_hash,
    ledger_digest, pubkey_set_digest,
    keypair_from_seed, sign_root, verify_root,
    assemble_witness,
)
```

**Artifacts.** Consumes published rows + one Ed25519 attestation; produces
an SP1 public-outputs JSON blob plus (in `--prove` mode) a proof binary.

**Requirements.** Python: `cryptography` (Ed25519). SP1 path requires
`cargo-prove` on PATH; install via `curl -L https://sp1.succinct.xyz | bash &&
sp1up`. The SP1 toolchain is **not** in the Nix runtime image — proving
happens off the GPU path.

**Layout.**

```
modules/proof_server/
├── api.py             stable Python facade
├── ledger.py          binary SHA-256 Merkle tree over arbitrary JSON-dict rows
├── envelopes.py       Ed25519 helpers (keypair / sign / verify)
├── partition.py       bounded-cost partition: planner, checker, canonical encoding
└── sp1/               Rust workspace: lib (shared types) +
                       program/host (ledger statement) +
                       partition-program/partition-host (partition statement)
```

**Status.** Research prototype. v0 proves soundness over the published
ledger rows under one signer (the proof server itself). Multi-signer
aggregation, payload-hash binding, completeness, bandwidth, correctness,
graph shallowness, and PoSE/erasure are explicitly out of scope and
tracked as follow-up statements that layer on the same commitment scheme.
