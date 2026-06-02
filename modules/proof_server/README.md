# proof_server — developer-controlled proof server

**Purpose.** Turn a public scrubbed compute ledger + the proof server's own
signed Merkle root into a zero-knowledge proof that *every published row is
backed by a valid signed Merkle leaf*. The auditor reads only the proof's
public outputs (auditor nonce, ledger digest, signer-pubkey-set digest);
the raw signature stays inside the developer trust boundary.

**v0 statement.** Network-traffic consistency, soundness direction
(`ledger ⊆ signed-Merkle-log`). See `demos/proof-server/plan.md` for the
demo design.

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
└── sp1/               Rust SP1 program (guest) + host harness
```

**Status.** Research prototype. v0 proves soundness over the published
ledger rows under one signer (the proof server itself). Multi-signer
aggregation, payload-hash binding, completeness, bandwidth, correctness,
graph shallowness, and PoSE/erasure are explicitly out of scope and
tracked as follow-up statements that layer on the same commitment scheme.
