//! Host harness for the proof-server SP1 program.
//!
//! Three modes:
//!
//!   --execute   Read a witness JSON blob from stdin, run the SP1 program in
//!               the RISC-V interpreter (no proof bytes), write the parsed
//!               public outputs to stdout as JSON. Exits non-zero if any
//!               `assert!` inside the guest fires.
//!
//!   --prove --out PATH
//!               Same input. Produce a Plonk proof, write proof bytes to
//!               PATH and the public outputs to stdout as JSON.
//!
//!   --verify --proof PATH --public PATH
//!               Read the proof bytes and a public-outputs JSON (with a
//!               `bytes_hex` field), run the SP1 verifier. Exits 0 on PASS.
//!
//! Input JSON shape (for --execute / --prove), with hex-encoded bytes:
//!
//!   {
//!     "auditor_nonce": "<64 hex>",
//!     "signer_pubkeys": ["<64 hex>", ...],
//!     "ledger_rows_canon_hex": ["<hex>", ...],
//!     "witnesses": [
//!       { "gw_idx": 0, "leaf_index": 0, "merkle_path_hex": ["<64 hex>", ...],
//!         "signed_root": "<64 hex>", "signature": "<128 hex>" }, ...
//!     ]
//!   }

use clap::Parser;
use proof_server_lib::{ProofInput, PublicOutputs, RowWitness, PUBLIC_OUTPUT_LEN};
use serde::Deserialize;
use sp1_sdk::{
    blocking::{ProveRequest, Prover, ProverClient},
    include_elf, Elf, ProvingKey, SP1Stdin,
};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

const PROGRAM_ELF: Elf = include_elf!("proof-server-program");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run the SP1 program in the RISC-V interpreter (no proof bytes).
    #[arg(long, conflicts_with_all = &["prove", "verify"])]
    execute: bool,

    /// Generate a Plonk proof.
    #[arg(long, conflicts_with_all = &["execute", "verify"])]
    prove: bool,

    /// Verify a proof previously generated with --prove.
    #[arg(long, conflicts_with_all = &["execute", "prove"])]
    verify: bool,

    /// Output path for the proof bytes (--prove) or input path for verification (--verify).
    #[arg(long)]
    proof: Option<PathBuf>,

    /// Path to a public-outputs JSON (with `bytes_hex` field) for --verify.
    #[arg(long)]
    public: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Input JSON (hex-encoded byte fields, for legibility)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WitnessJson {
    signer_idx: u32,
    leaf_index: u64,
    merkle_path_hex: Vec<String>,
    signed_root: String,
    signature: String,
}

#[derive(Debug, Deserialize)]
struct InputJson {
    auditor_nonce: String,
    signer_pubkeys: Vec<String>,
    ledger_rows_canon_hex: Vec<String>,
    witnesses: Vec<WitnessJson>,
}

fn hex_to_32(s: &str) -> [u8; 32] {
    let v = hex::decode(s).expect("invalid hex");
    assert_eq!(v.len(), 32, "expected 32 bytes, got {}", v.len());
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    out
}

fn hex_to_sig(s: &str) -> Vec<u8> {
    let v = hex::decode(s).expect("invalid hex");
    assert_eq!(v.len(), 64, "expected 64 bytes, got {}", v.len());
    v
}

fn parse_input(stdin_json: &str) -> ProofInput {
    let raw: InputJson = serde_json::from_str(stdin_json).expect("malformed input JSON");
    let auditor_nonce = hex_to_32(&raw.auditor_nonce);
    let signer_pubkeys: Vec<[u8; 32]> = raw.signer_pubkeys.iter().map(|s| hex_to_32(s)).collect();
    let ledger_rows_canon = raw
        .ledger_rows_canon_hex
        .iter()
        .map(|h| hex::decode(h).expect("invalid row hex"))
        .collect();
    let witnesses = raw
        .witnesses
        .into_iter()
        .map(|w| RowWitness {
            signer_idx: w.signer_idx,
            leaf_index: w.leaf_index,
            merkle_path: w.merkle_path_hex.iter().map(|h| hex_to_32(h)).collect(),
            signed_root: hex_to_32(&w.signed_root),
            signature: hex_to_sig(&w.signature),
        })
        .collect();
    ProofInput { auditor_nonce, signer_pubkeys, ledger_rows_canon, witnesses }
}

fn emit_public_outputs(bytes: &[u8]) -> String {
    assert_eq!(
        bytes.len(),
        PUBLIC_OUTPUT_LEN,
        "expected {PUBLIC_OUTPUT_LEN} bytes of public outputs, got {}",
        bytes.len(),
    );
    let parsed = PublicOutputs::from_bytes(bytes).expect("public outputs malformed");
    serde_json::json!({
        "bytes_hex": hex::encode(bytes),
        "auditor_nonce": hex::encode(parsed.auditor_nonce),
        "ledger_digest": format!("sha256:{}", hex::encode(parsed.ledger_digest)),
        "pubkey_set_digest": format!("sha256:{}", hex::encode(parsed.pubkey_set_digest)),
        "n_rows": parsed.n_rows,
        "n_signers": parsed.n_signers,
    })
    .to_string()
}

fn main() {
    sp1_sdk::utils::setup_logger();
    let args = Args::parse();

    if !(args.execute || args.prove || args.verify) {
        eprintln!("error: must pass one of --execute, --prove, --verify");
        std::process::exit(2);
    }

    if args.verify {
        let proof_path = args.proof.expect("--verify requires --proof PATH");
        let public_path = args.public.expect("--verify requires --public PATH");
        let proof_bytes = fs::read(&proof_path).expect("read proof");
        let public_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&public_path).expect("read public"))
                .expect("public json");
        let public_bytes_hex = public_json
            .get("bytes_hex")
            .and_then(|v| v.as_str())
            .expect("public json missing bytes_hex");
        let public_bytes = hex::decode(public_bytes_hex).expect("bytes_hex not hex");

        // Reconstruct the SP1 proof and verify.
        let proof: sp1_sdk::SP1ProofWithPublicValues =
            bincode::deserialize(&proof_bytes).expect("proof bytes are not bincode");
        // The proof's own public_values must match what the auditor was told.
        assert_eq!(
            proof.public_values.as_slice(),
            public_bytes.as_slice(),
            "proof public values disagree with the public outputs file",
        );
        let client = ProverClient::from_env();
        let pk = client.setup(PROGRAM_ELF).expect("setup");
        client
            .verify(&proof, pk.verifying_key(), None)
            .expect("SP1 verifier rejected the proof");
        println!("verify: PASS");
        return;
    }

    // Read witness JSON from stdin.
    let mut raw = String::new();
    io::stdin()
        .read_to_string(&mut raw)
        .expect("read stdin");
    let input = parse_input(&raw);

    let mut stdin = SP1Stdin::new();
    stdin.write(&input);

    let client = ProverClient::from_env();

    if args.execute {
        let (output, report) = client
            .execute(PROGRAM_ELF, stdin)
            .run()
            .expect("execute call failed at SDK level");
        let out_len = output.as_slice().len();
        eprintln!(
            "execute: cycles={} public_output_len={}",
            report.total_instruction_count(),
            out_len,
        );
        if out_len != PUBLIC_OUTPUT_LEN {
            // The guest didn't reach the final commit_slice calls, which
            // means an assert! fired inside the SP1 program. Surface that
            // as a non-zero exit rather than a parse panic so the demo
            // gets a clean error.
            eprintln!(
                "ERROR: guest produced {} public-output bytes (expected {}); \
                 an assert! inside the SP1 program fired -- typically a bad \
                 signature, mangled Merkle path, or out-of-order ledger row.",
                out_len, PUBLIC_OUTPUT_LEN,
            );
            std::process::exit(10);
        }
        let s = emit_public_outputs(output.as_slice());
        let _ = writeln!(io::stdout(), "{}", s);
    } else {
        let out_path = args.proof.expect("--prove requires --out PATH");
        let pk = client.setup(PROGRAM_ELF).expect("setup");
        let proof = client.prove(&pk, stdin).run().expect("prove failed");
        let proof_bytes = bincode::serialize(&proof).expect("serialize proof");
        fs::write(&out_path, &proof_bytes).expect("write proof");
        eprintln!("prove: wrote {} bytes to {}", proof_bytes.len(), out_path.display());
        let s = emit_public_outputs(proof.public_values.as_slice());
        let _ = writeln!(io::stdout(), "{}", s);
    }
}
