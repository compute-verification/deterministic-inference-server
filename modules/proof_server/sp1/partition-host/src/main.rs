//! Host harness for the partition SP1 program. Same three modes as the
//! proof-server host:
//!
//!   --execute   Read the partition witness JSON from stdin, run the SP1
//!               program in the RISC-V interpreter (no proof bytes), write
//!               the parsed public outputs to stdout as JSON. Exits non-zero
//!               if any `assert!` inside the guest fires.
//!
//!   --prove --proof PATH
//!               Same input. Produce a proof, write proof bytes to PATH and
//!               the public outputs to stdout as JSON.
//!
//!   --verify --proof PATH --public PATH
//!               Read the proof bytes and a public-outputs JSON (with a
//!               `bytes_hex` field), run the SP1 verifier. Exits 0 on PASS.
//!
//! Input JSON shape (for --execute / --prove); built by
//! modules.proof_server.partition.sp1_input_json:
//!
//!   {
//!     "auditor_nonce": "<64 hex>",
//!     "blind": "<64 hex>",                  // commitment blinding factor
//!     "cap_flops": <u64>, "cap_input": <u64>,
//!     "flops": [<u64>, ...], "in_size": [<u32>, ...],
//!     "whitelisted": [0|1, ...],
//!     "edges": [[src_idx, dst_idx], ...],   // strictly lex-sorted
//!     "parts": [<u32>, ...]
//!   }

use clap::Parser;
use proof_server_lib::{PartitionInput, PartitionPublicOutputs, PARTITION_PUBLIC_OUTPUT_LEN};
use serde::Deserialize;
use sp1_sdk::{
    blocking::{ProveRequest, Prover, ProverClient},
    include_elf, Elf, ProvingKey, SP1Stdin,
};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

const PROGRAM_ELF: Elf = include_elf!("partition-program");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run the SP1 program in the RISC-V interpreter (no proof bytes).
    #[arg(long, conflicts_with_all = &["prove", "verify"])]
    execute: bool,

    /// Generate a proof.
    #[arg(long, conflicts_with_all = &["execute", "verify"])]
    prove: bool,

    /// Verify a proof previously generated with --prove.
    #[arg(long, conflicts_with_all = &["execute", "prove"])]
    verify: bool,

    /// Output path for the proof bytes (--prove) or input path (--verify).
    #[arg(long)]
    proof: Option<PathBuf>,

    /// Path to a public-outputs JSON (with `bytes_hex` field) for --verify.
    #[arg(long)]
    public: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct InputJson {
    auditor_nonce: String,
    blind: String,
    cap_flops: u64,
    cap_input: u64,
    flops: Vec<u64>,
    in_size: Vec<u32>,
    whitelisted: Vec<u8>,
    edges: Vec<(u32, u32)>,
    parts: Vec<u32>,
}

fn hex_to_32(s: &str) -> [u8; 32] {
    let v = hex::decode(s).expect("invalid hex");
    assert_eq!(v.len(), 32, "expected 32 bytes, got {}", v.len());
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    out
}

fn parse_input(stdin_json: &str) -> PartitionInput {
    let raw: InputJson = serde_json::from_str(stdin_json).expect("malformed input JSON");
    PartitionInput {
        auditor_nonce: hex_to_32(&raw.auditor_nonce),
        blind: hex_to_32(&raw.blind),
        flops: raw.flops,
        in_size: raw.in_size,
        whitelisted: raw.whitelisted,
        edges: raw.edges,
        parts: raw.parts,
        cap_flops: raw.cap_flops,
        cap_input: raw.cap_input,
    }
}

fn emit_public_outputs(bytes: &[u8]) -> String {
    assert_eq!(
        bytes.len(),
        PARTITION_PUBLIC_OUTPUT_LEN,
        "expected {PARTITION_PUBLIC_OUTPUT_LEN} bytes of public outputs, got {}",
        bytes.len(),
    );
    let parsed = PartitionPublicOutputs::from_bytes(bytes).expect("public outputs malformed");
    serde_json::json!({
        "bytes_hex": hex::encode(bytes),
        "auditor_nonce": hex::encode(parsed.auditor_nonce),
        "graph_commitment": format!("sha256:{}", hex::encode(parsed.graph_commitment)),
        "cap_flops": parsed.cap_flops,
        "cap_input": parsed.cap_input,
        "n_parts": parsed.n_parts,
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

        let proof: sp1_sdk::SP1ProofWithPublicValues =
            bincode::deserialize(&proof_bytes).expect("proof bytes are not bincode");
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

    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw).expect("read stdin");
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
        if out_len != PARTITION_PUBLIC_OUTPUT_LEN {
            eprintln!(
                "ERROR: guest produced {} public-output bytes (expected {}); \
                 an assert! inside the SP1 program fired -- typically a part \
                 over budget, a backward edge between parts, or a malformed \
                 edge list.",
                out_len, PARTITION_PUBLIC_OUTPUT_LEN,
            );
            std::process::exit(10);
        }
        let s = emit_public_outputs(output.as_slice());
        let _ = writeln!(io::stdout(), "{}", s);
    } else {
        let out_path = args.proof.expect("--prove requires --proof PATH");
        let pk = client.setup(PROGRAM_ELF).expect("setup");
        let proof = client.prove(&pk, stdin).run().expect("prove failed");
        let proof_bytes = bincode::serialize(&proof).expect("serialize proof");
        fs::write(&out_path, &proof_bytes).expect("write proof");
        eprintln!("prove: wrote {} bytes to {}", proof_bytes.len(), out_path.display());
        let s = emit_public_outputs(proof.public_values.as_slice());
        let _ = writeln!(io::stdout(), "{}", s);
    }
}
