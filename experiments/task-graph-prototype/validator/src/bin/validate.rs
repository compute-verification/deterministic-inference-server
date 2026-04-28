//! CLI: `validate --graph <path> --replay <path>` -> verdict on stdout.
//!
//! Exits 0 on PASS, 1 on FAIL.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use task_graph_validator::{
    run_all, AttestedTaskGraph, CheckStatus, ReplayResponses, ValidatorContext,
};

#[derive(Parser, Debug)]
#[command(name = "validate")]
struct Args {
    #[arg(long)]
    graph: PathBuf,
    #[arg(long)]
    replay: PathBuf,
    /// If set, also accept Skip as a non-failing outcome (default true).
    #[arg(long, default_value_t = true)]
    accept_skip: bool,
    /// One-line summary mode for demo runner.
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let graph_bytes = std::fs::read(&args.graph)
        .with_context(|| format!("reading {}", args.graph.display()))?;
    let graph: AttestedTaskGraph = serde_json::from_slice(&graph_bytes)
        .with_context(|| format!("parsing graph {}", args.graph.display()))?;

    let replay_bytes = std::fs::read(&args.replay)
        .with_context(|| format!("reading {}", args.replay.display()))?;
    let replays: ReplayResponses = serde_json::from_slice(&replay_bytes)
        .with_context(|| format!("parsing replays {}", args.replay.display()))?;

    let ctx = ValidatorContext::new(&graph, &replays.responses);
    let results = run_all(&ctx);

    let mut overall_pass = true;
    let mut first_failure: Option<String> = None;
    for r in &results {
        match &r.status {
            CheckStatus::Pass => {}
            CheckStatus::Skip(_) if args.accept_skip => {}
            CheckStatus::Skip(msg) => {
                overall_pass = false;
                if first_failure.is_none() {
                    first_failure = Some(format!("{}: skipped not accepted ({})", r.name, msg));
                }
            }
            CheckStatus::Fail(msg) => {
                overall_pass = false;
                if first_failure.is_none() {
                    first_failure = Some(format!("{}: {}", r.name, msg));
                }
            }
        }
    }

    if args.json {
        let arr: Vec<serde_json::Value> = results.iter().map(|r| {
            let (status, message) = match &r.status {
                CheckStatus::Pass => ("pass", String::new()),
                CheckStatus::Skip(m) => ("skip", m.clone()),
                CheckStatus::Fail(m) => ("fail", m.clone()),
            };
            serde_json::json!({"check": r.name, "status": status, "message": message})
        }).collect();
        let verdict = serde_json::json!({
            "verdict": if overall_pass { "pass" } else { "fail" },
            "first_failure": first_failure,
            "checks": arr,
        });
        println!("{}", serde_json::to_string_pretty(&verdict)?);
    } else {
        for r in &results {
            let (label, msg) = match &r.status {
                CheckStatus::Pass => ("PASS", String::new()),
                CheckStatus::Skip(m) => ("SKIP", m.clone()),
                CheckStatus::Fail(m) => ("FAIL", m.clone()),
            };
            println!("  [{}] {} {}", label, r.name, msg);
        }
        println!();
        if overall_pass {
            println!("VERDICT: PASS");
        } else {
            println!("VERDICT: FAIL");
            if let Some(m) = first_failure {
                println!("  first failure: {}", m);
            }
        }
    }

    if !overall_pass {
        std::process::exit(1);
    }
    Ok(())
}
