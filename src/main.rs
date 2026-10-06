mod agent;
mod backend;
mod config;
mod exec;
mod pull;
mod tools;
mod ui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::agent::Agent;
use crate::backend::{ChatMsg, LlamaServer};
use crate::config::LynxConfig;

#[derive(Parser, Debug)]
#[command(name = "lynx", version, about = "Local-first CLI coding agent (llama.cpp + Qwen2.5-Coder abliterated)")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Workspace root (defaults to cwd).
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
    /// Backend base URL (overrides lynx.toml).
    #[arg(long, global = true)]
    backend: Option<String>,
    /// One-shot task (non-interactive). Empty = interactive TUI.
    #[arg(long, global = true)]
    task: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Start interactive chat (default).
    Chat,
    /// Download a model quant (default: Q4_K_M).
    Pull {
        /// Quant key: q4_k_m | iq4_xs | q4_0_8_8 | q4_0_4_8 | q5_k_m
        #[arg(long, default_value = "q4_k_m")]
        quant: String,
    },
    /// Start the llama-server backend (foreground).
    Serve,
    /// Show config + backend status.
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = LynxConfig::load()?;
    let workspace = cli
        .workspace
        .clone()
        .map(Ok)
        .unwrap_or_else(std::env::current_dir)?;
    let workspace = workspace.canonicalize().unwrap_or(workspace);
    let base = cli.backend.clone().unwrap_or_else(|| cfg.base_url());

    match cli.cmd.unwrap_or(Cmd::Chat) {
        Cmd::Pull { quant } => {
            let file = pull::quant_filename(&quant).ok_or_else(|| {
                anyhow::anyhow!("unknown quant `{quant}`. choices: q4_k_m iq4_xs q4_0_8_8 q4_0_4_8 q5_k_m")
            })?;
            let dest = workspace.join("models").join(file);
            pull::pull(&cfg.model.repo, file, &dest).await?;
            return Ok(());
        }
        Cmd::Status => {
            return status(&cfg, &workspace, &base).await;
        }
        Cmd::Serve => {
            return serve(&cfg, &base).await;
        }
        Cmd::Chat => {}
    }

    if let Some(task) = cli.task.clone() {
        return oneshot(&cfg, &workspace, &base, &task).await;
    }
    chat(&cfg, &workspace, &base).await
}

async fn ensure_backend(cfg: &LynxConfig, base: &str) -> Result<LlamaServer> {
    let mut srv = LlamaServer::new(base);
    if srv.probe().await.unwrap_or(false) {
        println!("{} {}", style("✓ backend live:").green(), base);
        return Ok(srv);
    }
    let model = cfg.resolve_model_path();
    if !model.exists() {
        anyhow::bail!(
            "model not found: {}\n  run: lynx pull   (or: hf download {repo} {file} --local-dir models/)\n  tip: models live in D:\\lynx\\models — run lynx from D:\\lynx or pass --workspace D:\\lynx",
            model.display(),
            repo = cfg.model.repo,
            file = cfg.model.file
        );
    }
    println!("{} {}", style("▲ starting llama-server:").cyan(), model.display());
    println!("  {}", style("loading into VRAM… (first boot takes ~30-60s)").dim());
    if cfg.model.ctx_size > 32768 && cfg.model.cache_type_k == "f16" {
        println!(
            "{}",
            style("  ⚠ large ctx without KV quantization — consider cache_type_k = \"q4_0\"").yellow()
        );
    }
    if cfg.model.cache_type_v != "f16" && !cfg.model.flash_attn {
        anyhow::bail!(
            "cache_type_v = \"{}\" requires flash_attn = true in lynx.toml",
            cfg.model.cache_type_v
        );
    }
    srv.spawn(
        &model,
        &cfg.server.host,
        cfg.server.port,
        cfg.model.ctx_size,
        cfg.model.n_gpu_layers,
        &cfg.model.cache_type_k,
        &cfg.model.cache_type_v,
        cfg.model.flash_attn,
        &cfg.server.extra_args,
    )
    .await
    .context("start llama-server")?;
    println!("{} {}", style("✓ backend ready:").green(), base);
    Ok(srv)
}

fn make_agent(cfg: &LynxConfig, workspace: &std::path::Path, srv: LlamaServer) -> Agent {
    let sandbox = crate::tools::Sandbox::new(workspace.to_path_buf(), cfg.agent.extra_roots.clone());
    let exec = crate::exec::Executor::new(
        workspace.to_path_buf(),
        sandbox,
        cfg.agent.confirm_bash,
        cfg.agent.confirm_writes,
    );
    let mut history = vec![ChatMsg::system(Agent::system_prompt(&workspace.display().to_string()))];
    // Seed a couple of workspace facts so the model doesn't guess.
    if let Ok(entries) = std::fs::read_dir(workspace) {
        let names: Vec<_> =
            entries.flatten().take(30).map(|e| e.file_name().to_string_lossy().to_string()).collect();
        history.push(ChatMsg::user(format!(
            "<workspace_files>\n{}\n</workspace_files>",
            names.join("\n")
        )));
    }
    Agent {
        server: srv,
        exec,
        temperature: cfg.model.temperature,
        top_p: cfg.model.top_p,
        n_predict: cfg.model.n_predict,
        max_steps: cfg.agent.max_steps,
        history,
    }
}

async fn chat(cfg: &LynxConfig, workspace: &std::path::Path, base: &str) -> Result<()> {
    let srv = ensure_backend(cfg, base).await?;
    let mut agent = make_agent(cfg, workspace, srv);
    ui::banner(&cfg.model.file, base);
    loop {
        let line = inquire::Text::new("›").prompt();
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // Ctrl-C
        };
        match ui::parse_line(&line) {
            ui::Slash::Task(t) if t.is_empty() => continue,
            ui::Slash::Task(t) => {
                if let Err(e) = agent.turn(&t).await {
                    println!("{} {e:#}", style("✗").red());
                }
            }
            ui::Slash::Help => ui::help(),
            ui::Slash::Tools => println!("{}", crate::tools::tool_catalog()),
            ui::Slash::Model => {
                println!("  model: {}", cfg.resolve_model_path().display());
                println!("  backend: {base}");
            }
            ui::Slash::Clear => {
                let sys = agent.history.first().cloned();
                agent.history = sys.into_iter().collect();
                println!("{}", style("(history cleared)").dim());
            }
            ui::Slash::Quit => break,
            ui::Slash::Unknown(u) => println!("{} unknown command {u} — try /help", style("?").yellow()),
        }
    }
    println!("{}", style("bye — server keeps running; `lynx serve` to reuse it.").dim());
    Ok(())
}

async fn oneshot(cfg: &LynxConfig, workspace: &std::path::Path, base: &str, task: &str) -> Result<()> {
    let srv = ensure_backend(cfg, base).await?;
    let mut agent = make_agent(cfg, workspace, srv);
    agent.turn(task).await
}

async fn serve(cfg: &LynxConfig, base: &str) -> Result<()> {
    let srv = ensure_backend(cfg, base).await?;
    println!("{} {}", style("serving at").green(), base);
    println!("{}", style("Ctrl-C to stop.").dim());
    tokio::signal::ctrl_c().await.ok();
    drop(srv);
    Ok(())
}

async fn status(cfg: &LynxConfig, workspace: &std::path::Path, base: &str) -> Result<()> {
    println!("{}", style("lynx status").bold());
    println!("  workspace: {}", workspace.display());
    let mp = cfg.resolve_model_path();
    println!(
        "  model: {} {}",
        mp.display(),
        if mp.exists() { style("(cached)").green() } else { style("(missing — run `lynx pull`)").red() }
    );
    println!(
        "  llama-server: {}",
        crate::backend::find_llama_exe()
            .map(|p| p.display().to_string())
            .unwrap_or("(not on PATH)".into())
    );
    let srv = LlamaServer::new(base);
    println!(
        "  backend {base}: {}",
        if srv.probe().await.unwrap_or(false) {
            style("live").green()
        } else {
            style("down").yellow()
        }
    );
    Ok(())
}


