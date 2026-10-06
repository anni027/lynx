//! `lynx pull`: download GGUF quants via `hf` CLI (or plain HTTPS fallback).

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

pub const QUANTS: &[(&str, &str)] = &[
    ("q4_k_m", "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf"),
    ("iq4_xs", "Qwen2.5-Coder-7B-Instruct-abliterated-IQ4_XS.gguf"),
    ("q4_0_8_8", "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_0_8_8.gguf"),
    ("q4_0_4_8", "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_0_4_8.gguf"),
    ("q5_k_m", "Qwen2.5-Coder-7B-Instruct-abliterated-Q5_K_M.gguf"),
];

pub fn quant_filename(quant: &str) -> Option<&'static str> {
    let q = quant.to_lowercase().replace('-', "_");
    QUANTS.iter().find(|(k, _)| *k == q).map(|(_, f)| *f)
}

pub async fn pull(repo: &str, file: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        println!("{} {}", style("✓ already cached:").green(), dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Prefer `hf` CLI (resumable, fast).
    if hf_pull(repo, file, dest).await.is_ok() {
        return Ok(());
    }
    // Fallback: direct HTTPS with progress bar.
    println!("{}", style("hf CLI failed — falling back to direct HTTPS…").yellow());
    https_pull(repo, file, dest).await
}

async fn hf_pull(repo: &str, file: &str, dest: &Path) -> Result<()> {
    let local_dir = dest.parent().unwrap_or(Path::new(".")).to_path_buf();
    let status = tokio::process::Command::new("hf")
        .args(["download", repo, file, "--local-dir"])
        .arg(&local_dir)
        .status()
        .await
        .context("run `hf download`")?;
    if !status.success() {
        anyhow::bail!("hf download exited {status}");
    }
    // `hf` may place the file at top level; move if needed.
    if !dest.exists() {
        let alt = local_dir.join(file);
        if alt.exists() && alt != dest {
            std::fs::rename(&alt, dest).ok();
        }
    }
    if dest.exists() {
        println!("{} {}", style("✓ downloaded:").green(), dest.display());
        Ok(())
    } else {
        anyhow::bail!("file not found after hf download")
    }
}

async fn https_pull(repo: &str, file: &str, dest: &Path) -> Result<()> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40}] {bytes}/{total_bytes} {bytes_per_sec}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(format!("pull {file}"));
    let mut f = tokio::fs::File::create(dest).await?;
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        pb.inc(chunk.len() as u64);
        f.write_all(&chunk).await?;
    }
    pb.finish_with_message("done");
    Ok(())
}
