//! llama.cpp backend driver: supervises local `llama-server` and talks to its
//! OpenAI-compatible endpoints (`/health`, `/v1/chat/completions` SSE).

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

impl ChatMsg {
    pub fn system(s: impl Into<String>) -> Self {
        Self { role: "system".into(), content: s.into() }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self { role: "user".into(), content: s.into() }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: s.into() }
    }
}

pub struct LlamaServer {
    child: Option<Child>,
    base: String,
    client: reqwest::Client,
}

impl LlamaServer {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            child: None,
            base: base.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .expect("reqwest client"),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// Probe `/health`. Ok(true) = ready, Ok(false) = not up yet.
    pub async fn probe(&self) -> Result<bool> {
        let url = format!("{}/health", self.base);
        match self.client.get(&url).timeout(Duration::from_secs(3)).send().await {
            Ok(r) => Ok(r.status().is_success()),
            Err(e) if e.is_timeout() || e.is_connect() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn wait_ready(&self, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();
        loop {
            if self.probe().await.unwrap_or(false) {
                return Ok(());
            }
            if start.elapsed() > timeout {
                anyhow::bail!("llama-server not ready at {} after {:?}", self.base, timeout);
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Spawn `llama-server -m <gguf> ...`. Reuses an already-live server.
    /// `llama.exe serve` (new unified CLI) is accepted too.
    pub async fn spawn(
        &mut self,
        model_path: &std::path::Path,
        host: &str,
        port: u16,
        ctx: u32,
        ngl: u32,
        cache_type_k: &str,
        cache_type_v: &str,
        flash_attn: bool,
        extra: &[String],
    ) -> Result<()> {
        if self.probe().await.unwrap_or(false) {
            return Ok(());
        }
        let exe = find_llama_exe().ok_or_else(|| {
            anyhow!("llama-server not on PATH. Install: irm https://llama.app/install.ps1 | iex")
        })?;
        let unified = exe
            .file_name()
            .map(|s| s == "llama.exe" || s == "llama")
            .unwrap_or(false);
        let mut cmd = Command::new(&exe);
        if unified {
            cmd.arg("serve");
        }
        cmd.arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg(ctx.to_string())
            .arg("-ngl")
            .arg(ngl.to_string())
            .arg("--jinja")
            .arg("-ctk")
            .arg(cache_type_k)
            .arg("-ctv")
            .arg(cache_type_v);
        if flash_attn {
            cmd.arg("-fa").arg("on");
        }
        cmd.args(extra)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let child = cmd.spawn().with_context(|| format!("spawn {}", exe.display()))?;
        self.child = Some(child);
        self.wait_ready(Duration::from_secs(240)).await?;
        Ok(())
    }

    /// Streaming chat completion. Calls `on_token` per delta; returns full text.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMsg],
        temperature: f32,
        top_p: f32,
        n_predict: i32,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let body = serde_json::json!({
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": n_predict,
            "stream": true,
            "cache_prompt": true,
        });
        let url = format!("{}/v1/chat/completions", self.base);
        let resp =
            self.client.post(&url).json(&body).send().await.with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("llama-server {code}: {text}");
        }
        let mut full = String::new();
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();
                if !line.starts_with("data:") {
                    continue;
                }
                let data = line.strip_prefix("data:").unwrap_or(&line).trim();
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    let delta = v
                        .pointer("/choices/0/delta/content")
                        .or_else(|| v.pointer("/choices/0/message/content"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    if !delta.is_empty() {
                        full.push_str(delta);
                        on_token(delta);
                    }
                }
            }
        }
        Ok(full)
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.start_kill();
        }
    }
}

pub fn find_llama_exe() -> Option<std::path::PathBuf> {
    for name in ["llama-server", "llama-server.exe", "llama.exe", "llama"] {
        if let Ok(paths) = which_paths(name) {
            if let Some(p) = paths.into_iter().next() {
                return Some(p);
            }
        }
    }
    None
}

fn which_paths(name: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut out = vec![];
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let c = dir.join(name);
            if c.exists() {
                out.push(c);
            }
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let wg = std::path::Path::new(&local).join("Microsoft").join("WinGet").join("Packages");
        if wg.exists() {
            if let Ok(entries) = std::fs::read_dir(&wg) {
                for e in entries.flatten() {
                    let c = e.path().join("llama.exe");
                    if c.exists() {
                        out.push(c);
                    }
                    let c2 = e.path().join("llama-server.exe");
                    if c2.exists() {
                        out.push(c2);
                    }
                }
            }
        }
    }
    Ok(out)
}

