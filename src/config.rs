//! Lynx configuration: `lynx.toml` next to the binary / cwd, else
//! `%APPDATA%/lynx/config.toml` (Windows) / `~/.config/lynx/config.toml`.
//!
//! The MVP talks to a local `llama-server` (llama.cpp) serving an abliterated
//! Qwen2.5-Coder-7B GGUF. No cloud, no API keys.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    8080
}
fn default_ctx() -> u32 {
    8192
}
fn default_n_predict() -> i32 {
    -1
}
fn default_temp() -> f32 {
    0.2
}
fn default_top_p() -> f32 {
    0.9
}
fn default_ngl() -> u32 {
    99
}
fn default_max_steps() -> usize {
    12
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCfg {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Extra args appended to `llama-server` when Lynx spawns it.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCfg {
    /// GGUF file. Relative paths resolve against the workspace root / exe dir.
    #[serde(default = "default_model_path")]
    pub path: PathBuf,
    /// HuggingFace repo + filename used by `lynx pull`.
    #[serde(default = "default_repo")]
    pub repo: String,
    #[serde(default = "default_file")]
    pub file: String,
    #[serde(default = "default_ctx")]
    pub ctx_size: u32,
    #[serde(default = "default_n_predict")]
    pub n_predict: i32,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Layers offloaded to GPU. 99 = all (llama.cpp clamps automatically).
    #[serde(default = "default_ngl")]
    pub n_gpu_layers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCfg {
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Ask before running shell commands.
    #[serde(default = "default_true")]
    pub confirm_bash: bool,
    /// Ask before writing / editing files.
    #[serde(default)]
    pub confirm_writes: bool,
    /// Extra workspace roots the agent may touch (beyond cwd).
    #[serde(default)]
    pub extra_roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LynxConfig {
    #[serde(default)]
    pub server: ServerCfg,
    #[serde(default)]
    pub model: ModelCfg,
    #[serde(default)]
    pub agent: AgentCfg,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            extra_args: vec![],
        }
    }
}

impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            path: default_model_path(),
            repo: default_repo(),
            file: default_file(),
            ctx_size: default_ctx(),
            n_predict: default_n_predict(),
            temperature: default_temp(),
            top_p: default_top_p(),
            n_gpu_layers: default_ngl(),
        }
    }
}

impl Default for AgentCfg {
    fn default() -> Self {
        Self {
            max_steps: default_max_steps(),
            confirm_bash: true,
            confirm_writes: false,
            extra_roots: vec![],
        }
    }
}

impl Default for LynxConfig {
    fn default() -> Self {
        Self {
            server: ServerCfg::default(),
            model: ModelCfg::default(),
            agent: AgentCfg::default(),
        }
    }
}

pub fn default_repo() -> String {
    "bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF".into()
}

/// Default quant: Q4_K_M — best quality/size tradeoff on CUDA.
/// See `lynx pull --quant` for IQ4_XS (smaller, imatrix) and Q4_0_8_8 (ARM turbo).
pub fn default_file() -> String {
    "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf".into()
}

pub fn default_model_path() -> PathBuf {
    PathBuf::from("models/Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf")
}

impl LynxConfig {
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.server.host, self.server.port)
    }

    /// Search order: `./lynx.toml` (cwd) -> exe-dir `lynx.toml` -> global config.
    pub fn load() -> Result<Self> {
        for cand in Self::candidates() {
            if cand.exists() {
                let text =
                    std::fs::read_to_string(&cand).with_context(|| format!("read {}", cand.display()))?;
                let cfg: Self =
                    toml::from_str(&text).with_context(|| format!("parse {}", cand.display()))?;
                return Ok(cfg);
            }
        }
        Ok(Self::default())
    }

    fn candidates() -> Vec<PathBuf> {
        let mut v = vec![PathBuf::from("lynx.toml")];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                v.push(dir.join("lynx.toml"));
            }
        }
        if let Some(base) = dirs::config_dir() {
            v.push(base.join("lynx").join("config.toml"));
        }
        // Repo-root fallback for dev: D:\lynx\lynx.toml
        v.push(PathBuf::from("D:\\lynx\\lynx.toml"));
        v
    }

    /// Resolve the model path against candidates (cwd, exe dir, absolute).
    pub fn resolve_model_path(&self) -> PathBuf {
        let p = &self.model.path;
        if p.is_absolute() && p.exists() {
            return p.clone();
        }
        if p.exists() {
            return p.clone();
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let c = dir.join(p);
                if c.exists() {
                    return c;
                }
            }
        }
        p.clone()
    }
}
