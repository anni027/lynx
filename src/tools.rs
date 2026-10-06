//! Tool definitions, `<tool_call>` parsing, sandbox dispatch, verifier.
//!
//! Slimmed from the Odysseus `tool_schemas.py` / `tool_execution.py` pattern:
//! deny-first path confinement, dedicated file tools (never bash redirects),
//! truncated results fed back to the model.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

pub const MAX_OUTPUT_CHARS: usize = 12_000;
pub const MAX_READ_CHARS: usize = 24_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn arg_str(&self, key: &str) -> Option<String> {
        self.arguments.get(key).and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else {
                Some(v.to_string())
            }
        })
    }
}

/// Extract tool calls from model text.
/// Primary: Qwen native `<tool_call>{"name":..,"arguments":{..}}</tool_call>`.
/// Fallback: bare `{"name":..,"arguments":{..}}` JSON object.
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut out = vec![];
    // 1. <tool_call>...</tool_call> blocks
    let re = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();
    for cap in re.captures_iter(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cap[1]) {
            if let Some(tc) = to_tool_call(&v) {
                out.push(tc);
                continue;
            }
        }
    }
    if !out.is_empty() {
        return out;
    }
    // 2. <tools> style or fenced json fallback: find all {"name": ...} objects
    for obj in find_json_objects(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&obj) {
            if let Some(tc) = to_tool_call(&v) {
                out.push(tc);
            }
        }
    }
    out
}

fn to_tool_call(v: &serde_json::Value) -> Option<ToolCall> {
    // {"name":..,"arguments":{..}}  or  {"function":{"name":..,"arguments":..}}
    if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
        let args = v.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
        let args = if let Some(s) = args.as_str() {
            serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
        } else {
            args
        };
        return Some(ToolCall { name: name.into(), arguments: args });
    }
    if let Some(f) = v.get("function") {
        return to_tool_call(f);
    }
    None
}

fn find_json_objects(text: &str) -> Vec<String> {
    // Brace-matching scan for objects containing `"name"`.
    let bytes = text.as_bytes();
    let mut objs = vec![];
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut depth = 0;
            let mut in_str = false;
            let mut esc = false;
            let start = i;
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j] as char;
                if in_str {
                    if esc {
                        esc = false;
                    } else if c == '\\' {
                        esc = true;
                    } else if c == '"' {
                        in_str = false;
                    }
                } else if c == '"' {
                    in_str = true;
                } else if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        let s = &text[start..=j];
                        if s.contains("\"name\"") {
                            objs.push(s.to_string());
                        }
                        break;
                    }
                }
                j += 1;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    objs
}

/// Tool catalogue shown to the model (kept small for 6GB VRAM context).
pub fn tool_catalog() -> &'static str {
    r#"You have these tools. Call them with <tool_call>{"name": "...", "arguments": {...}}</tool_call>.
One call per turn is fine; you may emit several in sequence across turns.

- read_file {"path": "relative or absolute file path"} — read a file (24k char cap).
- write_file {"path": "...", "content": "..."} — create or overwrite a file.
- edit_file {"path": "...", "old_string": "...", "new_string": "..."} — exact-substring replace (must appear exactly once).
- list_files {"path": ".", "max": 80} — list a directory.
- search {"pattern": "regex or text", "path": ".", "include": "*.rs"} — ripgrep-like text search.
- bash {"command": "..."} — run a shell command (cwd = workspace). Prefer dedicated file tools over redirects/heredocs/sed.
- done {"message": "..."} — finish the task with a summary for the user."#
}

/// Sensitive basenames denied everywhere (ported from Odysseus policy).
const SENSITIVE: &[&str] = &[
    ".ssh", ".gnupg", ".gitconfig", ".bashrc", ".bash_profile", ".zshrc", ".profile", ".env",
    ".netrc",
];
const SENSITIVE_FILES: &[&str] =
    &["authorized_keys", "id_rsa", "id_ed25519", "id_ecdsa", "known_hosts"];

/// Sandbox: workspace root + extra roots + system temp. Deny-first.
#[derive(Debug, Clone)]
pub struct Sandbox {
    pub roots: Vec<PathBuf>,
}

impl Sandbox {
    pub fn new(workspace: PathBuf, extra: Vec<PathBuf>) -> Self {
        let mut roots = vec![workspace];
        roots.extend(extra);
        // Always allow temp for scratch work.
        roots.push(std::env::temp_dir());
        Self { roots }
    }

    /// Resolve `p` (relative to first root) and check confinement + sensitivity.
    pub fn resolve(&self, p: &str) -> Result<PathBuf> {
        let raw = Path::new(p);
        let abs = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.roots[0].join(raw)
        };
        // Normalize `..` without touching disk.
        let mut norm = PathBuf::new();
        for comp in abs.components() {
            match comp {
                Component::CurDir => {}
                Component::ParentDir => {
                    norm.pop();
                }
                c => norm.push(c.as_os_str()),
            }
        }
        let norm_l = norm.to_string_lossy().replace('/', "\\").to_lowercase();
        for part in norm_l.split('\\') {
            if SENSITIVE.contains(&part) {
                anyhow::bail!("blocked sensitive path: {}", norm.display());
            }
        }
        if let Some(f) = norm.file_name().and_then(|f| f.to_str()) {
            let fl = f.to_lowercase();
            if SENSITIVE_FILES.iter().any(|s| fl.contains(s)) {
                anyhow::bail!("blocked sensitive file: {}", norm.display());
            }
        }
        let inside = self.roots.iter().any(|r| {
            let rl = r.to_string_lossy().replace('/', "\\").to_lowercase();
            norm_l.starts_with(&rl)
        });
        if !inside {
            anyhow::bail!(
                "path outside workspace roots: {} (roots: {:?})",
                norm.display(),
                self.roots
            );
        }
        Ok(norm)
    }
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!("{}…[truncated {} chars]", &s[..max], s.len() - max)
}

/// Strip tool-call blocks so the remaining prose can be shown to the user.
pub fn strip_tool_calls(text: &str) -> String {
    let re = Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap();
    let s = re.replace_all(text, "");
    let re2 = Regex::new("(?s)```json.*?```").unwrap();
    re2.replace_all(&s, "").trim().to_string()
}

