//! Tool executor: runs validated ToolCalls inside the Sandbox.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::tools::{truncate, ToolCall, MAX_OUTPUT_CHARS, MAX_READ_CHARS, Sandbox};

pub struct Executor {
    pub sandbox: Sandbox,
    pub workspace: PathBuf,
    pub confirm_bash: bool,
    pub confirm_writes: bool,
}

impl Executor {
    pub fn new(workspace: PathBuf, sandbox: Sandbox, confirm_bash: bool, confirm_writes: bool) -> Self {
        Self { sandbox, workspace, confirm_bash, confirm_writes }
    }

    fn confirm(&self, kind: &str, detail: &str, auto: bool) -> Result<()> {
        if auto {
            return Ok(());
        }
        let prompt = format!("Allow {kind}?\n  {detail}\n[y/a/N]: ");
        let ans = inquire::Text::new(&prompt).prompt().unwrap_or_default();
        let a = ans.trim().to_lowercase();
        if a == "y" || a == "yes" || a == "a" || a == "always" {
            Ok(())
        } else {
            anyhow::bail!("denied by user")
        }
    }

    pub async fn run(&self, tc: &ToolCall) -> String {
        match self.run_inner(tc).await {
            Ok(s) => truncate(&s, MAX_OUTPUT_CHARS),
            Err(e) => format!("ERROR: {e:#}"),
        }
    }

    async fn run_inner(&self, tc: &ToolCall) -> Result<String> {
        match tc.name.as_str() {
            "read_file" => {
                let p = tc.arg_str("path").ok_or_else(|| anyhow::anyhow!("read_file: missing path"))?;
                let full = self.sandbox.resolve(&p)?;
                let text = tokio::fs::read_to_string(&full)
                    .await
                    .with_context(|| format!("read {}", full.display()))?;
                Ok(truncate(&text, MAX_READ_CHARS))
            }
            "write_file" => {
                let p = tc.arg_str("path").ok_or_else(|| anyhow::anyhow!("write_file: missing path"))?;
                let content = tc.arg_str("content").unwrap_or_default();
                let full = self.sandbox.resolve(&p)?;
                self.confirm("write file", &full.display().to_string(), !self.confirm_writes)?;
                if let Some(parent) = full.parent() {
                    tokio::fs::create_dir_all(parent).await.ok();
                }
                tokio::fs::write(&full, content).await?;
                Ok(format!("wrote {}", full.display()))
            }
            "edit_file" => {
                let p = tc.arg_str("path").ok_or_else(|| anyhow::anyhow!("edit_file: missing path"))?;
                let old = tc.arg_str("old_string").ok_or_else(|| anyhow::anyhow!("edit_file: missing old_string"))?;
                let new = tc.arg_str("new_string").unwrap_or_default();
                let full = self.sandbox.resolve(&p)?;
                self.confirm("edit file", &full.display().to_string(), !self.confirm_writes)?;
                let text = tokio::fs::read_to_string(&full).await?;
                let count = text.matches(&old).count();
                if count != 1 {
                    anyhow::bail!("old_string appears {count} times (need exactly 1)");
                }
                tokio::fs::write(&full, text.replacen(&old, &new, 1)).await?;
                Ok(format!("edited {}", full.display()))
            }
            "list_files" => {
                let p = tc.arg_str("path").unwrap_or_else(|| ".".into());
                let max: usize =
                    tc.arg_str("max").and_then(|m| m.parse().ok()).unwrap_or(80);
                let full = self.sandbox.resolve(&p)?;
                let mut rd = tokio::fs::read_dir(&full).await?;
                let mut names = vec![];
                while let Ok(Some(e)) = rd.next_entry().await {
                    names.push(e.file_name().to_string_lossy().to_string());
                    if names.len() >= max {
                        break;
                    }
                }
                names.sort();
                Ok(names.join("\n"))
            }
            "search" => self.run_search(tc).await,
            "bash" => self.run_bash(tc).await,
            "done" => Ok(format!("DONE: {}", tc.arg_str("message").unwrap_or_default())),
            other => anyhow::bail!("unknown tool: {other}"),
        }
    }

    async fn run_search(&self, tc: &ToolCall) -> Result<String> {
        let pattern =
            tc.arg_str("pattern").ok_or_else(|| anyhow::anyhow!("search: missing pattern"))?;
        let path = tc.arg_str("path").unwrap_or_else(|| ".".into());
        let include = tc.arg_str("include").unwrap_or_else(|| "*".into());
        let full = self.sandbox.resolve(&path)?;
        let pat = regex::Regex::new(&pattern)
            .or_else(|_| regex::Regex::new(&regex::escape(&pattern)))?;
        let mut hits = vec![];
        for entry in
            ignore::WalkBuilder::new(&full).hidden(false).git_ignore(true).build().flatten()
        {
            if hits.len() >= 40 {
                break;
            }
            let fp = entry.path();
            if !fp.is_file() {
                continue;
            }
            if include != "*" {
                let ok = include
                    .split(',')
                    .any(|g| fp.to_string_lossy().ends_with(g.trim().replace('*', "").as_str()));
                if !ok {
                    continue;
                }
            }
            if let Ok(text) = std::fs::read_to_string(fp) {
                for (i, line) in text.lines().enumerate() {
                    if pat.is_match(line) {
                        hits.push(format!(
                            "{}:{}: {}",
                            fp.strip_prefix(&self.workspace).unwrap_or(fp).display(),
                            i + 1,
                            truncate(line.trim(), 200)
                        ));
                        if hits.len() >= 40 {
                            break;
                        }
                    }
                }
            }
        }
        if hits.is_empty() {
            Ok("(no matches)".into())
        } else {
            Ok(hits.join("\n"))
        }
    }

    async fn run_bash(&self, tc: &ToolCall) -> Result<String> {
        let cmd = tc.arg_str("command").ok_or_else(|| anyhow::anyhow!("bash: missing command"))?;
        if cmd.contains("<<") {
            anyhow::bail!("use write_file/edit_file instead of shell heredocs");
        }
        self.confirm("run command", &cmd, !self.confirm_bash)?;
        #[cfg(target_os = "windows")]
        let mut c = {
            let mut c = tokio::process::Command::new("powershell.exe");
            c.args(["-NoProfile", "-NonInteractive", "-Command", &cmd]);
            c
        };
        #[cfg(not(target_os = "windows"))]
        let mut c = {
            let mut c = tokio::process::Command::new("sh");
            c.args(["-c", &cmd]);
            c
        };
        c.current_dir(&self.workspace);
        let out = tokio::time::timeout(std::time::Duration::from_secs(120), c.output()).await??;
        let mut s = String::new();
        s.push_str(&String::from_utf8_lossy(&out.stdout));
        if !out.stderr.is_empty() {
            s.push_str("\n[stderr]\n");
            s.push_str(&String::from_utf8_lossy(&out.stderr));
        }
        s.push_str(&format!("\n[exit {}]", out.status.code().unwrap_or(-1)));
        Ok(s)
    }
}

