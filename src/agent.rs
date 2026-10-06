//! Agentic loop: system prompt -> stream -> parse tool calls -> sandbox exec.

use anyhow::Result;
use console::style;

use crate::backend::{ChatMsg, LlamaServer};
use crate::exec::Executor;
use crate::tools::{extract_tool_calls, strip_tool_calls, tool_catalog};

pub struct Agent {
    pub server: LlamaServer,
    pub exec: Executor,
    pub temperature: f32,
    pub top_p: f32,
    pub n_predict: i32,
    pub max_steps: usize,
    pub history: Vec<ChatMsg>,
}

impl Agent {
    pub fn system_prompt(workspace: &str) -> String {
        format!(
            "You are Lynx, a local-first coding agent running on the user's own machine (offline, private).\n\
             Workspace root: {workspace}\n\
             Rules: be concise; use tools to inspect before claiming; never invent file contents.\n\
             Prefer read_file/list_files/search over bash for inspection.\n\
             When done, call done with a short summary.\n\n{cat}",
            workspace = workspace,
            cat = tool_catalog()
        )
    }

    /// Run one user turn: up to `max_steps` model->tool iterations.
    /// Returns true if the session should continue.
    pub async fn turn(&mut self, user_text: &str) -> Result<()> {
        self.history.push(ChatMsg::user(user_text));
        for _step in 0..self.max_steps {
            print!("{} ", style("lynx").cyan().bold());
            use std::io::Write as _;
            std::io::stdout().flush().ok();
            let mut first = true;
            let full = self
                .server
                .chat_stream(
                    &self.history,
                    self.temperature,
                    self.top_p,
                    self.n_predict,
                    |tok| {
                        if first {
                            first = false;
                        }
                        print!("{tok}");
                        use std::io::Write as _;
                        std::io::stdout().flush().ok();
                    },
                )
                .await?;
            println!();
            self.history.push(ChatMsg::assistant(full.clone()));

            let calls = extract_tool_calls(&full);
            let prose = strip_tool_calls(&full);
            if calls.is_empty() {
                // Pure answer, no tools — end turn.
                if prose.is_empty() {
                    println!("{}", style("(empty response — try rephrasing)").dim());
                }
                return Ok(());
            }
            let mut done_msg: Option<String> = None;
            for tc in &calls {
                println!(
                    "  {} {} {}",
                    style("⚙").yellow(),
                    style(&tc.name).green().bold(),
                    style(truncate_args(tc)).dim()
                );
                let result = self.exec.run(tc).await;
                if tc.name == "done" {
                    done_msg = Some(result.clone());
                }
                println!("  {} {}", style("→").dim(), truncate_fb(&result));
                self.history.push(ChatMsg::user(format!(
                    "<tool_response name=\"{}\">\n{}\n</tool_response>",
                    tc.name, result
                )));
                // Keep context bounded.
                if self.history.len() > 40 {
                    self.history = compact_history(&self.history);
                }
            }
            if let Some(m) = done_msg {
                println!("{} {}", style("✓").green().bold(), m);
                return Ok(());
            }
        }
        println!("{}", style(format!("(stopped after {} steps — ask a follow-up to continue)", self.max_steps)).yellow());
        Ok(())
    }
}

fn truncate_args(tc: &crate::tools::ToolCall) -> String {
    let s = tc.arguments.to_string();
    if s.len() > 160 {
        format!("{}…", &s[..160])
    } else {
        s
    }
}

fn truncate_fb(s: &str) -> String {
    let first: String = s.lines().take(6).collect::<Vec<_>>().join("\n");
    if first.len() > 600 {
        format!("{}…", &first[..600])
    } else {
        first
    }
}

fn compact_history(h: &[ChatMsg]) -> Vec<ChatMsg> {
    // Keep system + first user + last 30 messages.
    let mut out = vec![];
    if let Some(first) = h.first() {
        out.push(first.clone());
    }
    let tail = h.len().saturating_sub(30);
    out.extend(h[tail..].iter().cloned());
    out
}
