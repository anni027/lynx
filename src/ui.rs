//! Terminal UI helpers: banner, markdown-ish render, slash commands.

use console::style;

pub fn banner(model: &str, backend: &str) {
    println!(
        "{} {} {}",
        style("▲ lynx").cyan().bold(),
        style("v0.1.0").dim(),
        style("· local-first agent").dim()
    );
    println!("  {} {}", style("model:").dim(), style(model).green());
    println!("  {} {}", style("backend:").dim(), style(backend).green());
    println!(
        "  {}",
        style("commands: /help /tools /model /clear /quit — plain text = task").dim()
    );
    println!();
}

/// Minimal markdown render: headers cyan, code fences dim, bullets kept.
pub fn render_md(text: &str) {
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            println!("{}", style(line).dim());
            continue;
        }
        if in_code {
            println!("  {}", style(line).dim());
            continue;
        }
        let t = line.trim_start();
        if let Some(h) = t.strip_prefix("### ") {
            println!("{}", style(format!("▸ {h}")).cyan().bold());
        } else if let Some(h) = t.strip_prefix("## ") {
            println!("{}", style(format!("▸ {h}")).cyan().bold());
        } else if let Some(h) = t.strip_prefix("# ") {
            println!("{}", style(format!("▸ {h}")).cyan().bold());
        } else if t.starts_with("- ") || t.starts_with("* ") {
            println!("  {} {}", style("•").cyan(), &t[2..]);
        } else {
            println!("{line}");
        }
    }
}

pub fn help() {
    println!("{}", style("Lynx commands").bold());
    println!("  /help          show this help");
    println!("  /tools         list available tools");
    println!("  /model         show model path + backend");
    println!("  /clear         clear conversation history");
    println!("  /quit          exit");
    println!("  <anything>     send a task to the agent");
}

pub enum Slash {
    Help,
    Tools,
    Model,
    Clear,
    Quit,
    Unknown(String),
    Task(String),
}

pub fn parse_line(line: &str) -> Slash {
    let t = line.trim();
    if t.is_empty() {
        return Slash::Task(String::new());
    }
    if !t.starts_with('/') {
        return Slash::Task(t.to_string());
    }
    match t.split_whitespace().next().unwrap_or("") {
        "/help" | "/h" | "/?" => Slash::Help,
        "/tools" => Slash::Tools,
        "/model" => Slash::Model,
        "/clear" | "/c" => Slash::Clear,
        "/quit" | "/q" | "/exit" => Slash::Quit,
        other => Slash::Unknown(other.to_string()),
    }
}
