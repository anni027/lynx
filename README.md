# ▲ Lynx

<p align="center">
  <strong>Local-first CLI coding agent</strong><br>
  Runs <strong>Qwen2.5-Coder-7B-Instruct (abliterated)</strong> locally via <strong>llama.cpp</strong><br>
  No cloud, no API keys, no fine-tuning
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.95%2B-orange" alt="Rust">
  <img src="https://img.shields.io/badge/Windows-CUDA_13-0078D6" alt="Windows">
  <img src="https://img.shields.io/badge/GPU-RTX_4050_6GB-green" alt="GPU">
  <img src="https://img.shields.io/badge/Context-128K-yellow" alt="Context">
</p>

---

## What is Lynx?

Lynx is a **local-first coding agent** that runs entirely on your machine. It combines the power of **Qwen2.5-Coder-7B-Instruct** (an abliterated, uncensored coding model) with **llama.cpp** for efficient inference, wrapped in a terminal-based agent harness with tool use, sandboxed execution, and streaming responses.

**No cloud. No API keys. No telemetry. Your code stays on your machine.**

---

## Architecture

```
TUI (inquire) → Agent loop (max_steps=12) → llama-server SSE
  → <tool_call> parse → Sandbox verify → Executor →  → repeat
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Inference Engine** | llama.cpp (CUDA) | GGUF model inference with Flash Attention |
| **Backend** | llama-server | OpenAI-compatible SSE API |
| **Agent Loop** | Rust + Tokio | Streaming tool-call orchestration |
| **Tools** | 7 built-in tools | File ops, search, bash execution |
| **Sandbox** | Path confinement | Deny-first workspace isolation |
| **UI** | inquire + console | Interactive TUI with slash commands |

---

## Prerequisites

- **OS**: Windows 10/11 (tested on Windows 11)
- **GPU**: NVIDIA RTX 4050 6GB VRAM (CUDA 13, driver 581+)
- **Rust**: 1.95+ ([rustup](https://rustup.rs/))
- **llama.cpp**: Install via one-liner:
  ```powershell
  irm https://llama.app/install.ps1 | iex
  ```
- **Hugging Face CLI** (for model downloads):
  ```powershell
  pip install huggingface_hub[cli]
  ```

---

## Quickstart

### 1. Clone & Build

```powershell
cd D:\lynx
cargo build --release
```

### 2. Download Model

```powershell
# Default: Q4_K_M quant (~4.4GB)
.\target\release\lynx.exe pull

# Smaller quants
.\target\release\lynx.exe pull --quant iq4_xs      # ~4.1GB, faster
.\target\release\lynx.exe pull --quant q5_k_m      # ~5.0GB, better quality
```

### 3. Start Chatting

```powershell
.\target\release\lynx.exe
```

### 4. One-Shot Task

```powershell
.\target\release\lynx.exe --task "list all .rs files and summarise the project"
```

---

## Model & Quant Guide

**Model**: `bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF`

| Quant | File | Size | VRAM (8K) | Notes |
|-------|------|------|-----------|-------|
| **q4_k_m** (default) | `…-Q4_K_M.gguf` | ~4.4 GB | ~5.9 GB | Best quality/size on CUDA |
| **iq4_xs** | `…-IQ4_XS.gguf` | ~4.1 GB | ~5.6 GB | Imatrix quant, faster generation |
| **q5_k_m** | `…-Q5_K_M.gguf` | ~5.0 GB | ~6.5 GB | Better quality if VRAM allows |

> **Note**: Turbo quants (`q4_0_8_8`, `q4_0_4_8`) are ARM-optimized layouts. On NVIDIA CUDA, they provide **no VRAM savings** and are **slower** than `q4_k_m`. Use only for ARM/Mobile.

---

## 128K Context on 6GB VRAM

Lynx supports **128K context** on RTX 4050 6GB using **KV cache quantization** + **Flash Attention**.

### How It Works

Qwen2.5-Coder-7B has 28 layers × 4 KV heads (GQA), making its KV cache much smaller than standard models:

| KV Cache Type | 128K KV Size | Total VRAM | Status |
|---------------|-------------|------------|--------|
| fp16 (default) | ~3.7 GB | ~8.1 GB | ❌ OOM |
| **q8_0** | ~1.8 GB | ~6.2 GB | ⚠️ Tight |
| **q4_0** | **~0.9 GB** | **~5.3 GB** | ✅ **Fits** |

### Configuration

Edit `lynx.toml`:

```toml
[model]
ctx_size = 131072      # 128K context
cache_type_k = "q4_0"  # Quantized K cache (75% smaller)
cache_type_v = "q4_0"  # Quantized V cache (requires flash_attn)
flash_attn = true      # Enable Flash Attention
n_gpu_layers = 99      # Offload all layers to GPU
```

**Requirements**:
- llama.cpp built with `-DGGML_CUDA_FA_VARIANT=1` (Flash Attention)
- CUDA 13+ driver
- Flash Attention automatically enabled when `-ctv q4_0` is set

**Fallback**: If output quality degrades, use `q8_0` KV cache (50% smaller, near-lossless).

---

## Tools

Lynx exposes 7 tools to the model (kept minimal for 6GB VRAM context efficiency):

| Tool | Purpose | Example |
|------|---------|---------|
| `read_file` | Read file contents (24K char cap) | `read_file {"path": "src/main.rs"}` |
| `write_file` | Create/overwrite files | `write_file {"path": "test.txt", "content": "..."}` |
| `edit_file` | Exact-substring replace (must appear exactly once) | `edit_file {"path": "a.txt", "old_string": "foo", "new_string": "bar"}` |
| `list_files` | List directory contents | `list_files {"path": ".", "max": 80}` |
| `search` | Ripgrep-like text search | `search {"pattern": "fn main", "path": ".", "include": "*.rs"}` |
| `bash` | Run shell commands (cwd = workspace) | `bash {"command": "cargo test"}` |
| `done` | Finish task with summary | `done {"message": "Fixed the bug"}` |

**Sandbox Policy**:
- Allowed: workspace root + `extra_roots` + `%TEMP%`
- Denied: `.ssh`, `.gnupg`, `.gitconfig`, `.bashrc`, `.env`, `id_rsa`, etc.
- Path normalization prevents `..` escape attacks

---

## Commands

```
/help          Show help
/tools         List available tools
/model         Show model path + backend status
/clear         Clear conversation history
/quit          Exit

<plain text>   Send a task to the agent
```

### Slash Commands

| Command | Action |
|---------|--------|
| `/help` or `/h` or `/?` | Show help |
| `/tools` | List available tools |
| `/model` | Show model path + backend URL |
| `/clear` or `/c` | Clear conversation history |
| `/quit` or `/q` or `/exit` | Exit (server keeps running) |

---

## Configuration

`lynx.toml` (project root or `%APPDATA%/lynx/config.toml`):

```toml
[server]
host = "127.0.0.1"
port = 8080
# extra_args = ["--flash-attn", "on"]

[model]
path = "models/Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf"
repo = "bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF"
file = "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf"
ctx_size = 8192
n_predict = -1
temperature = 0.2
top_p = 0.9
n_gpu_layers = 99
cache_type_k = "f16"   # "f16", "q8_0", "q4_0"
cache_type_v = "f16"   # "f16", "q8_0", "q4_0"
flash_attn = false

[agent]
max_steps = 12
confirm_bash = true
confirm_writes = false
# extra_roots = ["D:/projects"]
```

---

## Troubleshooting

### CUDA Out of Memory

```powershell
# Reduce context size in lynx.toml:
ctx_size = 4096

# Or use a smaller quant:
lynx pull --quant iq4_xs
```

### llama-server Not Found

```powershell
# Install llama.cpp:
irm https://llama.app/install.ps1 | iex

# Restart PowerShell and verify:
llama-server --version
```

### Model Not Found

```powershell
# Download the model:
.\target\release\lynx.exe pull

# Or manually:
hf download bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF `
  Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf `
  --local-dir D:\lynx\models
```

### Stale HuggingFace Lock

```powershell
# Kill hanging hf process:
Get-Process hf | Stop-Process -Force

# Remove stale lock:
Remove-Item D:\lynx\models\.cache\huggingface\download\*.lock -Force
```

---

## Performance

Tested on **RTX 4050 6GB, CUDA 13, driver 581**:

| Metric | Q4_K_M (8K ctx) | Q4_K_M (128K ctx + q4_0 KV) |
|--------|----------------|-----------------------------|
| Model Load Time | ~30s | ~30s |
| Prompt Processing (1K tokens) | ~1200 t/s | ~800 t/s |
| Generation Speed | ~35-45 t/s | ~30-40 t/s |
| VRAM Usage (idle) | ~4.8 GB | ~5.2 GB |
| VRAM Usage (active) | ~5.5 GB | ~5.7 GB |

---

## Roadmap

- [ ] TurboQuant (TQ3_0) support for 256K+ context
- [ ] Multi-modal (vision) tool
- [ ] Streaming tool call execution
- [ ] Plugin system for custom tools
- [ ] Conversation export/import
- [ ] macOS (Apple Silicon) optimization

---

## License

MIT — see [LICENSE](LICENSE) for details.

Model: `bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF` — [HuggingFace](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-abliterated-GGUF)

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF inference engine
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) — base model
- [bartowski](https://huggingface.co/bartowski) — GGUF quantization
