# LocalChat — Local LLM CLI with Tool Calling

LocalChat is a **fully local** command-line chat application that runs a **local LLM** (GGUF via llama.cpp / llama-cpp-python), supports **tool/function calling**, and executes **offline, deterministic file tools** inside a **workspace sandbox**.

This repository is an implementation of the **“Home Assignment: Local LLM CLI with Tool Calling”** (including the optional bonuses).

---

## What each Markdown file is for

- **README.md (this file)**: How to install, run, and verify the assignment requirements. Includes **step-by-step setup** and **the 3 required scenario runs**.
- **DESIGN.md**: The engineering rationale: how the requirements were decomposed, trade-offs made, and how each requirement (and bonus) was addressed.

---

## Verified environment (reference)

This project was validated on:

- **Azure VM**: Ubuntu
- **GPU**: NVIDIA **A100 80GB PCIe**
- **Python**: in-venv (example: `.venv`)

Example GPU detection output is included below in the scenario logs.

---

## Install

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install LocalChat (editable)

```bash
pip install -e .
```

### 3) (Recommended) GPU build for llama-cpp-python

This project depends on `llama-cpp-python`. Installation options differ depending on whether you want CPU-only or CUDA.
Follow the official `llama-cpp-python` installation instructions for your environment.

If CUDA is installed correctly, LocalChat will print **Backend: cuda** on startup.

---

## Usage

### CLI flags (required)

```bash
localchat --model <path-or-hf-repo-id>           --workspace <dir>           --system <system-prompt-file>           --temperature <float>           --max-tokens <int>
```

- `--model` (required): local GGUF file path **or** a Hugging Face repo id containing GGUF(s)
- `--workspace` (optional, default: `./workspace`): sandbox directory for file tools
- `--system` (optional): load a system prompt text file
- `--temperature` (optional): sampling temperature
- `--max-tokens` (optional): max tokens per assistant turn

### REPL commands

- `quit` / `exit`: end the session
- `clear`: clear in-memory conversation history (does not delete workspace files)

---

## Assignment requirements coverage (mapping)

### 1) CLI App
- ✅ Executable `localchat` via `pyproject.toml` script entrypoint (`localchat = "localchat.cli:main"`)
- ✅ Supports `--model`, `--workspace`, `--system`, `--temperature`, `--max-tokens`
- ✅ Interactive REPL loop with conversation history
- ✅ `clear` command to reset history (in-memory)

### 2) Local inference only
- ✅ Gracefully falls back to **CPU** if GPU acceleration is unavailable
- ✅ Uses **llama.cpp** backend via `llama-cpp-python` (no remote inference calls)
- ✅ Startup banner prints:
  - detected backend (cpu / cuda / metal)
  - repo id (if using Hugging Face)
  - resolved model file path
  - model size / context / GPU layers (best-effort)

### 3) Tool calling (required)
- ✅ Model can request tools, CLI executes them, and the tool results are fed back into the model
- ✅ Loop continues until model returns a normal assistant response

### 4) Tools (offline, deterministic)
- ✅ `read_file(path: string) -> string`
- ✅ `write_file(path: string, content: string) -> string`
- ✅ **Workspace sandbox enforced** (rejects attempts to access outside the workspace)

### 5) Logging / observability
- ✅ During chat, prints:
  - tool calls and tool results
  - rough token count / token rate / response time per assistant turn (best-effort)

### Deliverables
- ✅ Code + README.md + DESIGN.md
- ✅ Works with the 3 required scenarios (see below)

---

## The 3 required scenarios (with real logs)

Below are the three scenarios required by the assignment.  
These logs were produced on an **Azure Ubuntu VM with an A100** using the model repo:

- `google/gemma-3-4b-it-qat-q4_0-gguf`

### Scenario 1 — File summarization + write report

Command:

```bash
localchat --model "google/gemma-3-4b-it-qat-q4_0-gguf" --workspace ./workspace
```

Session excerpt:

```text
Loading model...
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA A100 80GB PCIe, compute capability 8.0, VMM: yes
==================================================
LocalChat - Runtime Information
==================================================
  Backend:      cuda
  Repo ID:      google/gemma-3-4b-it-qat-q4_0-gguf
  Model:        .../gemma-3-4b-it-q4_0.gguf
  Model size:   2.94 GB
  Context:      8192 tokens
  GPU layers:   -1
==================================================
Workspace: .../workspace

You: Read input.txt, summarize it in 5 bullets, and write the bullets to report.md.

Assistant: ...
  TOOL CALL: read_file {"path": "input.txt"}
  TOOL RESULT: The Rise of Artificial Intelligence in Healthcare
  ...
  TOOL CALL: write_file {"path": "report.md", "content": "...5 bullets..."}
  TOOL RESULT: Successfully wrote ... characters to report.md
  [426 tokens, 116.6 tok/s, 3.65s]
```

Expected outcome:
- `workspace/report.md` is created and contains 5 bullet points.

---

### Scenario 2 — Iterative refinement using the written file

Session excerpt:

```text
You: Now read report.md and rewrite it to be more formal and concise. Save as report_formal.md.

Assistant: ...
  TOOL CALL: read_file {"path": "report.md"}
  TOOL RESULT: Here's a summary of the key points:
  ...
  TOOL CALL: write_file {"path": "report_formal.md", "content": "...formal summary..."}
  TOOL RESULT: Successfully wrote ... characters to report_formal.md
  [420 tokens, 144.9 tok/s, 2.90s]
```

Expected outcome:
- `workspace/report_formal.md` is created.

---

### Scenario 3 — Sandbox security / path traversal rejection

Session excerpt:

```text
You: Read ../secrets.txt

Assistant:
  TOOL CALL: read_file {"path": "../secrets.txt"}
  TOOL RESULT: Error: Access denied: '../secrets.txt' is outside the workspace directory. Only files within '.../workspace' can be accessed.
  [80 tokens, 123.9 tok/s, 0.65s]
```

Expected outcome:
- The read is **rejected safely** (no crash, clear error).

---

## Bonus features implemented

### Streaming output
- LocalChat streams tokens when the backend supports it (and falls back gracefully if not).
- The CLI prints tokens progressively (or emulates streaming if needed).

### `--session` history persistence
- Optional: `--session <path>` stores conversation history to disk and reloads it on startup.
- The session file is written **atomically** to reduce corruption risk.

Example:

```bash
localchat --model "google/gemma-3-4b-it-qat-q4_0-gguf" --workspace ./workspace --session ./my_session.json
```

### Unit tests
Run:

```bash
pytest -q
```

### Packaging
- Installable via pip (PEP 517/518)
- Entry point: `localchat`

---

## Repository layout

- `src/localchat/cli.py` — CLI argument parsing and entry point
- `src/localchat/chat.py` — chat loop, tool-calling loop, streaming, session persistence, observability prints
- `src/localchat/model.py` — model loader + backend detection + HF GGUF resolution
- `src/localchat/tools.py` — deterministic file tools + strict workspace sandbox enforcement
- `tests/` — unit tests for tools/chat/model

---

## Troubleshooting

- **“Backend: cpu” but you expected CUDA**: confirm your CUDA-enabled `llama-cpp-python` installation.
- **Model download is slow**: Hugging Face download speed depends on region; you can also pass a local GGUF path via `--model /path/to/model.gguf`.
