# DESIGN.md — Engineering reasoning and decisions

This document explains **how I approached the Home Assignment**, how I decomposed the requirements, the trade-offs I made, and how each required (and bonus) feature was implemented.

---

## 1) Goals and constraints (from the assignment)

The CLI must:

1. Run a **local** LLM (no remote inference).
2. Support **tool/function calling** with an execution loop:
   - model requests tools
   - CLI executes tools
   - tool results are returned to the model
   - repeat until the model produces a normal assistant message
3. Provide **deterministic, offline tools**:
   - `read_file(path) -> string`
   - `write_file(path, content) -> string`
4. Enforce a **workspace sandbox** (path traversal must be rejected).
5. Provide **logging/observability** during chat.
6. Support **3 required scenarios** (summarize+write, refine+write, path traversal rejection).

Optional bonuses:
- Streaming output
- `--session` history persistence
- Unit tests
- Packaging

---

## 2) High-level design

I separated the system into four modules:

- **CLI** (`src/localchat/cli.py`)
  - Parses flags, sets defaults, calls the chat loop.
- **Model loader / backend detection** (`src/localchat/model.py`)
  - Loads GGUF locally via llama.cpp and prints runtime info.
  - If `--model` is a Hugging Face repo id, resolves a GGUF file automatically.
- **Chat runtime** (`src/localchat/chat.py`)
  - Maintains history, calls the model, detects tool calls, executes tools, and streams output.
  - Handles `clear`, `quit/exit`, and session persistence.
- **Tools + sandbox** (`src/localchat/tools.py`)
  - Implements deterministic file operations with strict workspace boundary checks.

This separation makes it easy to unit-test the tricky parts (sandbox/tool logic, tool-call loop, model selection).

---

## 3) Why I selected `google/gemma-3-4b-it-qat-q4_0-gguf` for validation

For the Azure A100 validation run, I chose:

**`google/gemma-3-4b-it-qat-q4_0-gguf`**

### The selection process (what I was optimizing for)
In this assignment, the goal is not “the strongest model,” but a model that makes it easy to *reliably demonstrate*:

- multi-step instruction following (read → summarize → write)
- tool calling behavior
- fast interactive iteration in a CLI loop
- easy reproducibility on a clean Ubuntu VM

Given that, I intentionally picked a model that is:

- available publicly on Hugging Face as **GGUF**
- small enough to download quickly and run smoothly
- instruction-tuned so the scenario prompts work without prompt-engineering

### Why this specific Gemma GGUF variant worked well
1. **Official + widely available**
   - Published under the `google/` namespace on Hugging Face (practical reliability for an assignment submission).
2. **Instruction-tuned (“-it”)**
   - Better at following the scenario prompts end-to-end without extra scaffolding.
3. **Quantized Q4_0 (QAT) = 2.94GB**
   - The `q4_0` GGUF is compact, which made the end-to-end validation on Azure simpler (download/start/run).
4. **Great fit for an A100 verification run**
   - On the A100, it runs very fast, so the interactive experience is smooth and the logs are easy to capture.
5. **Matches the implementation behavior**
   - When a Hugging Face repo id is provided, LocalChat auto-selects a compatible GGUF file (in this case, `gemma-3-4b-it-q4_0.gguf`), so the run is reproducible.

Important note: the **implementation is model-agnostic**; any GGUF model compatible with llama.cpp should work.


## 4) Tool calling loop design

### Requirements
- The model must be able to request a tool call.
- The CLI must execute the call and feed results back into the model.
- Repeat until the model returns a normal assistant response.

### Implementation approach
In `chat.py`:

1. Send the current conversation history to the model.
2. If the model response includes one or more tool calls:
   - Validate tool name + JSON arguments
   - Execute the tool
   - Append a `tool` role message containing the tool output
   - Call the model again with the updated history
3. If the model returns a normal assistant message:
   - Print it (streamed if possible)
   - Append it to history
   - End the turn

### Failure modes handled
- Unknown tools → clear error returned to the model as tool output
- Bad JSON arguments → clear error returned
- Tool exceptions → do not crash; return error string
- “Model did not stream” even though stream=True → fallback behavior prints full text

---

## 5) Workspace sandbox and security reasoning

The assignment explicitly requires **rejecting path traversal** like `../secrets.txt`.

In `tools.py`, I enforce:

- All file paths must resolve **within the workspace directory**
- `..`, absolute paths, and symlink tricks are rejected by:
  - Resolving the candidate path to an absolute, normalized path
  - Verifying it is a child of the resolved workspace root
- For writes:
  - Create parent directories as needed (inside workspace)
  - Write using UTF-8
  - Return a deterministic confirmation message

This guarantees scenario 3 is safe and deterministic.

---

## 6) Backend reporting and graceful CPU fallback

- LocalChat reports the detected backend (cpu/cuda/metal) at startup.
- If a GPU backend is not available, it runs on CPU without failing.

## 6) Logging / observability

The CLI prints:

- A startup banner:
  - backend (cpu/cuda/metal)
  - repo id (if applicable)
  - resolved model file path
  - model size / context / GPU layers (best-effort)
- During chat:
  - tool calls and tool results
  - a rough “tokens / tok/s / seconds” footer per assistant turn

The goal is not perfect accounting, but enough signal to debug and demonstrate performance.

---

## 7) Bonus features

### Streaming output
- The chat runtime attempts true backend streaming (token-by-token).
- If streaming is not supported, the code falls back to printing the full text.

### `--session` persistence
- History is saved to a JSON file after each assistant turn.
- Loading is attempted on startup if the file exists.
- Writes are atomic (write temp file then rename) to reduce corruption risk.

### Unit tests
Tests cover:
- Tool sandbox behavior
- Tool-call loop logic
- Model selection / runtime info
- CLI argument handling

### Packaging
- `pyproject.toml` defines a console script entry point: `localchat = "localchat.cli:main"`
- `src/` layout keeps imports clean and avoids accidental relative imports.

---

## 8) What I would improve next

If this were extended beyond the assignment:

- Add more tools (list_dir, mkdir, etc.) with the same sandbox guarantees
- Add a “max tool-call iterations” guard for pathological models
- Add a structured logging mode (JSON logs) for easier automation
- Add a non-interactive mode for scripting (e.g., `--prompt-file`)
