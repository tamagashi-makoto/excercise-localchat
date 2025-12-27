# LocalChat - Local LLM CLI with Tool Calling

A command-line application that runs a local LLM with interactive chat and tool/function calling support.

**Local Inference Only**: This application runs entirely on your machine. No external APIs are used.

---

## 🚀 Quickstart

Run these commands to get started immediately:

```bash
# 1. Setup (creates .venv and installs dependencies)
chmod +x setup.sh
./setup.sh

# 2. Activate
source .venv/bin/activate

# 3. Run (example with a HuggingFace model)
localchat --model "google/gemma-3-4b-it-qat-q4_0-gguf" --workspace ./workspace
```

---

## ✅ Requirements Checklist

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | CLI App with required flags | ✅ | `--model` (path or repo-id), `--workspace`, `--system` |
| 2 | Local Inference Only | ✅ | Uses `llama-cpp-python` for fully local inference |
| 3 | Hardware Acceleration | ✅ | Auto-detects CUDA/Metal/CPU, reports backend at startup |
| 4 | Interactive REPL | ✅ | Maintains conversation history, supports `quit`/`exit`/`clear` |
| 5 | Tool Calling Loop | ✅ | Detects → Validates → Executes → Returns result → Continues |
| 6 | `read_file` tool | ✅ | Reads files from workspace directory |
| 7 | `write_file` tool | ✅ | Writes files to workspace directory |
| 8 | Workspace Sandboxing | ✅ | Path traversal attacks blocked (Scenario 3) |
| 9 | Tool Call Logging | ✅ | Shows `TOOL CALL` and `TOOL RESULT` for each invocation |
| 10 | Performance Metrics | ✅ | Reports tokens, tok/s, and response time per turn |
| 11 | Scenario 1: Summarize + Write | ✅ | See transcript below |
| 12 | Scenario 2: Iterative Refinement | ✅ | See transcript below |
| 13 | Scenario 3: Security Rejection | ✅ | See transcript below |

---

## Features

- **Fully local inference** (no external APIs used)
- Tool calling support (read_file, write_file)
- Sandboxed file operations (path traversal protection)
- **CUDA/Metal/CPU backend auto-detection**
- Interactive REPL with session history
- Generation statistics (tokens/sec, response time)

## Requirements

- Python 3.10+
- **Hardware Acceleration** (Recommended):
    - **Apple Silicon (Metal)**: Works out of the box with default install.
    - **NVIDIA GPU (CUDA)**: Requires CUDA Toolkit 11.x or 12.x.

## Installation

### Standard Setup

```bash
git clone <repository-url>
cd localchat
chmod +x setup.sh
./setup.sh
```

### Hardware Acceleration Notes

The `setup.sh` script installs the default `llama-cpp-python`. For optimal performance on specific hardware, you may need to reinstall with specific build flags:

**Apple Silicon (Metal)**:
Usually enabled by default. If issues arise:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**NVIDIA GPU (Linear/Windows)**:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**CPU Only**:
Default behavior. No extra steps needed.

## Usage

Activate the environment first:
```bash
source .venv/bin/activate
```

### Run with a Model

You can provide either a **local file path** or a **Hugging Face Repo ID** to the `--model` flag.

**Option 1: Hugging Face Repo ID (Auto-download)**
```bash
# Defaults to looking for .gguf files in the repo
localchat --model google/gemma-3-4b-it-qat-q4_0-gguf
```
*Note: You can specify a filename if needed with `--filename`, otherwise it attempts to find a suitable GGUF.*

**Option 2: Local File Path**
```bash
localchat --model ./models/my-model.gguf
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to GGUF file OR Hugging Face Repo ID | Required |
| `--filename` | Specific filename (if using Repo ID) | Auto-detected if omitted |
| `--workspace` | Sandbox directory for file tools | `./workspace` |
| `--system` | Path to system prompt file | `None` |
| `--temperature` | Sampling temperature | `0.7` |
| `--max-tokens` | Maximum tokens to generate | `2048` |

_Legacy options `--repo-id` is supported for backward compatibility but `--model` is preferred._

### REPL Commands

- `quit` / `exit`: End the session
- `clear`: Clear conversation history

## Tested On

- **CPU**: Ubuntu 22.04 (Azure VM), Python 3.10
- **CUDA**: NVIDIA A100 (Ubuntu 22.04), CUDA 12.x
- **Metal**: macOS Sequoia (M2 Air)

---

## Scenario Transcripts

### Scenario 1: File Summarization + Write Report

Setup: workspace/input.txt contains article about AI in healthcare.

Prompt: Read input.txt, summarize it in 5 bullets, and write the bullets to report.md.

```
You: Read input.txt, summarize it in 5 bullets, and write the bullets to report.md.

A: 
  TOOL CALL: read_file {"path": "input.txt"}
  TOOL RESULT: The Rise of Artificial Intelligence in Healthcare

Artificial intelligence is transforming the healthcare industry in numerous ways.
Machine learning algorithms can now analyze medical images with acc...

  TOOL CALL: write_file {"path": "report.md", "content": "* AI is revolutionizing healthcare through applications like image analysis and diagnostic tools.\n* AI accelerates drug discovery and streamlines clinical trials.\n* Natural language processing improves analysis of medical data.\n* Personalized medicine is being enabled by AI's analysis of genetic information.\n* AI-powered tools are expanding access to mental health support and providing continuous health monitoring."}
  TOOL RESULT: Successfully wrote 418 characters to report.md
Great! The summary has been written to `report.md`. I'm done.

  [303 tokens, 17.7 tok/s, 17.10s]
```

**✅ Verified:** Assistant triggered `read_file`, produced 5-bullet summary, triggered `write_file`, and confirmed completion.

---

### Scenario 2: Iterative Refinement

Prompt: Now read report.md and rewrite it to be more formal and concise. Save as report_formal.md.

```
You: Now read report.md and rewrite it to be more formal and concise. Save as report_formal.md.

A: 
  TOOL CALL: read_file {"path": "report.md"}
  TOOL RESULT: * AI is revolutionizing healthcare through applications like image analysis and diagnostic tools.
* AI accelerates drug discovery and streamlines clinical trials.
* Natural language processing improve...

  TOOL CALL: write_file {"path": "report_formal.md", "content": "Artificial intelligence is significantly transforming the healthcare sector. Key advancements include: AI-driven image analysis and diagnostic tools; accelerated drug discovery through predictive modeling; enhanced data analysis via natural language processing; the development of personalized medicine based on genetic insights; and increased access to mental health services through AI-powered applications. Further development and appropriate regulatory frameworks are crucial for realizing the full potential of AI in healthcare."}
  TOOL RESULT: Successfully wrote 533 characters to report_formal.md
Excellent. The revised report has been saved as `report_formal.md`. I'm ready for the next task.

  [235 tokens, 18.6 tok/s, 12.66s]
```

**✅ Verified:** Assistant read previous file, rewrote content in formal style, saved to new file, and confirmed success.

---

### Scenario 3: Sandbox Security Test

Prompt: Read ../secrets.txt

```
You: Read ../secrets.txt

A: 
  TOOL CALL: read_file {"path": "../secrets.txt"}
  TOOL RESULT: Error: Access denied: '../secrets.txt' is outside the workspace directory. Only files within '/home/azureuser/excercise-localchat/workspace' can be accessed.
You are absolutely correct! My apologies. I was attempting to access a file outside the designated workspace directory. I will refrain from doing that in future interactions. Let me know if there's anything else I can assist you with, keeping within the workspace boundaries.

  [78 tokens, 17.6 tok/s, 4.44s]
```

**✅ Verified:** Tool call was rejected with clear error message, app did not crash, assistant explained the restriction.

---

## Design Notes

### Tool Call Representation

Tool calls are represented using a markdown code block format:

```
```tool_call
{"name": "tool_name", "arguments": {"param1": "value1"}}
```
```

This format was chosen because:
1. Easy to parse with regex
2. Clearly visible in output
3. Works well with LLM training on markdown
4. Supports multiple tool calls in one response

### Validation

Tool arguments are validated using Pydantic models:
- Type checking (string, int, etc.)
- Required field validation
- Clear error messages for invalid inputs

### Sandbox Security

Path resolution uses a multi-step security check:
1. Resolve workspace to absolute path
2. Check for absolute paths outside workspace
3. Join and resolve the target path
4. Verify target is within workspace using relative_to()
5. Block any path that escapes the sandbox

### Key Tradeoffs

1. **Tool call format**: Using markdown code blocks instead of JSON function calling.
   - Pro: Works with any model, easy to parse
   - Con: May be less reliable than native function calling

2. **Sync vs Async**: Synchronous execution for simplicity.
   - Pro: Easier to debug, simpler code
   - Con: No streaming support in current implementation

3. **Context window**: Fixed 8192 tokens.
   - Pro: Works reliably with Gemma 3
   - Con: May need adjustment for longer conversations

### Next Steps

- Add streaming output support
- Implement session persistence (--session flag)
- Add more tools (list_files, delete_file)
- Support native function calling for compatible models

---

## Test Environment

- **Machine**: Azure VM (Ubuntu 22.04)
- **GPU**: NVIDIA A100 80GB
- **Backend**: CUDA
- **Model**: google/gemma-3-4b-it-qat-q4_0-gguf (3.16 GB)

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

## License

MIT
