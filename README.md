# LocalChat - Local LLM CLI with Tool Calling

A command-line application that runs a local LLM with interactive chat and tool/function calling support.

## Features

- Fully local inference (no external APIs)
- Tool calling support (read_file, write_file)
- Sandboxed file operations (path traversal protection)
- CUDA/Metal/CPU backend auto-detection
- Interactive REPL with session history
- Generation statistics (tokens/sec, response time)

## Requirements

- Python 3.10+
- CUDA Toolkit 11.x or 12.x (for GPU acceleration)
- ~4GB disk space for model

## Installation

### Quick Setup (Recommended)

```bash
git clone <repository-url>
cd localchat
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -e .
mkdir -p models workspace
```

### Download Model

1. Visit https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
2. Accept the license agreement
3. Download:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-qat-q4_0.gguf --local-dir models
```

## Usage

```bash
source venv/bin/activate
localchat --model models/gemma-3-4b-it-qat-q4_0.gguf --workspace ./workspace
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| --model | Path to GGUF model file | (required) |
| --workspace | Sandbox directory for file tools | ./workspace |
| --system | Path to system prompt file | None |
| --temperature | Sampling temperature | 0.7 |
| --max-tokens | Maximum tokens to generate | 2048 |

### REPL Commands

- quit/exit: End the session
- clear: Clear conversation history

## Supported Tools

- read_file(path): Read file content from workspace
- write_file(path, content): Write content to file in workspace

Security: All file operations are sandboxed. Path traversal attempts are blocked.

---

## Scenario Transcripts

> NOTE: Replace with actual transcripts from your test runs

### Scenario 1: File Summarization + Write Report

Setup: workspace/input.txt contains article about AI in healthcare.

Prompt: Read input.txt, summarize it in 5 bullets, and write the bullets to report.md.

Expected output:
```
You: Read input.txt, summarize it in 5 bullets, and write the bullets to report.md.

  TOOL CALL: read_file {"path": "input.txt"}
  TOOL RESULT: The Rise of Artificial Intelligence in Healthcare...

  TOOL CALL: write_file {"path": "report.md", "content": "# Summary\n\n- Point 1..."}
  TOOL RESULT: Successfully wrote 425 characters to report.md

Assistant: Done! report.md has been written.

  [156 tokens, 45.2 tok/s, 3.45s]
```

### Scenario 2: Iterative Refinement

Prompt: Now read report.md and rewrite it to be more formal. Save as report_formal.md.

```
You: Now read report.md and rewrite it to be more formal. Save as report_formal.md.

  TOOL CALL: read_file {"path": "report.md"}
  TOOL RESULT: # Summary...

  TOOL CALL: write_file {"path": "report_formal.md", "content": "..."}
  TOOL RESULT: Successfully wrote 380 characters to report_formal.md

Assistant: Done! The formal version has been saved.

  [142 tokens, 48.1 tok/s, 2.95s]
```

### Scenario 3: Sandbox Security Test

Prompt: Read ../secrets.txt

```
You: Read ../secrets.txt

  TOOL CALL: read_file {"path": "../secrets.txt"}
  TOOL RESULT: Error: Access denied: '../secrets.txt' is outside the workspace directory.

Assistant: I cannot read that file. The path ../secrets.txt attempts to access a location outside the workspace directory. For security reasons, I can only access files within the designated workspace folder.

  [89 tokens, 52.3 tok/s, 1.70s]
```

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