# Design Approach Explanation

This document explains the thinking process and technical decisions made when designing and implementing the LocalChat CLI.

---

## 1. Problem Analysis

I broke down the challenge into four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    LocalChat Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CLI Layer │→ │  Chat Layer │→ │  LLM Inference Layer│  │
│  │ (cli.py)    │  │ (chat.py)   │  │  (llama-cpp-python) │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────┘  │
│                          │                                   │
│                          ↓                                   │
│                   ┌─────────────┐                            │
│                   │  Tools Layer│                            │
│                   │ (tools.py)  │                            │
│                   └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Rationale for Technology Selection

### Why llama-cpp-python?

| Option | Pros | Cons | Adopted |
|--------|------|------|---------|
| llama-cpp-python | Auto-detects CUDA/Metal, GGUF support, lightweight | Complex configuration | ✅ |
| Ollama | Easy setup | Requires daemon process | ❌ |
| transformers | HuggingFace integration | High memory usage | ❌ |
| vLLM | High throughput | Complex, overkill | ❌ |

**Decision rationale**: Best suited for the "local inference only" requirement, with automatic hardware acceleration detection.

### Why Gemma 3 4B?

- **Size**: 3.16GB after quantization (runs comfortably on A100)
- **Tool Calling capability**: Can generate tool calls following instructions
- **License**: Commercially usable
- **Performance**: Practical speed at ~18 tok/s

---

## 3. Tool Calling Design Thinking Process

### Challenge: How to make the LLM call tools

**Options considered:**

1. **Native Function Calling** (OpenAI format)
   - Pro: Structured
   - Con: Model-dependent, limited support in llama-cpp

2. **Markdown Code Block format** ← Adopted
   - Pro: Works with any model, easy to parse
   - Con: Model may not strictly follow the format

**Adopted format:**
```
```tool_call
{"name": "read_file", "arguments": {"path": "input.txt"}}
```
```

**Rationale**: Gemma 3 understands Markdown and is trained to generate code blocks, making this format natural to output.

---

## 4. Sandbox Design Thinking Process

### Threat Model

```
Attack: User requests access to "../secrets.txt" or "/etc/passwd"
      ↓
Defense: Multi-layer checks during path resolution
```

### Implemented Defenses

```python
def _resolve_safe_path(self, path: str) -> Path:
    # 1. Directly reject absolute paths
    if os.path.isabs(path):
        if not path.startswith(str(self.workspace)):
            raise SecurityError(...)
    
    # 2. Resolve relative paths
    target = (self.workspace / path).resolve()
    
    # 3. Verify resolved path is within workspace
    target.relative_to(self.workspace)  # Raises exception if outside
    
    return target
```

**Why multi-step?**
- Prevents traversal via `../`
- Prevents symlink attacks
- Prevents absolute path injection

---

## 5. Tool Calling Loop Design

```
User Input
    ↓
┌─────────────────────────────────────────┐
│           LLM Response Generation       │
└────────────────┬────────────────────────┘
                 ↓
         ┌───────────────┐
         │ Tool Call     │
         │ Detected?     │
         └───────┬───────┘
           Yes   │   No
            ↓    │    ↓
    ┌────────────┐   ┌────────────┐
    │ Execute    │   │ Display    │
    │ Tool       │   │ Response   │
    └─────┬──────┘   └────────────┘
          ↓
    ┌────────────┐
    │ Add Result │
    │ to Messages│
    └─────┬──────┘
          ↓
    (Return to LLM Response Generation - Loop)
```

**Important design decisions:**
- Add tool results as user role, not system role (aligned with Gemma 3 behavior)
- Set maximum loop count to prevent infinite loops

---

## 6. Implementation Order Strategy

Implementation proceeded in the following order:

```
Phase 1: Foundation Building
├── Project structure design
├── CLI argument parser implementation
└── Basic LLM loading

Phase 2: Core Features
├── REPL loop implementation
├── Conversation history management
└── Tool definitions (Pydantic schemas)

Phase 3: Tool Execution
├── Tool Call parser
├── read_file / write_file implementation
└── Sandbox functionality

Phase 4: Integration and Testing
├── Verification of 3 scenarios
├── Error handling improvements
└── Log output adjustments
```

---

## 7. Challenges Encountered and Solutions

### Challenge 1: Model not generating Tool Calls

**Cause**: Insufficient system prompt

**Solution**: Detailed system prompt explicitly describing how to use tools
```python
SYSTEM_PROMPT = """You are a helpful assistant with access to file tools.
When you need to read or write files, use the tool_call format:
```tool_call
{"name": "tool_name", "arguments": {...}}
```
"""
```

### Challenge 2: Path traversal detection

**Cause**: Simple string checks cannot detect `./foo/../../../etc/passwd`

**Solution**: Normalize with `Path.resolve()` then validate with `relative_to()`

### Challenge 3: Model download from HuggingFace

**Cause**: `--model` alone cannot handle retrieval from HuggingFace repositories

**Solution**: Added `--repo-id` and `--filename` options, using `huggingface_hub` API for automatic download

---

## 8. Future Improvement Ideas

| Priority | Improvement Item | Reason |
|----------|-----------------|--------|
| High | Streaming output | UX improvement, reduced perceived wait time |
| High | Session persistence | Conversation continuity |
| Medium | Additional tools | list_files, delete_file, etc. |
| Medium | Expanded unit tests | Quality assurance |
| Low | Native Function Calling | Waiting for model support |

---

## Summary

This project followed these design principles:

1. **Simplicity first**: Avoid complex frameworks, implement only what's necessary
2. **Security focused**: Robust sandbox with defense in depth
3. **Extensibility**: Structure that makes adding new tools easy
4. **Observability**: Clear logging of tool calls and results

As a result, a CLI application meeting all requirements was completed in approximately 4 hours.
