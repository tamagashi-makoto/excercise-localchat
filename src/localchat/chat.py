"""Chat engine with tool calling support."""

import json
import re
import time
import os
import sys
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# from llama_cpp import Llama  # Removed to avoid runtime dependency

@runtime_checkable
class LlamaModelProtocol(Protocol):
    """Protocol for Llama model to avoid runtime dependency on llama_cpp."""
    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 256,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        ...

from localchat.tools import ToolExecutor, TOOL_DEFINITIONS


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    name: str
    arguments: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            name=data["name"],
            arguments=data["arguments"]
        )


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # For tool result messages

    def to_dict(self) -> dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        tool_calls = []
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]
        
        return cls(
            role=data["role"],
            content=data["content"] or "",
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id")
        )


@dataclass 
class GenerationStats:
    """Statistics for a generation."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    
    @property
    def tokens_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.completion_tokens / self.duration_seconds
        return 0.0


class ChatEngine:
    """
    Chat engine that handles conversation flow and tool calling.
    
    Supports the tool calling loop:
    1. User sends message
    2. Model may request tool calls
    3. Tools are executed
    4. Results are sent back to model
    5. Repeat until model gives final response
    """
    
    def __init__(
        self,
        model: LlamaModelProtocol,
        tool_executor: ToolExecutor,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize the chat engine.
        
        Args:
            model: The loaded Llama model (conforming to LlamaModelProtocol)
            tool_executor: Tool executor for handling tool calls
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tool_executor = tool_executor
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize conversation history
        self.history: list[Message] = []
        
        # Set up system prompt with tool information
        self.system_prompt = self._build_system_prompt(system_prompt)
    
    def _build_system_prompt(self, user_system_prompt: Optional[str]) -> str:
        """Build the system prompt including tool definitions."""
        tool_descriptions = []
        for tool_def in TOOL_DEFINITIONS:
            func = tool_def["function"]
            params = func["parameters"]["properties"]
            param_strs = []
            for name, info in params.items():
                param_strs.append(f'{name}: {info["type"]}')
            tool_descriptions.append(
                f"- {func['name']}({', '.join(param_strs)}): {func['description']}"
            )
        
        tools_text = "\n".join(tool_descriptions)
        
        base_prompt = f"""You are a helpful assistant with access to the following tools:

{tools_text}

To use a tool, respond with a JSON object in this exact format:
```tool_call
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
```

You can make multiple tool calls by including multiple tool_call blocks.
After receiving tool results, provide your final response to the user.
Only use tools when necessary to complete the user's request."""

        if user_system_prompt:
            return f"{base_prompt}\n\n{user_system_prompt}"
        return base_prompt
    
    def _build_messages_for_model(self) -> list[dict]:
        """Convert history to format expected by llama-cpp-python."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in self.history:
            if msg.role == "tool":
                # Format tool results as assistant context
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {msg.tool_call_id}:\n{msg.content}"
                })
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return messages
    
    def _parse_tool_calls(self, response_text: str) -> tuple[list[ToolCall], str]:
        """
        Parse tool calls from model response.
        
        Returns:
            Tuple of (list of tool calls, remaining text)
        """
        tool_calls = []
        
        # Pattern to match tool_call blocks
        pattern = r'```tool_call\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if "name" in data and "arguments" in data:
                    tool_calls.append(ToolCall(
                        name=data["name"],
                        arguments=data["arguments"]
                    ))
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
        
        # Remove tool_call blocks from response
        clean_text = re.sub(pattern, '', response_text, flags=re.DOTALL).strip()
        
        return tool_calls, clean_text
    
    def _generate_response(self) -> tuple[str, GenerationStats]:
        """Generate a response from the model, optionally streaming it."""
        messages = self._build_messages_for_model()
        
        start_time = time.time()
        
        full_text = ""
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        
        # Buffer for tool call detection
        buffer = ""
        in_tool_call_block = False
        
        try:
            # Try to stream first
            response_iter = self.model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            # If the model doesn't support streaming (or returns dict immediately), 
            # it might not return an iterator. 
            # However, Llama.cpp usually returns a generator if stream=True.
            # If it creates a single dict (not iterable of chunks), we handle it.
            if isinstance(response_iter, dict):
                 # Fallback for non-streaming response that ignored stream=True
                 response = response_iter
                 full_text = response["choices"][0]["message"]["content"] or ""
                 usage = response.get("usage", {})
                 print(full_text, end="", flush=True) # Emulate stream
                 
                 return full_text, GenerationStats(
                    total_tokens=usage.get("total_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    duration_seconds=time.time() - start_time,
                 )

            marker = "```tool_call"
            
            for chunk in response_iter:
                if not isinstance(chunk, dict):
                    continue
                    
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    completion_tokens += 1
                    full_text += content
                    buffer += content
                    
                    if in_tool_call_block:
                         # We are in tool block, suppress output
                         # We could check for end of block here if we wanted to resume printing
                         # But per design, we suppress everything after tool call starts
                         pass
                    else:
                        # Check if we found the marker
                        if marker in buffer:
                            in_tool_call_block = True
                            # Print everything before the marker
                            split_index = buffer.find(marker)
                            to_print = buffer[:split_index]
                            if to_print:
                                print(to_print, end="", flush=True)
                            buffer = buffer[split_index:] # Keep rest in buffer (suppressed)
                        else:
                            # Check for partial marker at the end
                            possible_match_len = 0
                            # Check suffixes of buffer
                            # Optimization: only check suffixes up to len(marker) - 1
                            # and start checking from longest potential match
                            max_check = min(len(buffer), len(marker) - 1)
                            for i in range(max_check, 0, -1):
                                suffix = buffer[-i:]
                                if marker.startswith(suffix):
                                    possible_match_len = i
                                    break
                            
                            if possible_match_len > 0:
                                # Printable part is everything except the suffix
                                to_print = buffer[:-possible_match_len]
                                if to_print:
                                    print(to_print, end="", flush=True)
                                buffer = buffer[-possible_match_len:] # Keep the partial match for next iteration
                            else:
                                # No partial match, safe to print all
                                print(buffer, end="", flush=True)
                                buffer = ""
        
        except TypeError:
            # Fallback if unexpected arguments or behavior
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            full_text = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            print(full_text, end="", flush=True) # Print all at once

        duration = time.time() - start_time
        
        # If we didn't get usage from a direct dict API, estimate it
        if total_tokens == 0:
             # Very rough estimate
             # We don't have prompt tokens easily available from stream chunks 
             # (llama-cpp-python puts usage in the *last* chunk usually, but not guaranteed)
             # Let's try to trust the stats return if possible, or just use length.
             completion_tokens = len(full_text) // 4 
             
        stats = GenerationStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration,
        )
        
        return full_text, stats
    
    def chat(self, user_input: str) -> tuple[str, GenerationStats]:
        """
        Process a user message and return the assistant's response.
        
        Handles the full tool calling loop if needed.
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (assistant response, generation stats)
        """
        # Add user message to history
        self.history.append(Message(role="user", content=user_input))
        
        total_stats = GenerationStats(
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            duration_seconds=0.0,
        )
        
        max_tool_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_tool_iterations:
            iteration += 1
            
            # Generate response
            response_text, stats = self._generate_response()
            
            # Accumulate stats
            total_stats.total_tokens += stats.total_tokens
            total_stats.prompt_tokens += stats.prompt_tokens
            total_stats.completion_tokens += stats.completion_tokens
            total_stats.duration_seconds += stats.duration_seconds
            
            # Parse tool calls
            tool_calls, clean_text = self._parse_tool_calls(response_text)
            
            if not tool_calls:
                # No tool calls, this is the final response
                self.history.append(Message(role="assistant", content=response_text))
                return response_text, total_stats
            
            # Process tool calls
            self.history.append(Message(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls
            ))
            
            for tool_call in tool_calls:
                # Log the tool call
                print(f"\n  TOOL CALL: {tool_call.name} {json.dumps(tool_call.arguments)}")
                
                # Execute the tool
                result = self.tool_executor.execute(
                    tool_call.name,
                    tool_call.arguments
                )
                
                # Log the result (truncated)
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  TOOL RESULT: {result_preview}")
                
                # Add tool result to history
                self.history.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call.name
                ))
        
        # Max iterations reached
        final_response = "I apologize, but I was unable to complete the request within the allowed number of steps."
        self.history.append(Message(role="assistant", content=final_response))
        return final_response, total_stats
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()

    def get_history_as_dicts(self) -> list[dict[str, Any]]:
        """Return history as a list of dictionaries."""
        return [msg.to_dict() for msg in self.history]

    def load_history_from_dicts(self, history_data: list[dict[str, Any]]) -> None:
        """Load history from a list of dictionaries."""
        self.history = [Message.from_dict(msg_data) for msg_data in history_data]


def run_repl(
    model: LlamaModelProtocol,
    tool_executor: ToolExecutor,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    session_file: Optional[Path] = None,
) -> None:
    """
    Run the interactive REPL loop.
    
    Args:
        model: The loaded Llama model
        tool_executor: Tool executor for handling tool calls
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        session_file: Optional path to session file for persistence
    """
    engine = ChatEngine(
        model=model,
        tool_executor=tool_executor,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Load session if provided
    if session_file and session_file.exists():
        try:
            print(f"Loading session from {session_file}...")
            content = session_file.read_text(encoding="utf-8")
            data = json.loads(content)
            if "history" in data:
                engine.load_history_from_dicts(data["history"])
                print(f"Restored {len(engine.history)} messages.")
            else:
                print("Warning: Session file missing 'history' key. Starting new session.")
        except Exception as e:
            print(f"Warning: Failed to load session: {e}. Starting new session.")
    
    print("\nLocalChat Ready! Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to clear conversation history.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                engine.clear_history()
                print("Conversation history cleared.")
                continue
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response, stats = engine.chat(user_input)
            
            # Print response (if not already printed during tool calls)
            if response and False: # Disabled because chat() now streams to stdout
                print(response)
            
            # Print stats
            print(f"\n  [{stats.completion_tokens} tokens, {stats.tokens_per_second:.1f} tok/s, {stats.duration_seconds:.2f}s]")
            
            # Save session if configured
            if session_file:
                try:
                    data = {
                        "version": 1,
                        "created_at": datetime.datetime.now().isoformat(),
                        "history": engine.get_history_as_dicts()
                    }
                    # Atomic write
                    temp_file = session_file.with_suffix(session_file.suffix + ".tmp")
                    temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    os.replace(temp_file, session_file)
                except Exception as e:
                    print(f"\nWarning: Failed to save session: {e}", file=sys.stderr)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
