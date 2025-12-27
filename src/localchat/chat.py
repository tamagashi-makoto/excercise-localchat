"""Chat engine with tool calling support."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_cpp import Llama

from localchat.tools import ToolExecutor, TOOL_DEFINITIONS


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    name: str
    arguments: dict


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # For tool result messages


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
        model: Llama,
        tool_executor: ToolExecutor,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Initialize the chat engine.
        
        Args:
            model: The loaded Llama model
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
        """Generate a response from the model."""
        messages = self._build_messages_for_model()
        
        start_time = time.time()
        
        response = self.model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        duration = time.time() - start_time
        
        # Extract response text
        response_text = response["choices"][0]["message"]["content"]
        
        # Extract usage stats
        usage = response.get("usage", {})
        stats = GenerationStats(
            total_tokens=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_seconds=duration,
        )
        
        return response_text, stats
    
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


def run_repl(
    model: Llama,
    tool_executor: ToolExecutor,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> None:
    """
    Run the interactive REPL loop.
    
    Args:
        model: The loaded Llama model
        tool_executor: Tool executor for handling tool calls
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    engine = ChatEngine(
        model=model,
        tool_executor=tool_executor,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
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
            if response:
                print(response)
            
            # Print stats
            print(f"\n  [{stats.completion_tokens} tokens, {stats.tokens_per_second:.1f} tok/s, {stats.duration_seconds:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
