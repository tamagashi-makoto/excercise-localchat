"""Tests for chat engine."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile

from localchat.chat import ChatEngine, ToolCall, Message, GenerationStats


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        yield workspace


@pytest.fixture
def mock_model():
    """Create a mock Llama model."""
    model = Mock()
    return model


@pytest.fixture
def tool_executor(temp_workspace):
    """Create a real ToolExecutor with temporary workspace."""
    from localchat.tools import ToolExecutor
    return ToolExecutor(temp_workspace)


class TestChatEngine:
    """Tests for ChatEngine."""
    
    def test_initialization(self, mock_model, tool_executor):
        """Test that ChatEngine initializes correctly."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
            system_prompt="Test prompt",
            temperature=0.5,
            max_tokens=1024,
        )
        
        assert engine.model == mock_model
        assert engine.temperature == 0.5
        assert engine.max_tokens == 1024
        assert len(engine.history) == 0
        assert "Test prompt" in engine.system_prompt
    
    def test_system_prompt_includes_tools(self, mock_model, tool_executor):
        """Test that system prompt includes tool definitions."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        assert "read_file" in engine.system_prompt
        assert "write_file" in engine.system_prompt
        assert "tool_call" in engine.system_prompt
    
    def test_parse_tool_calls_single(self, mock_model, tool_executor):
        """Test parsing a single tool call."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response = '''Let me read that file for you.
```tool_call
{"name": "read_file", "arguments": {"path": "test.txt"}}
```'''
        
        tool_calls, clean_text = engine._parse_tool_calls(response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].arguments == {"path": "test.txt"}
        assert "Let me read that file" in clean_text
        assert "tool_call" not in clean_text
    
    def test_parse_tool_calls_multiple(self, mock_model, tool_executor):
        """Test parsing multiple tool calls."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response = '''I'll read and then write.
```tool_call
{"name": "read_file", "arguments": {"path": "input.txt"}}
```
```tool_call
{"name": "write_file", "arguments": {"path": "output.txt", "content": "Hello"}}
```'''
        
        tool_calls, clean_text = engine._parse_tool_calls(response)
        
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "read_file"
        assert tool_calls[1].name == "write_file"
    
    def test_parse_tool_calls_none(self, mock_model, tool_executor):
        """Test parsing response with no tool calls."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response = "This is just a normal response without any tool calls."
        
        tool_calls, clean_text = engine._parse_tool_calls(response)
        
        assert len(tool_calls) == 0
        assert clean_text == response
    
    def test_parse_tool_calls_invalid_json(self, mock_model, tool_executor):
        """Test parsing response with invalid JSON in tool call."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response = '''Here's a broken tool call:
```tool_call
{invalid json here}
```'''
        
        tool_calls, clean_text = engine._parse_tool_calls(response)
        
        assert len(tool_calls) == 0
    
    def test_chat_simple_response(self, mock_model, tool_executor):
        """Test a simple chat without tool calls."""
        mock_model.create_chat_completion = Mock(return_value={
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10}
        })
        
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response, stats = engine.chat("Hi there!")
        
        assert "Hello" in response
        assert len(engine.history) == 2  # user + assistant
        assert engine.history[0].role == "user"
        assert engine.history[1].role == "assistant"
    
    def test_chat_with_tool_call(self, mock_model, tool_executor, temp_workspace):
        """Test chat with a tool call."""
        # Create test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello from file!")
        
        # First response: tool call
        # Second response: final answer
        mock_model.create_chat_completion = Mock(side_effect=[
            {
                "choices": [{"message": {"content": '''```tool_call
{"name": "read_file", "arguments": {"path": "test.txt"}}
```'''}}],
                "usage": {"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10}
            },
            {
                "choices": [{"message": {"content": "The file contains: Hello from file!"}}],
                "usage": {"total_tokens": 40, "prompt_tokens": 30, "completion_tokens": 10}
            },
        ])
        
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        response, stats = engine.chat("Read test.txt")
        
        assert "Hello from file" in response
        assert stats.completion_tokens == 20  # Accumulated from both calls
    
    def test_clear_history(self, mock_model, tool_executor):
        """Test clearing conversation history."""
        engine = ChatEngine(
            model=mock_model,
            tool_executor=tool_executor,
        )
        
        engine.history.append(Message(role="user", content="test"))
        engine.history.append(Message(role="assistant", content="response"))
        
        assert len(engine.history) == 2
        
        engine.clear_history()
        
        assert len(engine.history) == 0


class TestGenerationStats:
    """Tests for GenerationStats."""
    
    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        stats = GenerationStats(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            duration_seconds=2.0,
        )
        
        assert stats.tokens_per_second == 25.0
    
    def test_tokens_per_second_zero_duration(self):
        """Test tokens per second with zero duration."""
        stats = GenerationStats(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            duration_seconds=0.0,
        )
        
        assert stats.tokens_per_second == 0.0
