"""Tests for tools and sandbox security."""

import pytest
from pathlib import Path
import tempfile
import os

from localchat.tools import (
    ToolExecutor,
    SandboxError,
    resolve_safe_path,
    ReadFileParams,
    WriteFileParams,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        yield workspace


@pytest.fixture
def executor(temp_workspace):
    """Create a ToolExecutor with temporary workspace."""
    return ToolExecutor(temp_workspace)


class TestSandboxSecurity:
    """Tests for sandbox path resolution and security."""
    
    def test_valid_path(self, temp_workspace):
        """Test that valid paths are resolved correctly."""
        result = resolve_safe_path(temp_workspace, "test.txt")
        assert result == temp_workspace / "test.txt"
    
    def test_nested_path(self, temp_workspace):
        """Test that nested paths work correctly."""
        result = resolve_safe_path(temp_workspace, "subdir/test.txt")
        assert result == temp_workspace / "subdir" / "test.txt"
    
    def test_path_traversal_blocked(self, temp_workspace):
        """Test that path traversal attacks are blocked."""
        with pytest.raises(SandboxError) as exc_info:
            resolve_safe_path(temp_workspace, "../secret.txt")
        assert "outside the workspace" in str(exc_info.value)
    
    def test_deep_path_traversal_blocked(self, temp_workspace):
        """Test that deep path traversal is blocked."""
        with pytest.raises(SandboxError):
            resolve_safe_path(temp_workspace, "foo/../../secret.txt")
    
    def test_absolute_path_blocked(self, temp_workspace):
        """Test that absolute paths outside workspace are blocked."""
        with pytest.raises(SandboxError):
            resolve_safe_path(temp_workspace, "/etc/passwd")
    
    def test_absolute_path_with_leading_slash_blocked(self, temp_workspace):
        """Test that paths with leading slash are blocked (treated as absolute)."""
        # /test.txt is an absolute path, not relative to workspace
        with pytest.raises(SandboxError):
            resolve_safe_path(temp_workspace, "/test.txt")


class TestReadFile:
    """Tests for read_file tool."""
    
    def test_read_existing_file(self, executor, temp_workspace):
        """Test reading an existing file."""
        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello, World!")
        
        result = executor.execute("read_file", {"path": "test.txt"})
        assert result == "Hello, World!"
    
    def test_read_nested_file(self, executor, temp_workspace):
        """Test reading a file in a subdirectory."""
        subdir = temp_workspace / "subdir"
        subdir.mkdir()
        test_file = subdir / "nested.txt"
        test_file.write_text("Nested content")
        
        result = executor.execute("read_file", {"path": "subdir/nested.txt"})
        assert result == "Nested content"
    
    def test_read_nonexistent_file(self, executor):
        """Test reading a file that doesn't exist."""
        result = executor.execute("read_file", {"path": "nonexistent.txt"})
        assert "Error:" in result
        assert "not found" in result.lower()
    
    def test_read_path_traversal(self, executor):
        """Test that path traversal is blocked."""
        result = executor.execute("read_file", {"path": "../secret.txt"})
        assert "Error:" in result
        assert "outside the workspace" in result
    
    def test_read_missing_path_param(self, executor):
        """Test that missing path parameter is caught."""
        result = executor.execute("read_file", {})
        assert "Error:" in result


class TestWriteFile:
    """Tests for write_file tool."""
    
    def test_write_new_file(self, executor, temp_workspace):
        """Test writing a new file."""
        result = executor.execute("write_file", {
            "path": "output.txt",
            "content": "Test content"
        })
        
        assert "Successfully wrote" in result
        assert (temp_workspace / "output.txt").read_text() == "Test content"
    
    def test_write_creates_directories(self, executor, temp_workspace):
        """Test that write_file creates parent directories."""
        result = executor.execute("write_file", {
            "path": "deep/nested/dir/file.txt",
            "content": "Deep content"
        })
        
        assert "Successfully wrote" in result
        assert (temp_workspace / "deep/nested/dir/file.txt").read_text() == "Deep content"
    
    def test_write_overwrites_existing(self, executor, temp_workspace):
        """Test that write_file overwrites existing files."""
        test_file = temp_workspace / "existing.txt"
        test_file.write_text("Old content")
        
        result = executor.execute("write_file", {
            "path": "existing.txt",
            "content": "New content"
        })
        
        assert "Successfully wrote" in result
        assert test_file.read_text() == "New content"
    
    def test_write_path_traversal(self, executor):
        """Test that path traversal is blocked on write."""
        result = executor.execute("write_file", {
            "path": "../malicious.txt",
            "content": "Bad content"
        })
        assert "Error:" in result
        assert "outside the workspace" in result
    
    def test_write_missing_content_param(self, executor):
        """Test that missing content parameter is caught."""
        result = executor.execute("write_file", {"path": "test.txt"})
        assert "Error:" in result


class TestToolExecutor:
    """Tests for ToolExecutor general functionality."""
    
    def test_unknown_tool(self, executor):
        """Test calling an unknown tool."""
        result = executor.execute("unknown_tool", {})
        assert "Error:" in result
        assert "Unknown tool" in result
    
    def test_invalid_json_arguments(self, executor):
        """Test handling of invalid JSON string arguments."""
        result = executor.execute("read_file", "not valid json")
        assert "Error:" in result
        assert "Invalid JSON" in result
    
    def test_get_tool_definitions(self, executor):
        """Test that tool definitions are returned correctly."""
        definitions = executor.get_tool_definitions()
        
        assert len(definitions) == 2
        
        tool_names = [d["function"]["name"] for d in definitions]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
