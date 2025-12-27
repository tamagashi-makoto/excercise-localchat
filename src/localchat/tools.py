"""Tool definitions and implementations with sandbox security."""

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


# =============================================================================
# Tool Parameter Schemas (Pydantic models for validation)
# =============================================================================

class ReadFileParams(BaseModel):
    """Parameters for read_file tool."""
    path: str = Field(..., description="Path to the file to read (relative to workspace)")


class WriteFileParams(BaseModel):
    """Parameters for write_file tool."""
    path: str = Field(..., description="Path to the file to write (relative to workspace)")
    content: str = Field(..., description="Content to write to the file")


# =============================================================================
# Sandbox Security
# =============================================================================

class SandboxError(Exception):
    """Raised when a path traversal or sandbox violation is detected."""
    pass


def resolve_safe_path(workspace: Path, relative_path: str) -> Path:
    """
    Safely resolve a path within the workspace sandbox.
    
    Args:
        workspace: The workspace root directory (absolute path)
        relative_path: The user-provided relative path
        
    Returns:
        Absolute path that is guaranteed to be within workspace
        
    Raises:
        SandboxError: If the path would escape the workspace
    """
    # Normalize workspace to absolute path
    workspace = workspace.resolve()
    
    # Check for absolute paths that don't start with workspace
    if relative_path.startswith("/") or relative_path.startswith("\\"):
        # Check if it's trying to access a path outside workspace
        test_path = Path(relative_path).resolve()
        try:
            test_path.relative_to(workspace)
            # It's within workspace, allow it
        except ValueError:
            raise SandboxError(
                f"Access denied: '{relative_path}' is outside the workspace directory. "
                f"Only files within '{workspace}' can be accessed."
            )
    
    # Clean the relative path (remove leading slashes for joining)
    clean_path = relative_path.lstrip("/").lstrip("\\")
    
    # Resolve the full path
    target = (workspace / clean_path).resolve()
    
    # Security check: ensure target is within workspace
    try:
        target.relative_to(workspace)
    except ValueError:
        raise SandboxError(
            f"Access denied: '{relative_path}' is outside the workspace directory. "
            f"Only files within '{workspace}' can be accessed."
        )
    
    return target


# =============================================================================
# Tool Implementations
# =============================================================================

def read_file_impl(workspace: Path, params: ReadFileParams) -> str:
    """
    Read a file from the workspace.
    
    Args:
        workspace: The workspace root directory
        params: Validated parameters
        
    Returns:
        File content as string
    """
    target = resolve_safe_path(workspace, params.path)
    
    if not target.exists():
        raise FileNotFoundError(f"File not found: {params.path}")
    
    if not target.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {params.path}")
    
    return target.read_text(encoding="utf-8")


def write_file_impl(workspace: Path, params: WriteFileParams) -> str:
    """
    Write content to a file in the workspace.
    
    Args:
        workspace: The workspace root directory
        params: Validated parameters
        
    Returns:
        Confirmation message
    """
    target = resolve_safe_path(workspace, params.path)
    
    # Create parent directories if needed
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the file
    target.write_text(params.content, encoding="utf-8")
    
    return f"Successfully wrote {len(params.content)} characters to {params.path}"


# =============================================================================
# Tool Registry
# =============================================================================

# Tool definitions for the LLM (JSON Schema format)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to workspace)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative to workspace)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]


class ToolExecutor:
    """Executor for tools with validation and sandbox security."""
    
    def __init__(self, workspace: Path):
        """
        Initialize the tool executor.
        
        Args:
            workspace: The workspace root directory for file operations
        """
        self.workspace = workspace.resolve()
        
        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Tool registry: name -> (param_model, implementation)
        self._tools: dict[str, tuple[type[BaseModel], Callable]] = {
            "read_file": (ReadFileParams, self._read_file),
            "write_file": (WriteFileParams, self._write_file),
        }
    
    def _read_file(self, params: ReadFileParams) -> str:
        """Execute read_file tool."""
        return read_file_impl(self.workspace, params)
    
    def _write_file(self, params: WriteFileParams) -> str:
        """Execute write_file tool."""
        return write_file_impl(self.workspace, params)
    
    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for the LLM."""
        return TOOL_DEFINITIONS
    
    def execute(self, tool_name: str, arguments: dict[str, Any] | str) -> str:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments (dict or JSON string)
            
        Returns:
            Tool result as string
        """
        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON arguments: {e}"
        
        # Check if tool exists
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self._tools.keys())}"
        
        param_model, impl = self._tools[tool_name]
        
        # Validate arguments
        try:
            params = param_model(**arguments)
        except ValidationError as e:
            errors = e.errors()
            error_msgs = [f"{err['loc'][0]}: {err['msg']}" for err in errors]
            return f"Error: Invalid arguments: {'; '.join(error_msgs)}"
        
        # Execute the tool
        try:
            result = impl(params)
            return result
        except SandboxError as e:
            return f"Error: {e}"
        except FileNotFoundError as e:
            return f"Error: {e}"
        except IsADirectoryError as e:
            return f"Error: {e}"
        except PermissionError as e:
            return f"Error: Permission denied: {e}"
        except Exception as e:
            return f"Error: Unexpected error: {type(e).__name__}: {e}"
