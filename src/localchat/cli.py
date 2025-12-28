"""Command-line interface for LocalChat."""

import argparse
import sys
import json
import os
import datetime
from pathlib import Path

from localchat.model import load_model
from localchat.tools import ToolExecutor
from localchat.chat import run_repl


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="localchat",
        description="Local LLM CLI with Tool Calling",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to GGUF model file OR Hugging Face Repo ID",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        help="[Legacy] Hugging Face Hub Repository ID",
    )

    parser.add_argument(
        "--filename",
        type=str,
        help="Filename in the repository (required if --repo-id is used, optional if --model is used as Repo ID)",
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="./workspace",
        help="Sandbox directory for file tools (default: ./workspace)",
    )
    
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Path to system prompt file",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to session file for history persistence",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Determine mode
    model_path = None
    repo_id = None
    filename = args.filename

    if args.repo_id:
        if args.model:
            print("Error: Cannot provide both --model and --repo-id", file=sys.stderr)
            return 1
        if not args.filename:
            print("Error: --filename is required when using --repo-id", file=sys.stderr)
            return 1
        
        repo_id = args.repo_id
        # filename already set
        
    elif args.model:
        # Check if it looks like a file path
        p = Path(args.model)
        if p.exists():
            model_path = p
        else:
            # Assume it's a repo-id
            repo_id = args.model
            # filename remains as passed in args (optional)
            
    else:
        print("Error: Either --model or --repo-id must be provided", file=sys.stderr)
        return 1
    
    # Ensure workspace directory exists
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Load system prompt if provided
    system_prompt = None
    if args.system:
        system_path = Path(args.system)
        if system_path.exists():
            system_prompt = system_path.read_text()
        else:
            print(f"Warning: System prompt file not found: {system_path}", 
                  file=sys.stderr)
    
    # Load model with backend detection
    print("Loading model...")
    try:
        model, runtime_info = load_model(
            model_path=model_path,
            repo_id=repo_id,
            filename=filename,
        )
        runtime_info.display()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1
    
    # Initialize tool executor
    tool_executor = ToolExecutor(workspace)
    print(f"Workspace: {workspace.absolute()}")
    
    # Start REPL
    run_repl(
        model=model,
        tool_executor=tool_executor,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        session_file=Path(args.session) if args.session else None,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
