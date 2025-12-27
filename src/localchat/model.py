"""Model loader with backend detection."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

try:
    from llama_cpp import Llama
    import llama_cpp
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = Any  # type: ignore


@dataclass
class RuntimeInfo:
    """Information about the runtime environment."""
    
    backend: str  # "cuda", "metal", "cpu"
    model_path: str
    model_size_gb: float
    n_ctx: int
    n_gpu_layers: int
    repo_id: Optional[str] = None
    gpu_offload_supported: bool = False
    
    def display(self) -> None:
        """Print runtime information to stdout."""
        print("=" * 50)
        print("LocalChat - Runtime Information")
        print("=" * 50)
        
        backend_display = self.backend
        if self.backend == "cpu":
             backend_display += f" (gpu_offload_supported={self.gpu_offload_supported})"
             
        print(f"  Backend:      {backend_display}")
        if self.repo_id:
            print(f"  Repo ID:      {self.repo_id}")
        print(f"  Model:        {self.model_path}")
        print(f"  Model size:   {self.model_size_gb:.2f} GB")
        print(f"  Context:      {self.n_ctx} tokens")
        print(f"  GPU layers:   {self.n_gpu_layers}")
        print("=" * 50)


def detect_backend() -> Tuple[str, int, bool]:
    """
    Detect available backend and return info.
    
    Returns:
        Tuple of (backend_name, n_gpu_layers, gpu_offload_supported)
    """
    if not LLAMA_AVAILABLE:
        return ("cpu", 0, False)

    # Check for GPU support in llama-cpp-python
    gpu_supported = False
    if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
         gpu_supported = llama_cpp.llama_supports_gpu_offload()
    
    if gpu_supported:
        # We can't definitively know if it's CUDA or Metal just from python bindings 
        # without checking platform or build info deeper, but we can infer.
        # But commonly we just want to know if we can offload.
        
        # Simple heuristic:
        if sys.platform == "darwin":
            return ("metal", -1, True)
        else:
            return ("cuda", -1, True)
            
    return ("cpu", 0, False)


def load_model(
    model_path: Optional[Path] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    n_ctx: int = 8192,
    n_gpu_layers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Any, RuntimeInfo]:
    """
    Load a GGUF model with automatic backend detection.
    
    Args:
        model_path: Path to the GGUF model file
        repo_id: Hugging Face Hub Repsoitory ID
        filename: Filename in the repository (required if repo_id is provided)
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all, None for auto)
        verbose: Whether to print loading progress
        
    Returns:
        Tuple of (Llama model instance, RuntimeInfo)
    """
    if not LLAMA_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Please install it using 'pip install llama-cpp-python' or run setup.sh."
        )

    if not model_path and not repo_id:
        raise ValueError("Either model_path or repo_id must be provided")
    
    if model_path and repo_id:
        raise ValueError("Cannot provide both model_path and repo_id")

    # Detect backend if not specified
    gpu_supported = False
    if n_gpu_layers is None:
        backend, n_gpu_layers, gpu_supported = detect_backend()
    else:
        # User manual override
        backend, _, gpu_supported = detect_backend() # Check capability
        if n_gpu_layers != 0 and not gpu_supported:
             print("Warning: GPU layers requested but backend reports no GPU support.", file=sys.stderr)
             backend = "cpu"
        elif n_gpu_layers != 0:
             backend = "cuda" if sys.platform != "darwin" else "metal"
        else:
             backend = "cpu"
    
    # Common initialization args
    init_args = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
    }

    try:
        if repo_id:
            # Resolve filename if not provided
            if not filename:
                try:
                    from huggingface_hub import list_repo_files
                    files = list_repo_files(repo_id=repo_id)
                    # Simple heuristic: find first .gguf file
                    # Ideally we might look for specific quantizations like q4_k_m, but first available is a safe start
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    
                    if not gguf_files:
                        raise ValueError(f"No .gguf files found in repository {repo_id}")
                    
                    # Try to find a reasonable default (q4_0 is common/balanced)
                    selected_file = next((f for f in gguf_files if "q4_0" in f), gguf_files[0])
                    filename = selected_file
                    print(f"Auto-selected model file: {filename}")
                    
                except ImportError:
                    pass # Fall through to error from Llama if missing
                except Exception as e:
                    print(f"Warning: Failed to auto-detect filename from HF ({e}).", file=sys.stderr)
                    # Let Llama.from_pretrained fail naturally if it requires it

            kwargs = init_args.copy()
            # Now we should have a filename if auto-detection worked or user provided it
            # We explicitly pass it as a positional argument if the library demands it, 
            # or as kwargs if that's what we tested. The error said "missing 1 required positional argument: 'filename'"
            # This suggests Llama.from_pretrained(repo_id, filename, ...) signature.
            
            # Let's try passing it as kwargs if key exists, but the error suggests positional?
            # Actually Llama.from_pretrained signature is usually (repo_id, filename, ...).
            # If we rely on kwargs expansion **kwargs, we need 'filename' in there OR pass it explicitly.
            
            if filename:
                model = Llama.from_pretrained(
                    repo_id,
                    filename,
                    **init_args
                )
            else:
                 # If we still lack filename, this will likely fail again, but we can't do much more.
                model = Llama.from_pretrained(
                    repo_id=repo_id,
                    **init_args
                )
            model_path_str = model.model_path
        else:
            model_path_str = str(model_path)
            model = Llama(
                model_path=model_path_str,
                **init_args
            )
        
        # Verify GPU is actually being used (heuristic)
        if n_gpu_layers != 0:
             pass
        
    except Exception as e:
        if n_gpu_layers != 0:
            # Fallback to CPU
            print(f"Warning: GPU loading failed ({e}), falling back to CPU", 
                  file=sys.stderr)
            backend = "cpu"
            n_gpu_layers = 0
            
            init_args["n_gpu_layers"] = 0
            
            if repo_id:
                if filename:
                     model = Llama.from_pretrained(
                        repo_id,
                        filename,
                        **init_args
                    )
                else:
                    model = Llama.from_pretrained(
                        repo_id=repo_id,
                        **init_args
                    )
                model_path_str = model.model_path
            else:
                 model = Llama(
                    model_path=model_path_str,
                    **init_args
                )
        else:
            raise
            
    # Calculate model size
    try:
        model_size_gb = Path(model_path_str).stat().st_size / (1024 ** 3)
    except Exception:
        model_size_gb = 0.0
    
    runtime_info = RuntimeInfo(
        backend=backend,
        model_path=model_path_str,
        model_size_gb=model_size_gb,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        repo_id=repo_id,
        gpu_offload_supported=gpu_supported
    )
    
    return model, runtime_info

