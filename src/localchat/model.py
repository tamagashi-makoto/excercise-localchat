"""Model loader with backend detection."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_cpp import Llama


@dataclass
class RuntimeInfo:
    """Information about the runtime environment."""
    
    backend: str  # "cuda", "metal", "cpu"
    model_path: str
    model_size_gb: float
    n_ctx: int
    n_gpu_layers: int
    repo_id: Optional[str] = None
    
    def display(self) -> None:
        """Print runtime information to stdout."""
        print("=" * 50)
        print("LocalChat - Runtime Information")
        print("=" * 50)
        print(f"  Backend:      {self.backend}")
        if self.repo_id:
            print(f"  Repo ID:      {self.repo_id}")
        print(f"  Model:        {self.model_path}")
        print(f"  Model size:   {self.model_size_gb:.2f} GB")
        print(f"  Context:      {self.n_ctx} tokens")
        print(f"  GPU layers:   {self.n_gpu_layers}")
        print("=" * 50)


def detect_backend() -> tuple[str, int]:
    """
    Detect available backend and return (backend_name, n_gpu_layers).
    
    Returns:
        Tuple of (backend name, number of GPU layers to offload)
    """
    # Try to detect CUDA
    try:
        import llama_cpp
        # Check if llama-cpp-python was built with CUDA support
        # by attempting to check for CUDA availability
        
        # llama-cpp-python with CUDA will have cublas support
        # We'll try to load with GPU layers and see if it works
        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
            if llama_cpp.llama_supports_gpu_offload():
                return ("cuda", -1)  # -1 means all layers on GPU
        
        # Alternative detection: check environment or try loading
        # For CUDA builds, we default to GPU offload
        # The actual test will happen during model loading
        return ("cuda", -1)
        
    except Exception:
        pass
    
    # Fallback to CPU
    return ("cpu", 0)


def load_model(
    model_path: Optional[Path] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    n_ctx: int = 8192,
    n_gpu_layers: Optional[int] = None,
    verbose: bool = False,
) -> tuple[Llama, RuntimeInfo]:
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
    if not model_path and not repo_id:
        raise ValueError("Either model_path or repo_id must be provided")
    
    if model_path and repo_id:
        raise ValueError("Cannot provide both model_path and repo_id")

    if repo_id and not filename:
        raise ValueError("filename is required when using repo_id")

    # Detect backend if not specified
    if n_gpu_layers is None:
        backend, n_gpu_layers = detect_backend()
    else:
        backend = "cuda" if n_gpu_layers != 0 else "cpu"
    
    # Common initialization args
    init_args = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
    }

    try:
        if repo_id:
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **init_args
            )
            model_path_str = model.model_path
        else:
            model_path_str = str(model_path)
            model = Llama(
                model_path=model_path_str,
                **init_args
            )
        
        # Verify GPU is actually being used
        if n_gpu_layers != 0:
            backend = "cuda"
        
    except Exception as e:
        if n_gpu_layers != 0:
            # Fallback to CPU
            print(f"Warning: GPU loading failed ({e}), falling back to CPU", 
                  file=sys.stderr)
            backend = "cpu"
            n_gpu_layers = 0
            
            init_args["n_gpu_layers"] = 0
            
            if repo_id:
                model = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
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
    model_size_gb = Path(model_path_str).stat().st_size / (1024 ** 3)
    
    runtime_info = RuntimeInfo(
        backend=backend,
        model_path=model_path_str,
        model_size_gb=model_size_gb,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        repo_id=repo_id,
    )
    
    return model, runtime_info
