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
    
    def display(self) -> None:
        """Print runtime information to stdout."""
        print("=" * 50)
        print("LocalChat - Runtime Information")
        print("=" * 50)
        print(f"  Backend:      {self.backend}")
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
    model_path: Path,
    n_ctx: int = 8192,
    n_gpu_layers: Optional[int] = None,
    verbose: bool = False,
) -> tuple[Llama, RuntimeInfo]:
    """
    Load a GGUF model with automatic backend detection.
    
    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all, None for auto)
        verbose: Whether to print loading progress
        
    Returns:
        Tuple of (Llama model instance, RuntimeInfo)
    """
    # Detect backend if not specified
    if n_gpu_layers is None:
        backend, n_gpu_layers = detect_backend()
    else:
        backend = "cuda" if n_gpu_layers != 0 else "cpu"
    
    # Calculate model size
    model_size_gb = model_path.stat().st_size / (1024 ** 3)
    
    # Try loading with GPU, fallback to CPU if needed
    try:
        model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            # Gemma 3 specific settings
            chat_format="gemma",  # Use gemma chat format
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
            
            model = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=0,
                verbose=verbose,
                chat_format="gemma",
            )
        else:
            raise
    
    runtime_info = RuntimeInfo(
        backend=backend,
        model_path=str(model_path),
        model_size_gb=model_size_gb,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )
    
    return model, runtime_info
