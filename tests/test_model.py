"""Tests for model loading and detection."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import importlib

@pytest.fixture(scope="function")
def model_module():
    """
    Fixture that provides the localchat.model module with mocked dependencies.
    Reloads the module to ensure fresh state for each test.
    """
    # Create mock llama_cpp module
    mock_llama_cpp = MagicMock()
    # Setup Llama class mock
    mock_llama_class = MagicMock()
    mock_llama_cpp.Llama = mock_llama_class
    # Setup gpu support
    mock_llama_cpp.llama_supports_gpu_offload.return_value = True

    # Create mock huggingface_hub
    mock_hv_hub = MagicMock()
    
    with patch.dict(sys.modules, {'llama_cpp': mock_llama_cpp, 'huggingface_hub': mock_hv_hub}):
        # Import and reload localchat.model
        if 'localchat.model' not in sys.modules:
             import localchat.model
        else:
             import localchat.model
             importlib.reload(localchat.model)
        
        # Now patch detect_backend on the reloaded module
        with patch.object(localchat.model, 'detect_backend', return_value=("cuda", -1, True)) as mock_detect:
             localchat.model._mock_llama = mock_llama_class
             localchat.model._mock_detect = mock_detect
             localchat.model._mock_hv_hub = mock_hv_hub # Store for access
             yield localchat.model

def test_load_model_local(model_module):
    """Test loading a model from local path."""
    path = Path("test.gguf")
    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 1024**3  # 1GB
        
        model, info = model_module.load_model(model_path=path)
        
        assert info.model_path == "test.gguf"
        assert info.backend == "cuda"
        model_module._mock_llama.assert_called_once()
        assert not info.repo_id

def test_load_model_hf(model_module):
    """Test loading a model from HF Hub with explicit filename."""
    mock_instance = model_module._mock_llama.from_pretrained.return_value
    mock_instance.model_path = "/cache/model.gguf"
    
    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 1024**3
        
        model, info = model_module.load_model(
            repo_id="test/repo",
            filename="model.gguf"
        )
        
        assert info.repo_id == "test/repo"
        assert info.model_path == "/cache/model.gguf"
        
        # In my new implementation, I call Llama.from_pretrained(repo, filename, **kwargs)
        model_module._mock_llama.from_pretrained.assert_called_once()
        args, kwargs = model_module._mock_llama.from_pretrained.call_args
        assert args[0] == "test/repo"
        assert args[1] == "model.gguf"

def test_load_model_hf_auto_filename(model_module):
    """Test loading a model from HF Hub with auto filename detection."""
    mock_instance = model_module._mock_llama.from_pretrained.return_value
    mock_instance.model_path = "/cache/auto_model.gguf"
    
    # Mock list_repo_files
    model_module._mock_hv_hub.list_repo_files.return_value = ["model.json", "model_q4_0.gguf", "model_q8.gguf"]
    
    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 1024**3
        
        model, info = model_module.load_model(
             repo_id="test/repo"
             # No filename provided
        )
        
        # Should have detected model_q4_0.gguf
        model_module._mock_hv_hub.list_repo_files.assert_called_with(repo_id="test/repo")
        
        args, kwargs = model_module._mock_llama.from_pretrained.call_args
        assert args[0] == "test/repo"
        assert args[1] == "model_q4_0.gguf" # Should pick the one we prioritized or first one
        assert info.repo_id == "test/repo"

def test_load_model_exclusive_args(model_module):
    """Test that model_path and repo_id are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot provide both"):
        model_module.load_model(model_path=Path("test.gguf"), repo_id="test/repo")

def test_load_model_missing_args(model_module):
    """Test that at least one source is required."""
    with pytest.raises(ValueError, match="Either model_path or repo_id"):
        model_module.load_model()

def test_load_model_cpu_fallback(model_module):
    """Test fallback to CPU when GPU load fails."""
    # detect_backend already mocked to return cuda via fixture
    
    # Configure Llama mock to raise on first call
    fallback_model = MagicMock()
    fallback_model.model_path = "fallback.gguf"
    
    mock_llama = model_module._mock_llama
    mock_llama.side_effect = [Exception("CUDA Error"), fallback_model] # First raise, then return logic?
    # Wait, Llama(...) is a class init. 
    # If I access mock_llama directly as the class, side_effect on the class causes instantiation to raise.
    
    path = Path("test.gguf")
    with patch.object(Path, "stat"):
        model, info = model_module.load_model(model_path=path)
        
        assert info.backend == "cpu"
        assert info.n_gpu_layers == 0
        
        assert mock_llama.call_count == 2
        args, kwargs = mock_llama.call_args
        assert kwargs["n_gpu_layers"] == 0

def test_load_model_manual_gpu_override_no_support(model_module):
    """Test requesting GPU layers when backend says no support."""
    model_module._mock_detect.return_value = ("cpu", 0, False)
    
    path = Path("test.gguf")
    with patch.object(Path, "stat"):
        model, info = model_module.load_model(model_path=path, n_gpu_layers=-1)
        
        assert info.backend == "cpu"

