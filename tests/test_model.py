"""Tests for model loading and detection."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

from localchat.model import load_model, RuntimeInfo

@pytest.fixture
def mock_llama():
    with patch("localchat.model.Llama") as mock:
        yield mock

def test_load_model_local(mock_llama):
    """Test loading a model from local path."""
    path = Path("test.gguf")
    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 1024**3  # 1GB
        
        model, info = load_model(model_path=path)
        
        assert info.model_path == "test.gguf"
        assert info.backend == "cuda"  # Default mock behavior in detect_backend
        mock_llama.assert_called_once()
        assert not info.repo_id

def test_load_model_hf(mock_llama):
    """Test loading a model from HF Hub."""
    mock_instance = mock_llama.from_pretrained.return_value
    mock_instance.model_path = "/cache/model.gguf"
    
    with patch.object(Path, "stat") as mock_stat:
        mock_stat.return_value.st_size = 1024**3
        
        model, info = load_model(
            repo_id="test/repo",
            filename="model.gguf"
        )
        
        assert info.repo_id == "test/repo"
        assert info.model_path == "/cache/model.gguf"
        mock_llama.from_pretrained.assert_called_once()

def test_load_model_exclusive_args():
    """Test that model_path and repo_id are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot provide both"):
        load_model(model_path=Path("test.gguf"), repo_id="test/repo")

def test_load_model_missing_args():
    """Test that at least one source is required."""
    with pytest.raises(ValueError, match="Either model_path or repo_id"):
        load_model()

def test_load_model_missing_filename():
    """Test that filename is required with repo_id."""
    with pytest.raises(ValueError, match="filename is required"):
        load_model(repo_id="test/repo")
