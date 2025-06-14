"""
TensorStream: High-performance PyTorch tensor streaming library.

TensorStream enables transparent offloading and streaming of PyTorch model layers
from disk, allowing you to run large models that exceed your available GPU memory.

Basic usage:
    import torch
    import tensorstream
    from transformers import AutoModelForCausalLM

    # Load model
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Configure TensorStream
    config = tensorstream.Config(
        storage_path="/path/to/shards/",
        vram_budget_gb=5,
        backend='auto'
    )
    
    # Apply offloading
    offloaded_model = tensorstream.offload(model, config)
    
    # Use as normal
    offloaded_model.to('cuda:0')
    output = offloaded_model.generate(...)
"""

__version__ = "0.1.0"
__author__ = "TensorStream Contributors"
__email__ = "info@tensorstream.ai"
__license__ = "MIT"

# Core imports
from .config import Config
from .api import offload
from .exceptions import (
    TensorStreamError,
    BackendError,
    ConfigurationError,
    MemoryError as TSMemoryError,
    StorageError,
)

# Backend availability flags
try:
    from . import _cuda_core
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

try:
    from . import _gds_core
    HAS_GDS = True
except ImportError:
    HAS_GDS = False

# Version info
__all__ = [
    "Config",
    "offload", 
    "TensorStreamError",
    "BackendError",
    "ConfigurationError",
    "TSMemoryError",
    "StorageError",
    "HAS_CUDA",
    "HAS_GDS",
    "__version__",
]
