"""
CUDA core backend for TensorStream.

This backend uses compiled CUDA extensions for optimized GPU memory management
and tensor operations. It provides better performance than the mmap fallback
for GPU-intensive workloads.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch

from . import BackendInterface
from ..exceptions import BackendError
from ..io import load_from_ts, get_ts_file_info


class CudaCoreBackend(BackendInterface):
    """
    CUDA core backend using compiled extensions.
    
    This backend leverages compiled CUDA/C++ extensions for optimal performance
    when loading and managing tensors on GPU devices.
    """
    
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.name = "cuda_core"
        self._cuda_core = None
        self._memory_pool = {}
        
    def initialize(self) -> None:
        """Initialize the CUDA core backend."""
        if self._initialized:
            return
        
        try:
            from .. import _cuda_core
            self._cuda_core = _cuda_core
        except ImportError as e:
            self._raise_backend_error(
                "initialize", 
                f"CUDA core extension not available: {e}",
                {"suggestion": "Install with CUDA support or use mmap backend"}
            )
        
        # Check CUDA device availability
        if not torch.cuda.is_available():
            self._raise_backend_error(
                "initialize",
                "CUDA not available on this system"
            )
        
        # Initialize CUDA core module
        try:
            self._cuda_core.initialize()
        except Exception as e:
            self._raise_backend_error(
                "initialize",
                f"Failed to initialize CUDA core: {e}"
            )
        
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup backend resources."""
        if self._cuda_core and hasattr(self._cuda_core, 'cleanup'):
            try:
                self._cuda_core.cleanup()
            except Exception as e:
                warnings.warn(f"Failed to cleanup CUDA core: {e}")
        
        # Clear memory pool
        self._memory_pool.clear()
        self._initialized = False
    
    def load_tensor(self, path: Union[str, Path], 
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Load a tensor using CUDA core optimizations.
        
        Args:
            path: Path to the tensor file
            device: Target device for the tensor
            
        Returns:
            Loaded tensor
        """
        if not self._initialized:
            self.initialize()
        
        path = Path(path)
        device = device or self.config.device
        
        try:
            # For CPU tensors, use standard loading
            if device.type == 'cpu':
                return load_from_ts(path, device=device)
            
            # For GPU tensors, use CUDA core optimizations
            if device.type == 'cuda':
                return self._load_tensor_cuda_optimized(path, device)
            
            # Fallback
            return load_from_ts(path, device=device)
        
        except Exception as e:
            self._raise_backend_error("load_tensor", str(e), {"path": str(path)})
    
    def _load_tensor_cuda_optimized(self, path: Path, device: torch.device) -> torch.Tensor:
        """Load tensor with CUDA core optimizations."""
        # Check if file is compressed
        info = get_ts_file_info(path)
        if info["compressed"]:
            # For compressed files, decompress on CPU then transfer
            tensor = load_from_ts(path, device=torch.device('cpu'))
            return tensor.to(device)
        
        # Use CUDA core for direct loading
        try:
            tensor_data = self._cuda_core.load_tensor_direct(str(path), device.index)
            
            # Convert the loaded data to PyTorch tensor
            # Note: This is a placeholder - actual implementation would depend
            # on the C++ extension interface
            return tensor_data
        
        except Exception as e:
            # Fallback to standard loading
            warnings.warn(f"CUDA core loading failed, using fallback: {e}")
            tensor = load_from_ts(path, device=torch.device('cpu'))
            return tensor.to(device)
    
    def unload_tensor(self, tensor: torch.Tensor) -> None:
        """
        Unload a tensor and free its memory using CUDA core.
        
        Args:
            tensor: Tensor to unload
        """
        if tensor.device.type == 'cuda' and self._cuda_core:
            try:
                # Use CUDA core for optimized memory management
                self._cuda_core.free_tensor_memory(tensor.data_ptr())
            except Exception as e:
                warnings.warn(f"CUDA core unload failed: {e}")
        
        # Standard cleanup
        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get current memory usage information."""
        info = {}
        
        # Standard CUDA memory info
        if torch.cuda.is_available():
            info.update({
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_cached": torch.cuda.memory_reserved(),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(),
            })
        
        # CUDA core specific info
        if self._cuda_core and hasattr(self._cuda_core, 'get_memory_info'):
            try:
                core_info = self._cuda_core.get_memory_info()
                info.update({f"cuda_core_{k}": v for k, v in core_info.items()})
            except Exception as e:
                warnings.warn(f"Failed to get CUDA core memory info: {e}")
        
        return info
    
    def is_available(self) -> bool:
        """Check if CUDA core backend is available."""
        try:
            from .. import _cuda_core
            return torch.cuda.is_available()
        except ImportError:
            return False
