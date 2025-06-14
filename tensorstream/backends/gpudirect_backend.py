"""
GPUDirect Storage backend for TensorStream.

This backend uses NVIDIA's GPUDirect Storage (GDS) technology for direct
storage-to-GPU transfers, bypassing system memory entirely for maximum
performance with large models.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch

from . import BackendInterface
from ..exceptions import BackendError
from ..io import load_from_ts, get_ts_file_info


class GPUDirectBackend(BackendInterface):
    """
    GPUDirect Storage backend for maximum I/O performance.
    
    This backend leverages NVIDIA's GPUDirect Storage (GDS) to transfer
    data directly from storage to GPU memory, bypassing the CPU entirely.
    Requires compatible storage systems and CUDA drivers.
    """
    
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.name = "gpudirect"
        self._gds_core = None
        self._file_handles = {}
        self._memory_registrations = {}
        
    def initialize(self) -> None:
        """Initialize the GPUDirect Storage backend."""
        if self._initialized:
            return
        
        try:
            from .. import _gds_core
            self._gds_core = _gds_core
        except ImportError as e:
            self._raise_backend_error(
                "initialize",
                f"GPUDirect Storage extension not available: {e}",
                {"suggestion": "Install with GDS support or use alternative backend"}
            )
        
        # Check system requirements
        if not torch.cuda.is_available():
            self._raise_backend_error(
                "initialize",
                "CUDA not available - required for GPUDirect Storage"
            )
        
        # Initialize GDS subsystem
        try:
            result = self._gds_core.initialize_gds()
            if not result:
                self._raise_backend_error(
                    "initialize",
                    "Failed to initialize GPUDirect Storage subsystem"
                )
        except Exception as e:
            self._raise_backend_error(
                "initialize",
                f"GDS initialization error: {e}"
            )
        
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup backend resources."""
        # Close all file handles
        for file_path, handle in self._file_handles.items():
            try:
                if self._gds_core:
                    self._gds_core.close_file(handle)
            except Exception as e:
                warnings.warn(f"Failed to close GDS file {file_path}: {e}")
        
        self._file_handles.clear()
        
        # Unregister memory regions
        for ptr, registration in self._memory_registrations.items():
            try:
                if self._gds_core:
                    self._gds_core.unregister_memory(registration)
            except Exception as e:
                warnings.warn(f"Failed to unregister memory {ptr}: {e}")
        
        self._memory_registrations.clear()
        
        # Cleanup GDS subsystem
        if self._gds_core and hasattr(self._gds_core, 'cleanup_gds'):
            try:
                self._gds_core.cleanup_gds()
            except Exception as e:
                warnings.warn(f"Failed to cleanup GDS: {e}")
        
        self._initialized = False
    
    def load_tensor(self, path: Union[str, Path], 
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Load a tensor using GPUDirect Storage.
        
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
        
        # GPUDirect only works for GPU devices
        if device.type != 'cuda':
            return load_from_ts(path, device=device)
        
        try:
            return self._load_tensor_gds(path, device)
        except Exception as e:
            # Fallback to standard loading
            warnings.warn(f"GDS loading failed, using fallback: {e}")
            return load_from_ts(path, device=device)
    
    def _load_tensor_gds(self, path: Path, device: torch.device) -> torch.Tensor:
        """Load tensor using GPUDirect Storage optimizations."""
        # Check if file is compressed
        info = get_ts_file_info(path)
        if info["compressed"]:
            # GDS works best with uncompressed data
            # Fall back to standard loading for compressed files
            tensor = load_from_ts(path, device=torch.device('cpu'))
            return tensor.to(device)
        
        # Open file with GDS if not already open
        file_path_str = str(path)
        if file_path_str not in self._file_handles:
            try:
                handle = self._gds_core.open_file(file_path_str)
                self._file_handles[file_path_str] = handle
            except Exception as e:
                raise RuntimeError(f"Failed to open file with GDS: {e}")
        
        file_handle = self._file_handles[file_path_str]
        
        # Read header to understand tensor layout
        from ..io import read_ts_header
        header = read_ts_header(path)
        
        # Allocate GPU memory for tensor
        torch_dtype = eval(header.tensor_dtype)
        
        # Calculate tensor size
        element_size = torch.tensor([], dtype=torch_dtype).element_size()
        total_elements = 1
        for dim in header.tensor_shape:
            total_elements *= dim
        tensor_bytes = total_elements * element_size
        
        # Allocate tensor on GPU
        tensor = torch.empty(header.tensor_shape, dtype=torch_dtype, device=device)
        
        # Register GPU memory with GDS
        gpu_ptr = tensor.data_ptr()
        if gpu_ptr not in self._memory_registrations:
            try:
                registration = self._gds_core.register_memory(gpu_ptr, tensor_bytes)
                self._memory_registrations[gpu_ptr] = registration
            except Exception as e:
                raise RuntimeError(f"Failed to register GPU memory with GDS: {e}")
        
        # Perform direct storage-to-GPU transfer
        try:
            bytes_transferred = self._gds_core.read_direct(
                file_handle,
                header.header_size,  # offset in file
                gpu_ptr,             # GPU memory address
                tensor_bytes         # number of bytes to transfer
            )
            
            if bytes_transferred != tensor_bytes:
                raise RuntimeError(
                    f"Incomplete transfer: {bytes_transferred}/{tensor_bytes} bytes"
                )
        
        except Exception as e:
            raise RuntimeError(f"GDS direct read failed: {e}")
        
        return tensor
    
    def unload_tensor(self, tensor: torch.Tensor) -> None:
        """
        Unload a tensor and free its memory.
        
        Args:
            tensor: Tensor to unload
        """
        if tensor.device.type == 'cuda':
            # Unregister memory if it was registered with GDS
            gpu_ptr = tensor.data_ptr()
            if gpu_ptr in self._memory_registrations:
                try:
                    registration = self._memory_registrations[gpu_ptr]
                    self._gds_core.unregister_memory(registration)
                    del self._memory_registrations[gpu_ptr]
                except Exception as e:
                    warnings.warn(f"Failed to unregister GDS memory: {e}")
        
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
        
        # GDS specific info
        if self._gds_core and hasattr(self._gds_core, 'get_stats'):
            try:
                gds_stats = self._gds_core.get_stats()
                info.update({f"gds_{k}": v for k, v in gds_stats.items()})
            except Exception as e:
                warnings.warn(f"Failed to get GDS stats: {e}")
        
        # File handles and memory registrations
        info.update({
            "gds_open_files": len(self._file_handles),
            "gds_registered_regions": len(self._memory_registrations),
        })
        
        return info
    
    def is_available(self) -> bool:
        """Check if GPUDirect Storage backend is available."""
        try:
            from .. import _gds_core
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                self.cleanup()
            except Exception:
                pass  # Ignore errors during destruction
