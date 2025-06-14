"""
Memory-mapped file backend for TensorStream.

This backend uses Python's mmap module for file I/O with fallback CUDA memory
management using ctypes. It provides a pure Python implementation that works
on any system with CUDA support.
"""

import ctypes
import mmap
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch

from . import BackendInterface
from ..exceptions import BackendError
from ..io import load_from_ts, get_ts_file_info


# CUDA Runtime API function signatures
try:
    # Try to load CUDA runtime
    if os.name == 'nt':  # Windows
        cuda_rt = ctypes.CDLL('cudart64_110.dll')  # Adjust version as needed
    else:  # Linux/Unix
        cuda_rt = ctypes.CDLL('libcudart.so')
    
    # Define function signatures
    cuda_rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda_rt.cudaMalloc.restype = ctypes.c_int
    
    cuda_rt.cudaFree.argtypes = [ctypes.c_void_p]
    cuda_rt.cudaFree.restype = ctypes.c_int
    
    cuda_rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                   ctypes.c_size_t, ctypes.c_int]
    cuda_rt.cudaMemcpy.restype = ctypes.c_int
    
    cuda_rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(ctypes.c_size_t)]
    cuda_rt.cudaMemGetInfo.restype = ctypes.c_int
    
    # CUDA memory copy types
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    
    CUDA_RT_AVAILABLE = True

except (OSError, AttributeError) as e:
    CUDA_RT_AVAILABLE = False
    cuda_rt = None
    warnings.warn(f"CUDA runtime not available for mmap backend: {e}")


class MmapBackend(BackendInterface):
    """
    Memory-mapped file backend with CUDA support.
    
    This backend uses mmap for efficient file I/O and ctypes to interface
    with CUDA for GPU memory management. It provides a fallback implementation
    when native CUDA extensions are not available.
    """
    
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.name = "mmap"
        self._memory_pool = {}  # Track allocated memory
        self._mmap_files = {}   # Track memory-mapped files
        
    def initialize(self) -> None:
        """Initialize the mmap backend."""
        if self._initialized:
            return
        
        # Check CUDA availability if needed
        if self.config.device.type == 'cuda' and not CUDA_RT_AVAILABLE:
            warnings.warn(
                "CUDA runtime not available. GPU operations will use PyTorch's "
                "default memory management, which may be slower."
            )
        
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup backend resources."""
        # Close all memory-mapped files
        for file_path, (mm, f) in self._mmap_files.items():
            try:
                mm.close()
                f.close()
            except Exception as e:
                warnings.warn(f"Failed to close mmap for {file_path}: {e}")
        
        self._mmap_files.clear()
        
        # Free any remaining CUDA memory
        if CUDA_RT_AVAILABLE:
            for ptr in list(self._memory_pool.keys()):
                try:
                    self._cuda_free(ptr)
                except Exception as e:
                    warnings.warn(f"Failed to free CUDA memory: {e}")
        
        self._memory_pool.clear()
        self._initialized = False
    
    def load_tensor(self, path: Union[str, Path], 
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Load a tensor using memory mapping.
        
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
            # For CPU tensors, use direct mmap loading
            if device.type == 'cpu':
                return self._load_tensor_cpu_mmap(path)
            
            # For GPU tensors, use optimized loading if available
            if device.type == 'cuda' and CUDA_RT_AVAILABLE:
                return self._load_tensor_gpu_optimized(path, device)
            else:
                # Fallback to PyTorch's default loading
                return self._load_tensor_gpu_fallback(path, device)
        
        except Exception as e:
            self._raise_backend_error("load_tensor", str(e), {"path": str(path)})
    
    def _load_tensor_cpu_mmap(self, path: Path) -> torch.Tensor:
        """Load tensor directly using memory mapping for CPU."""
        # Get file info first
        info = get_ts_file_info(path)
        
        # If compressed, fall back to regular loading
        if info["compressed"]:
            return load_from_ts(path, device=torch.device('cpu'))
        
        # Open file with mmap
        f = open(path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Store for cleanup
        self._mmap_files[str(path)] = (mm, f)
        
        # Read header to get data offset
        from ..io import read_ts_header
        header = read_ts_header(path)
        
        # Map tensor data directly
        data_offset = header.header_size
        tensor_size = header.data_size
        
        # Create numpy array from mmap
        import numpy as np
        
        # Get numpy dtype
        torch_dtype = eval(header.tensor_dtype)
        if torch_dtype == torch.bfloat16:
            numpy_dtype = np.float32
        else:
            from ..io import TORCH_TO_NUMPY_DTYPE
            numpy_dtype = TORCH_TO_NUMPY_DTYPE[torch_dtype]
        
        # Create memory view
        buffer = mm[data_offset:data_offset + tensor_size]
        numpy_array = np.frombuffer(buffer, dtype=numpy_dtype)
        
        # Convert to tensor
        tensor = torch.from_numpy(numpy_array)
        if torch_dtype == torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        
        # Reshape
        tensor = tensor.reshape(header.tensor_shape)
        
        return tensor
    
    def _load_tensor_gpu_optimized(self, path: Path, device: torch.device) -> torch.Tensor:
        """Load tensor with optimized GPU transfer using CUDA runtime."""
        # Get tensor info
        info = get_ts_file_info(path)
        
        # If compressed, use fallback
        if info["compressed"]:
            return self._load_tensor_gpu_fallback(path, device)
        
        # Open file with mmap
        f = open(path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        try:
            # Read header
            from ..io import read_ts_header
            header = read_ts_header(path)
            
            # Allocate GPU memory directly
            data_size = header.data_size
            gpu_ptr = self._cuda_malloc(data_size)
            
            # Copy data from mmap to GPU
            data_offset = header.header_size
            host_ptr = ctypes.cast(
                ctypes.addressof(ctypes.c_char.from_buffer(mm, data_offset)),
                ctypes.c_void_p
            )
            
            result = cuda_rt.cudaMemcpy(
                gpu_ptr, host_ptr, data_size, cudaMemcpyHostToDevice
            )
            
            if result != 0:
                self._cuda_free(gpu_ptr)
                raise RuntimeError(f"CUDA memcpy failed with error {result}")
            
            # Create tensor from GPU memory
            torch_dtype = eval(header.tensor_dtype)
            
            # Create tensor using PyTorch's from_blob (with custom deleter)
            # Note: This is a simplified version - production code would need
            # proper memory management integration with PyTorch
            
            # For now, fall back to standard method but keep optimization structure
            return self._load_tensor_gpu_fallback(path, device)
        
        finally:
            mm.close()
            f.close()
    
    def _load_tensor_gpu_fallback(self, path: Path, device: torch.device) -> torch.Tensor:
        """Fallback GPU loading using PyTorch's standard methods."""
        tensor = load_from_ts(path, device=torch.device('cpu'))
        return tensor.to(device)
    
    def _cuda_malloc(self, size: int) -> ctypes.c_void_p:
        """Allocate CUDA memory."""
        if not CUDA_RT_AVAILABLE:
            raise RuntimeError("CUDA runtime not available")
        
        gpu_ptr = ctypes.c_void_p()
        result = cuda_rt.cudaMalloc(ctypes.byref(gpu_ptr), size)
        
        if result != 0:
            raise RuntimeError(f"cudaMalloc failed with error {result}")
        
        self._memory_pool[gpu_ptr.value] = size
        return gpu_ptr
    
    def _cuda_free(self, ptr: ctypes.c_void_p) -> None:
        """Free CUDA memory."""
        if not CUDA_RT_AVAILABLE:
            return
        
        if isinstance(ptr, int):
            ptr_val = ptr
            ptr = ctypes.c_void_p(ptr)
        else:
            ptr_val = ptr.value
        
        if ptr_val in self._memory_pool:
            result = cuda_rt.cudaFree(ptr)
            if result != 0:
                warnings.warn(f"cudaFree failed with error {result}")
            del self._memory_pool[ptr_val]
    
    def unload_tensor(self, tensor: torch.Tensor) -> None:
        """
        Unload a tensor and free its memory.
        
        For mmap backend, this primarily involves freeing GPU memory
        and closing memory-mapped files when appropriate.
        """
        if tensor.device.type == 'cuda':
            # Force garbage collection of GPU memory
            del tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get current memory usage information."""
        info = {}
        
        # System memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.update({
                "system_total": memory.total,
                "system_available": memory.available,
                "system_used": memory.used,
            })
        except ImportError:
            pass
        
        # CUDA memory info
        if torch.cuda.is_available():
            info.update({
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_cached": torch.cuda.memory_reserved(),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(),
            })
            
            # Custom CUDA memory pool info
            if CUDA_RT_AVAILABLE:
                pool_usage = sum(self._memory_pool.values())
                info["cuda_pool_allocated"] = pool_usage
        
        # Memory-mapped files info
        mmap_size = 0
        for mm, _ in self._mmap_files.values():
            mmap_size += mm.size()
        info["mmap_total_size"] = mmap_size
        info["mmap_file_count"] = len(self._mmap_files)
        
        return info
    
    def is_available(self) -> bool:
        """Check if mmap backend is available."""
        # mmap is always available in Python
        return True
    
    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                self.cleanup()
            except Exception:
                pass  # Ignore errors during destruction
