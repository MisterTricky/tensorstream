"""
Base backend interface for TensorStream.

Defines the abstract interface that all backends must implement.
"""

import abc
from typing import Any, Dict, Optional, Union
from pathlib import Path

import torch

from ..exceptions import BackendError


class BackendInterface(abc.ABC):
    """Abstract base class for TensorStream backends."""
    
    def __init__(self, config: Any) -> None:
        """
        Initialize the backend.
        
        Args:
            config: TensorStream configuration object
        """
        self.config = config
        self.name = self.__class__.__name__
        self._initialized = False
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the backend and check for prerequisites."""
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """Cleanup backend resources."""
        pass
    
    @abc.abstractmethod
    def load_tensor(self, path: Union[str, Path], 
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Load a tensor from storage to the specified device.
        
        Args:
            path: Path to the tensor file
            device: Target device for the tensor
            
        Returns:
            Loaded tensor
        """
        pass
    
    @abc.abstractmethod
    def unload_tensor(self, tensor: torch.Tensor) -> None:
        """
        Unload a tensor and free its memory.
        
        Args:
            tensor: Tensor to unload
        """
        pass
    
    @abc.abstractmethod
    def get_memory_info(self) -> Dict[str, int]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory statistics
        """
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on the current system.
        
        Returns:
            True if backend is available
        """
        pass
    
    def get_name(self) -> str:
        """Get the backend name."""
        return self.name
    
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized
    
    def _raise_backend_error(self, operation: str, message: str, 
                           details: Optional[Dict[str, Any]] = None) -> None:
        """Helper to raise a BackendError."""
        raise BackendError(self.name, operation, message, details)
