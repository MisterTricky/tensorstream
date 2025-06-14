"""
TensorStream exception hierarchy.

Provides a comprehensive set of exceptions for different error conditions
that can occur during tensor streaming operations.
"""

from typing import Optional, Any


class TensorStreamError(Exception):
    """Base exception class for all TensorStream errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class BackendError(TensorStreamError):
    """Raised when a backend operation fails."""
    
    def __init__(self, backend_name: str, operation: str, message: str, 
                 details: Optional[dict] = None) -> None:
        full_message = f"Backend '{backend_name}' failed during '{operation}': {message}"
        super().__init__(full_message, details)
        self.backend_name = backend_name
        self.operation = operation


class ConfigurationError(TensorStreamError):
    """Raised when configuration is invalid or inconsistent."""
    
    def __init__(self, config_field: str, message: str, 
                 details: Optional[dict] = None) -> None:
        full_message = f"Configuration error in '{config_field}': {message}"
        super().__init__(full_message, details)
        self.config_field = config_field


class MemoryError(TensorStreamError):
    """Raised when memory allocation or management fails."""
    
    def __init__(self, memory_type: str, requested_bytes: int, 
                 available_bytes: Optional[int] = None, 
                 details: Optional[dict] = None) -> None:
        if available_bytes is not None:
            message = (f"Failed to allocate {requested_bytes} bytes of {memory_type} "
                      f"memory (available: {available_bytes} bytes)")
        else:
            message = f"Failed to allocate {requested_bytes} bytes of {memory_type} memory"
        
        super().__init__(message, details)
        self.memory_type = memory_type
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes


class StorageError(TensorStreamError):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, path: str, message: str, 
                 details: Optional[dict] = None) -> None:
        full_message = f"Storage {operation} failed for '{path}': {message}"
        super().__init__(full_message, details)
        self.operation = operation
        self.path = path


class LayerError(TensorStreamError):
    """Raised when layer operations fail."""
    
    def __init__(self, layer_name: str, operation: str, message: str,
                 details: Optional[dict] = None) -> None:
        full_message = f"Layer '{layer_name}' failed during '{operation}': {message}"
        super().__init__(full_message, details)
        self.layer_name = layer_name
        self.operation = operation


class OrchestrationError(TensorStreamError):
    """Raised when orchestration operations fail."""
    
    def __init__(self, operation: str, message: str, 
                 details: Optional[dict] = None) -> None:
        full_message = f"Orchestration error during '{operation}': {message}"
        super().__init__(full_message, details)
        self.operation = operation
