"""
Configuration management for TensorStream.

Provides comprehensive configuration options for tensor streaming behavior,
memory management, and backend selection.
"""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch

from .exceptions import ConfigurationError


class BackendType(Enum):
    """Available backend types for tensor streaming."""
    AUTO = "auto"
    MMAP = "mmap" 
    GPUDIRECT = "gpudirect"
    CUDA_CORE = "cuda_core"


class PrefetchStrategy(Enum):
    """Prefetching strategies for layer loading."""
    NONE = "none"
    NEXT_LAYER = "next_layer"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"


class MemoryPressureMode(Enum):
    """Memory pressure handling modes."""
    STRICT = "strict"        # Fail if budget exceeded
    ADAPTIVE = "adaptive"    # Dynamically adjust strategy
    LENIENT = "lenient"      # Allow temporary overruns


@dataclass
class Config:
    """
    Configuration class for TensorStream operations.
    
    This class encapsulates all configuration options for tensor streaming,
    including memory budgets, backend selection, prefetching strategies,
    and storage options.
    
    Args:
        storage_path: Path where tensor shards will be stored
        vram_budget_gb: Maximum VRAM usage in GB (default: auto-detect)
        backend: Backend type to use for I/O operations
        prefetch_strategy: Strategy for prefetching layers
        memory_pressure_mode: How to handle memory pressure
        cache_size_layers: Number of layers to keep in cache
        compression_enabled: Whether to compress tensor data
        compression_level: Compression level (1-9, if compression enabled)
        num_io_threads: Number of I/O threads for parallel loading
        chunk_size_mb: Size of I/O chunks in MB
        verify_checksums: Whether to verify data integrity
        enable_profiling: Whether to enable performance profiling
        device: Default device for tensor operations
        dtype: Default data type for tensors
        pin_memory: Whether to use pinned memory for transfers
        debug_mode: Enable debug logging and checks
        temp_dir: Temporary directory for intermediate files
        metadata: Additional user-defined metadata
    """
    
    # Core configuration
    storage_path: Union[str, Path]
    vram_budget_gb: Optional[float] = None
    backend: BackendType = BackendType.AUTO
    
    # Memory management
    prefetch_strategy: PrefetchStrategy = PrefetchStrategy.NEXT_LAYER
    memory_pressure_mode: MemoryPressureMode = MemoryPressureMode.ADAPTIVE
    cache_size_layers: int = 2
    
    # Compression settings
    compression_enabled: bool = False
    compression_level: int = 6
    
    # I/O settings
    num_io_threads: int = 4
    chunk_size_mb: int = 64
    verify_checksums: bool = True
    
    # Performance and debugging
    enable_profiling: bool = False
    pin_memory: bool = True
    debug_mode: bool = False
    
    # Device and tensor settings
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    
    # Advanced options
    temp_dir: Optional[Union[str, Path]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        self._validate_and_normalize()
    
    def _validate_and_normalize(self) -> None:
        """Validate configuration parameters and set defaults."""
        # Normalize storage path
        self.storage_path = Path(self.storage_path).resolve()
        
        # Validate storage path
        if not self.storage_path.exists():
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ConfigurationError(
                    "storage_path",
                    f"Cannot create storage directory: {e}",
                    {"path": str(self.storage_path)}
                )
        
        if not self.storage_path.is_dir():
            raise ConfigurationError(
                "storage_path",
                "Storage path must be a directory",
                {"path": str(self.storage_path)}
            )
        
        # Auto-detect VRAM budget if not specified
        if self.vram_budget_gb is None:
            self.vram_budget_gb = self._auto_detect_vram_budget()
        
        # Validate VRAM budget
        if self.vram_budget_gb <= 0:
            raise ConfigurationError(
                "vram_budget_gb",
                "VRAM budget must be positive",
                {"value": self.vram_budget_gb}
            )
        
        # Validate backend type
        if isinstance(self.backend, str):
            try:
                self.backend = BackendType(self.backend)
            except ValueError:
                valid_backends = [b.value for b in BackendType]
                raise ConfigurationError(
                    "backend",
                    f"Invalid backend type. Valid options: {valid_backends}",
                    {"value": self.backend}
                )
        
        # Validate prefetch strategy
        if isinstance(self.prefetch_strategy, str):
            try:
                self.prefetch_strategy = PrefetchStrategy(self.prefetch_strategy)
            except ValueError:
                valid_strategies = [s.value for s in PrefetchStrategy]
                raise ConfigurationError(
                    "prefetch_strategy",
                    f"Invalid prefetch strategy. Valid options: {valid_strategies}",
                    {"value": self.prefetch_strategy}
                )
        
        # Validate memory pressure mode
        if isinstance(self.memory_pressure_mode, str):
            try:
                self.memory_pressure_mode = MemoryPressureMode(self.memory_pressure_mode)
            except ValueError:
                valid_modes = [m.value for m in MemoryPressureMode]
                raise ConfigurationError(
                    "memory_pressure_mode",
                    f"Invalid memory pressure mode. Valid options: {valid_modes}",
                    {"value": self.memory_pressure_mode}
                )
        
        # Validate numeric parameters
        if self.cache_size_layers < 1:
            raise ConfigurationError(
                "cache_size_layers",
                "Cache size must be at least 1",
                {"value": self.cache_size_layers}
            )
        
        if self.compression_enabled:
            if not (1 <= self.compression_level <= 9):
                raise ConfigurationError(
                    "compression_level",
                    "Compression level must be between 1 and 9",
                    {"value": self.compression_level}
                )
        
        if self.num_io_threads < 1:
            raise ConfigurationError(
                "num_io_threads",
                "Number of I/O threads must be at least 1",
                {"value": self.num_io_threads}
            )
        
        if self.chunk_size_mb < 1:
            raise ConfigurationError(
                "chunk_size_mb",
                "Chunk size must be at least 1 MB",
                {"value": self.chunk_size_mb}
            )
        
        # Set default device if not specified
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        # Set temp directory
        if self.temp_dir is None:
            self.temp_dir = self.storage_path / "tmp"
        else:
            self.temp_dir = Path(self.temp_dir)
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _auto_detect_vram_budget(self) -> float:
        """Auto-detect available VRAM budget."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, setting VRAM budget to 0")
            return 0.0
        
        try:
            # Get total VRAM and reserve some for PyTorch overhead
            total_vram = torch.cuda.get_device_properties(0).total_memory
            total_vram_gb = total_vram / (1024**3)
            
            # Reserve 20% for PyTorch overhead and other operations
            available_gb = total_vram_gb * 0.8
            
            if self.debug_mode:
                print(f"Auto-detected VRAM budget: {available_gb:.1f} GB "
                      f"(total: {total_vram_gb:.1f} GB)")
            
            return available_gb
        
        except Exception as e:
            warnings.warn(f"Failed to auto-detect VRAM: {e}, defaulting to 4GB")
            return 4.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (BackendType, PrefetchStrategy, MemoryPressureMode)):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, torch.device):
                result[key] = str(value)
            elif isinstance(value, torch.dtype):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        # Convert string enums back to enum objects
        if "backend" in config_dict and isinstance(config_dict["backend"], str):
            config_dict["backend"] = BackendType(config_dict["backend"])
        
        if "prefetch_strategy" in config_dict and isinstance(config_dict["prefetch_strategy"], str):
            config_dict["prefetch_strategy"] = PrefetchStrategy(config_dict["prefetch_strategy"])
        
        if "memory_pressure_mode" in config_dict and isinstance(config_dict["memory_pressure_mode"], str):
            config_dict["memory_pressure_mode"] = MemoryPressureMode(config_dict["memory_pressure_mode"])
        
        # Convert device string back to torch.device
        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = torch.device(config_dict["device"])
        
        # Convert dtype string back to torch.dtype
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            config_dict["dtype"] = getattr(torch, config_dict["dtype"].split(".")[-1])
        
        return cls(**config_dict)
    
    def get_backend_priority(self) -> List[BackendType]:
        """Get backend priority list based on configuration."""
        if self.backend == BackendType.AUTO:
            # Auto-selection priority
            priority = []
            
            # Check for GPUDirect availability
            try:
                from . import _gds_core
                priority.append(BackendType.GPUDIRECT)
            except ImportError:
                pass
            
            # Check for CUDA core availability
            try:
                from . import _cuda_core
                priority.append(BackendType.CUDA_CORE)
            except ImportError:
                pass
            
            # Always have mmap as fallback
            priority.append(BackendType.MMAP)
            
            return priority
        else:
            return [self.backend]
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"Config(storage_path={self.storage_path}, "
                f"vram_budget_gb={self.vram_budget_gb}, "
                f"backend={self.backend.value}, "
                f"prefetch_strategy={self.prefetch_strategy.value})")


def create_default_config(storage_path: Union[str, Path]) -> Config:
    """Create a default configuration with sensible defaults."""
    return Config(storage_path=storage_path)
