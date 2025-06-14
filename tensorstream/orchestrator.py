"""
Orchestration Engine for TensorStream.

The orchestration engine is the core component that manages layer states,
coordinates I/O operations, handles memory pressure, and implements
prefetching strategies.
"""

import threading
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import queue

import torch

from .config import Config, BackendType, PrefetchStrategy, MemoryPressureMode
from .exceptions import OrchestrationError, MemoryError as TSMemoryError
from .backends import BackendInterface
from .backends.mmap_backend import MmapBackend
from .backends.cuda_backend import CudaCoreBackend
from .backends.gpudirect_backend import GPUDirectBackend


class LayerState(Enum):
    """Possible states for a layer."""
    DISK = "disk"           # On storage only
    LOADING = "loading"     # Currently being loaded
    RAM = "ram"            # In system memory
    TRANSFERRING = "transferring"  # Being transferred to GPU
    VRAM = "vram"          # In GPU memory
    ERROR = "error"        # Error state


class LoadRequest:
    """Represents a request to load a layer."""
    
    def __init__(self, layer_id: str, priority: int = 0, callback: Optional[callable] = None):
        self.layer_id = layer_id
        self.priority = priority
        self.callback = callback
        self.timestamp = time.time()
        self.future: Optional[Future] = None
    
    def __lt__(self, other):
        # Higher priority first, then earlier timestamp
        return (self.priority, -self.timestamp) < (other.priority, -other.timestamp)


class LayerInfo:
    """Information about a layer."""
    
    def __init__(self, layer_id: str, file_path: Path, size_bytes: int):
        self.layer_id = layer_id
        self.file_path = file_path
        self.size_bytes = size_bytes
        self.state = LayerState.DISK
        self.tensor: Optional[torch.Tensor] = None
        self.last_access = time.time()
        self.access_count = 0
        self.load_time = 0.0
        self.error_message = ""


class OrchestrationEngine:
    """
    Core orchestration engine for TensorStream.
    
    Manages layer states, coordinates I/O operations, implements prefetching,
    and handles memory pressure situations.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.backend: Optional[BackendInterface] = None
        
        # Layer management
        self.layers: Dict[str, LayerInfo] = {}
        self.layer_order: List[str] = []  # Execution order
        
        # State tracking
        self.layer_states: Dict[str, LayerState] = {}
        self.vram_usage = 0
        self.vram_budget = int(config.vram_budget_gb * 1024**3)  # Convert to bytes
        
        # Request management
        self.request_queue = queue.PriorityQueue()
        self.active_requests: Dict[str, LoadRequest] = {}
        self.completed_requests: Dict[str, torch.Tensor] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=config.num_io_threads)
        self.orchestrator_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Locks
        self.state_lock = threading.RLock()
        self.memory_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "total_load_time": 0.0,
            "memory_pressure_events": 0,
        }
    
    def initialize(self) -> None:
        """Initialize the orchestration engine."""
        self._initialize_backend()
        self.running = True
        self.orchestrator_thread = threading.Thread(target=self._orchestrator_loop, daemon=True)
        self.orchestrator_thread.start()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False
        
        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            self.orchestrator_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        
        if self.backend:
            self.backend.cleanup()
        
        # Clear all tensors
        with self.state_lock:
            for layer_info in self.layers.values():
                if layer_info.tensor is not None:
                    del layer_info.tensor
                    layer_info.tensor = None
            self.completed_requests.clear()
    
    def _initialize_backend(self) -> None:
        """Initialize the appropriate backend."""
        backend_priority = self.config.get_backend_priority()
        
        for backend_type in backend_priority:
            try:
                if backend_type == BackendType.GPUDIRECT:
                    backend = GPUDirectBackend(self.config)
                elif backend_type == BackendType.CUDA_CORE:
                    backend = CudaCoreBackend(self.config)
                elif backend_type == BackendType.MMAP:
                    backend = MmapBackend(self.config)
                else:
                    continue
                
                if backend.is_available():
                    backend.initialize()
                    self.backend = backend
                    if self.config.debug_mode:
                        print(f"TensorStream: Using {backend.get_name()} backend")
                    return
            
            except Exception as e:
                if self.config.debug_mode:
                    print(f"TensorStream: Failed to initialize {backend_type.value} backend: {e}")
                continue
        
        raise OrchestrationError(
            "initialize",
            "No suitable backend available"
        )
    
    def register_layer(self, layer_id: str, file_path: Path, size_bytes: int) -> None:
        """Register a layer with the orchestrator."""
        with self.state_lock:
            layer_info = LayerInfo(layer_id, file_path, size_bytes)
            self.layers[layer_id] = layer_info
            self.layer_states[layer_id] = LayerState.DISK
    
    def set_layer_order(self, layer_order: List[str]) -> None:
        """Set the execution order of layers."""
        with self.state_lock:
            self.layer_order = layer_order.copy()
    
    def request_layer(self, layer_id: str, priority: int = 0, 
                     callback: Optional[callable] = None) -> torch.Tensor:
        """
        Request a layer to be loaded and available in VRAM.
        
        Args:
            layer_id: ID of the layer to load
            priority: Priority of the request (higher = more urgent)
            callback: Optional callback when loading completes
            
        Returns:
            The loaded tensor
        """
        with self.state_lock:
            if layer_id not in self.layers:
                raise OrchestrationError(
                    "request_layer",
                    f"Layer {layer_id} not registered"
                )
            
            layer_info = self.layers[layer_id]
            layer_info.access_count += 1
            layer_info.last_access = time.time()
            
            # Check if already in VRAM
            if layer_info.state == LayerState.VRAM and layer_info.tensor is not None:
                self.stats["cache_hits"] += 1
                return layer_info.tensor
            
            # Check if already loaded and cached
            if layer_id in self.completed_requests:
                tensor = self.completed_requests[layer_id]
                if tensor.device.type == self.config.device.type:
                    self.stats["cache_hits"] += 1
                    layer_info.tensor = tensor
                    layer_info.state = LayerState.VRAM
                    self._update_vram_usage()
                    return tensor
            
            self.stats["cache_misses"] += 1
            
            # Check if request is already active
            if layer_id in self.active_requests:
                request = self.active_requests[layer_id]
                if request.future:
                    return request.future.result()  # Block until ready
            
            # Create new request
            request = LoadRequest(layer_id, priority, callback)
            self.request_queue.put(request)
            self.active_requests[layer_id] = request
            
            # Submit to executor
            request.future = self.executor.submit(self._load_layer, layer_id)
            
            return request.future.result()  # Block until ready
    
    def _load_layer(self, layer_id: str) -> torch.Tensor:
        """Load a layer from storage."""
        start_time = time.time()
        
        try:
            with self.state_lock:
                layer_info = self.layers[layer_id]
                layer_info.state = LayerState.LOADING
            
            # Ensure memory is available
            self._ensure_memory_available(layer_info.size_bytes)
            
            # Load tensor using backend
            tensor = self.backend.load_tensor(layer_info.file_path, self.config.device)
            
            with self.state_lock:
                layer_info.tensor = tensor
                layer_info.state = LayerState.VRAM
                layer_info.load_time = time.time() - start_time
                
                # Update statistics
                self.stats["total_loads"] += 1
                self.stats["total_load_time"] += layer_info.load_time
                
                # Cache the result
                self.completed_requests[layer_id] = tensor
                
                # Remove from active requests
                if layer_id in self.active_requests:
                    del self.active_requests[layer_id]
                
                self._update_vram_usage()
            
            # Trigger prefetching
            self._trigger_prefetch(layer_id)
            
            return tensor
        
        except Exception as e:
            with self.state_lock:
                layer_info = self.layers[layer_id]
                layer_info.state = LayerState.ERROR
                layer_info.error_message = str(e)
                
                if layer_id in self.active_requests:
                    del self.active_requests[layer_id]
            
            raise OrchestrationError(
                "load_layer",
                f"Failed to load layer {layer_id}: {e}"
            )
    
    def _ensure_memory_available(self, required_bytes: int) -> None:
        """Ensure sufficient VRAM is available."""
        with self.memory_lock:
            if self.vram_usage + required_bytes <= self.vram_budget:
                return  # Sufficient memory available
            
            # Memory pressure situation
            self.stats["memory_pressure_events"] += 1
            
            if self.config.memory_pressure_mode == MemoryPressureMode.STRICT:
                available = self.vram_budget - self.vram_usage
                raise TSMemoryError(
                    "VRAM",
                    required_bytes,
                    available,
                    {"budget": self.vram_budget, "current_usage": self.vram_usage}
                )
            
            # Adaptive or lenient mode - try to free memory
            bytes_to_free = (self.vram_usage + required_bytes) - self.vram_budget
            self._evict_layers(bytes_to_free)
    
    def _evict_layers(self, bytes_to_free: int) -> None:
        """Evict layers to free memory."""
        with self.state_lock:
            # Sort layers by eviction priority (LRU with access count consideration)
            eviction_candidates = []
            for layer_id, layer_info in self.layers.items():
                if (layer_info.state == LayerState.VRAM and 
                    layer_info.tensor is not None and
                    layer_id not in self.active_requests):
                    
                    # Score based on last access time and access frequency
                    score = layer_info.last_access - (layer_info.access_count * 100)
                    eviction_candidates.append((score, layer_id, layer_info))
            
            # Sort by score (lower = evict first)
            eviction_candidates.sort(key=lambda x: x[0])
            
            bytes_freed = 0
            for _, layer_id, layer_info in eviction_candidates:
                if bytes_freed >= bytes_to_free:
                    break
                
                # Evict the layer
                if layer_info.tensor is not None:
                    if self.backend:
                        self.backend.unload_tensor(layer_info.tensor)
                    
                    bytes_freed += layer_info.size_bytes
                    layer_info.tensor = None
                    layer_info.state = LayerState.DISK
                    
                    # Remove from cache
                    if layer_id in self.completed_requests:
                        del self.completed_requests[layer_id]
                    
                    self.stats["evictions"] += 1
                    
                    if self.config.debug_mode:
                        print(f"TensorStream: Evicted layer {layer_id} "
                              f"({layer_info.size_bytes} bytes)")
            
            self._update_vram_usage()
            
            if bytes_freed < bytes_to_free:
                warnings.warn(
                    f"Could only free {bytes_freed} bytes of requested {bytes_to_free} bytes"
                )
    
    def _update_vram_usage(self) -> None:
        """Update VRAM usage tracking."""
        usage = 0
        for layer_info in self.layers.values():
            if (layer_info.state == LayerState.VRAM and 
                layer_info.tensor is not None):
                usage += layer_info.size_bytes
        
        self.vram_usage = usage
    
    def _trigger_prefetch(self, current_layer_id: str) -> None:
        """Trigger prefetching based on the strategy."""
        if self.config.prefetch_strategy == PrefetchStrategy.NONE:
            return
        
        try:
            current_idx = self.layer_order.index(current_layer_id)
        except ValueError:
            return  # Layer not in execution order
        
        candidates = []
        
        if self.config.prefetch_strategy == PrefetchStrategy.NEXT_LAYER:
            if current_idx + 1 < len(self.layer_order):
                candidates = [self.layer_order[current_idx + 1]]
        
        elif self.config.prefetch_strategy == PrefetchStrategy.ADAPTIVE:
            # Prefetch next 1-2 layers based on available memory
            available_budget = self.vram_budget - self.vram_usage
            for i in range(1, min(3, len(self.layer_order) - current_idx)):
                next_layer_id = self.layer_order[current_idx + i]
                next_layer_info = self.layers.get(next_layer_id)
                
                if (next_layer_info and 
                    next_layer_info.state == LayerState.DISK and
                    next_layer_info.size_bytes <= available_budget):
                    candidates.append(next_layer_id)
                    available_budget -= next_layer_info.size_bytes
        
        elif self.config.prefetch_strategy == PrefetchStrategy.AGGRESSIVE:
            # Prefetch as many upcoming layers as memory allows
            available_budget = self.vram_budget - self.vram_usage
            for i in range(1, len(self.layer_order) - current_idx):
                next_layer_id = self.layer_order[current_idx + i]
                next_layer_info = self.layers.get(next_layer_id)
                
                if (next_layer_info and 
                    next_layer_info.state == LayerState.DISK and
                    next_layer_info.size_bytes <= available_budget):
                    candidates.append(next_layer_id)
                    available_budget -= next_layer_info.size_bytes
                else:
                    break
        
        # Submit prefetch requests with lower priority
        for layer_id in candidates:
            if layer_id not in self.active_requests:
                request = LoadRequest(layer_id, priority=-1)  # Low priority
                self.request_queue.put(request)
                self.active_requests[layer_id] = request
                request.future = self.executor.submit(self._load_layer, layer_id)
    
    def _orchestrator_loop(self) -> None:
        """Main orchestrator loop for background operations."""
        while self.running:
            try:
                # Process any pending maintenance tasks
                self._cleanup_completed_requests()
                self._update_statistics()
                
                time.sleep(0.1)  # Small sleep to prevent busy waiting
            
            except Exception as e:
                if self.config.debug_mode:
                    print(f"TensorStream orchestrator error: {e}")
    
    def _cleanup_completed_requests(self) -> None:
        """Clean up old completed requests to free memory."""
        with self.state_lock:
            # Keep only recent requests (last 100)
            if len(self.completed_requests) > 100:
                # Sort by access time and keep most recent
                sorted_layers = sorted(
                    self.layers.items(),
                    key=lambda x: x[1].last_access,
                    reverse=True
                )
                
                to_keep = set(layer_id for layer_id, _ in sorted_layers[:50])
                
                for layer_id in list(self.completed_requests.keys()):
                    if layer_id not in to_keep:
                        del self.completed_requests[layer_id]
    
    def _update_statistics(self) -> None:
        """Update internal statistics."""
        # This method can be extended to compute additional metrics
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        with self.state_lock:
            stats = self.stats.copy()
            
            # Add current state information
            state_counts = defaultdict(int)
            for layer_info in self.layers.values():
                state_counts[layer_info.state.value] += 1
            
            stats.update({
                "layer_states": dict(state_counts),
                "vram_usage_bytes": self.vram_usage,
                "vram_budget_bytes": self.vram_budget,
                "vram_utilization": self.vram_usage / self.vram_budget if self.vram_budget > 0 else 0,
                "active_requests": len(self.active_requests),
                "cached_layers": len(self.completed_requests),
                "total_layers": len(self.layers),
            })
            
            # Add backend statistics
            if self.backend:
                try:
                    backend_stats = self.backend.get_memory_info()
                    stats["backend"] = backend_stats
                except Exception:
                    pass
            
            return stats
    
    def get_layer_info(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific layer."""
        with self.state_lock:
            layer_info = self.layers.get(layer_id)
            if not layer_info:
                return None
            
            return {
                "layer_id": layer_info.layer_id,
                "file_path": str(layer_info.file_path),
                "size_bytes": layer_info.size_bytes,
                "state": layer_info.state.value,
                "last_access": layer_info.last_access,
                "access_count": layer_info.access_count,
                "load_time": layer_info.load_time,
                "has_tensor": layer_info.tensor is not None,
                "error_message": layer_info.error_message,
            }
