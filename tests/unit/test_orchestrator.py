"""
Unit tests for TensorStream orchestrator module.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from tensorstream.orchestrator import (
    LayerState, TensorStreamOrchestrator, 
    MemoryPressureMonitor, PrefetchStrategy
)
from tensorstream.config import TensorStreamConfig, PrefetchMode, BackendType
from tensorstream.exceptions import OrchestratorError, BackendError, TensorStreamError
from tensorstream.backends import BackendInterface


class TestLayerState:
    """Test LayerState enum functionality."""

    def test_layer_state_values(self):
        """Test LayerState enum values."""
        assert LayerState.DISK == "disk"
        assert LayerState.LOADING == "loading" 
        assert LayerState.MEMORY == "memory"
        assert LayerState.ERROR == "error"

    def test_layer_state_transitions(self):
        """Test valid layer state transitions."""
        # Valid transitions: disk -> loading -> memory
        # Error can come from any state
        valid_transitions = {
            LayerState.DISK: [LayerState.LOADING, LayerState.ERROR],
            LayerState.LOADING: [LayerState.MEMORY, LayerState.ERROR, LayerState.DISK],
            LayerState.MEMORY: [LayerState.DISK, LayerState.ERROR],
            LayerState.ERROR: [LayerState.DISK, LayerState.LOADING]
        }
        
        for from_state, to_states in valid_transitions.items():
            for to_state in to_states:
                # This is just documenting expected transitions
                assert from_state in LayerState
                assert to_state in LayerState


class TestMemoryPressureMonitor:
    """Test MemoryPressureMonitor functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryPressureMonitor(threshold=0.8, check_interval=1.0)
        assert monitor.threshold == 0.8
        assert monitor.check_interval == 1.0
        assert not monitor.running
        assert monitor._thread is None
        assert len(monitor._callbacks) == 0

    def test_monitor_register_callback(self):
        """Test registering pressure callbacks."""
        monitor = MemoryPressureMonitor()
        callback = Mock()
        
        monitor.register_callback(callback)
        assert callback in monitor._callbacks
        
        # Test duplicate registration
        monitor.register_callback(callback)
        assert monitor._callbacks.count(callback) == 1

    def test_monitor_unregister_callback(self):
        """Test unregistering pressure callbacks."""
        monitor = MemoryPressureMonitor()
        callback = Mock()
        
        # Unregister non-existent callback
        monitor.unregister_callback(callback)
        assert callback not in monitor._callbacks
        
        # Register and unregister
        monitor.register_callback(callback)
        monitor.unregister_callback(callback)
        assert callback not in monitor._callbacks

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    def test_monitor_get_memory_usage(self, mock_max_mem, mock_current_mem):
        """Test memory usage calculation."""
        mock_current_mem.return_value = 1024 * 1024 * 1024  # 1GB
        mock_max_mem.return_value = 2048 * 1024 * 1024      # 2GB
        
        monitor = MemoryPressureMonitor()
        usage = monitor.get_memory_usage()
        
        assert usage == 0.5  # 1GB / 2GB
        mock_current_mem.assert_called_once()
        mock_max_mem.assert_called_once()

    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    def test_monitor_check_pressure(self, mock_max_mem, mock_current_mem):
        """Test pressure check and callback invocation."""
        mock_current_mem.return_value = 1800 * 1024 * 1024  # 1.8GB
        mock_max_mem.return_value = 2048 * 1024 * 1024      # 2GB
        
        monitor = MemoryPressureMonitor(threshold=0.8)  # 80%
        callback = Mock()
        monitor.register_callback(callback)
        
        # Should trigger callback (1.8/2 = 0.875 > 0.8)
        monitor._check_pressure()
        callback.assert_called_once_with(0.875)
        
        # Reset and test below threshold
        callback.reset_mock()
        mock_current_mem.return_value = 1536 * 1024 * 1024  # 1.5GB
        
        monitor._check_pressure()
        callback.assert_not_called()

    def test_monitor_start_stop(self):
        """Test monitor start/stop functionality."""
        monitor = MemoryPressureMonitor(check_interval=0.1)
        
        # Start monitor
        monitor.start()
        assert monitor.running
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        
        # Stop monitor
        monitor.stop()
        assert not monitor.running
        
        # Wait for thread to finish
        monitor._thread.join(timeout=1.0)
        assert not monitor._thread.is_alive()

    def test_monitor_context_manager(self):
        """Test monitor as context manager."""
        monitor = MemoryPressureMonitor(check_interval=0.1)
        
        with monitor:
            assert monitor.running
            assert monitor._thread is not None
        
        assert not monitor.running


class TestPrefetchStrategy:
    """Test PrefetchStrategy functionality."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = PrefetchStrategy()
        assert strategy.look_ahead == 1
        assert strategy.mode == PrefetchMode.SEQUENTIAL

    def test_strategy_predict_next_layers_sequential(self):
        """Test sequential prefetch prediction."""
        strategy = PrefetchStrategy(mode=PrefetchMode.SEQUENTIAL, look_ahead=2)
        
        # Test with linear sequence
        access_history = ["layer1", "layer2", "layer3"]
        predictions = strategy.predict_next_layers(access_history)
        
        assert predictions == ["layer4", "layer5"]

    def test_strategy_predict_next_layers_adaptive(self):
        """Test adaptive prefetch prediction."""
        strategy = PrefetchStrategy(mode=PrefetchMode.ADAPTIVE, look_ahead=2)
        
        # Test with pattern recognition
        access_history = ["layer1", "layer2", "layer1", "layer2", "layer1"]
        predictions = strategy.predict_next_layers(access_history)
        
        # Should predict continuation of pattern
        assert "layer2" in predictions

    def test_strategy_predict_next_layers_disabled(self):
        """Test disabled prefetch prediction."""
        strategy = PrefetchStrategy(mode=PrefetchMode.DISABLED)
        
        access_history = ["layer1", "layer2", "layer3"]
        predictions = strategy.predict_next_layers(access_history)
        
        assert predictions == []

    def test_strategy_update_access_pattern(self):
        """Test access pattern update."""
        strategy = PrefetchStrategy()
        
        strategy.update_access_pattern("layer1")
        strategy.update_access_pattern("layer2")
        
        assert strategy.access_history == ["layer1", "layer2"]

    def test_strategy_access_history_limit(self):
        """Test access history size limiting."""
        strategy = PrefetchStrategy()
        
        # Fill beyond history limit
        for i in range(150):
            strategy.update_access_pattern(f"layer{i}")
        
        # Should be truncated to 100
        assert len(strategy.access_history) == 100
        assert strategy.access_history[0] == "layer50"
        assert strategy.access_history[-1] == "layer149"


class TestTensorStreamOrchestrator:
    """Test TensorStreamOrchestrator functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        backend = Mock(spec=BackendInterface)
        backend.load_tensor = Mock()
        backend.store_tensor = Mock()
        backend.remove_tensor = Mock()
        backend.get_memory_usage = Mock(return_value=0.5)
        return backend

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=1024 * 1024 * 1024,  # 1GB
            prefetch_mode=PrefetchMode.SEQUENTIAL,
            backend_type=BackendType.MMAP
        )

    @pytest.fixture
    def orchestrator(self, mock_backend, config):
        """Create test orchestrator."""
        orch = TensorStreamOrchestrator(config, mock_backend)
        return orch

    def test_orchestrator_initialization(self, orchestrator, mock_backend, config):
        """Test orchestrator initialization."""
        assert orchestrator.config == config
        assert orchestrator.backend == mock_backend
        assert len(orchestrator.layer_states) == 0
        assert len(orchestrator.layer_metadata) == 0
        assert len(orchestrator.lru_queue) == 0
        assert orchestrator.memory_pressure_monitor is not None
        assert orchestrator.prefetch_strategy is not None

    def test_orchestrator_register_layer(self, orchestrator):
        """Test layer registration."""
        layer_id = "test_layer"
        file_path = Path("/tmp/test.ts")
        metadata = {"shape": [1024, 1024], "dtype": "float32"}
        
        orchestrator.register_layer(layer_id, file_path, metadata)
        
        assert layer_id in orchestrator.layer_states
        assert orchestrator.layer_states[layer_id] == LayerState.DISK
        assert layer_id in orchestrator.layer_metadata
        assert orchestrator.layer_metadata[layer_id]["file_path"] == file_path
        assert orchestrator.layer_metadata[layer_id]["metadata"] == metadata

    def test_orchestrator_register_layer_duplicate(self, orchestrator):
        """Test duplicate layer registration."""
        layer_id = "test_layer"
        file_path = Path("/tmp/test.ts")
        
        orchestrator.register_layer(layer_id, file_path, {})
        
        # Should not raise error, but update metadata
        new_path = Path("/tmp/test2.ts")
        orchestrator.register_layer(layer_id, new_path, {})
        
        assert orchestrator.layer_metadata[layer_id]["file_path"] == new_path

    @pytest.mark.asyncio
    async def test_orchestrator_load_layer_success(self, orchestrator, mock_backend):
        """Test successful layer loading."""
        layer_id = "test_layer"
        file_path = Path("/tmp/test.ts")
        test_tensor = torch.randn(2, 2)
        
        mock_backend.load_tensor.return_value = test_tensor
        
        orchestrator.register_layer(layer_id, file_path, {})
        result = await orchestrator.load_layer(layer_id)
        
        assert torch.equal(result, test_tensor)
        assert orchestrator.layer_states[layer_id] == LayerState.MEMORY
        assert layer_id in orchestrator.lru_queue
        mock_backend.load_tensor.assert_called_once_with(file_path)

    @pytest.mark.asyncio
    async def test_orchestrator_load_layer_not_registered(self, orchestrator):
        """Test loading unregistered layer."""
        with pytest.raises(OrchestratorError, match="not registered"):
            await orchestrator.load_layer("nonexistent_layer")

    @pytest.mark.asyncio
    async def test_orchestrator_load_layer_backend_error(self, orchestrator, mock_backend):
        """Test layer loading with backend error."""
        layer_id = "test_layer"
        file_path = Path("/tmp/test.ts")
        
        mock_backend.load_tensor.side_effect = BackendError("Load failed")
        
        orchestrator.register_layer(layer_id, file_path, {})
        
        with pytest.raises(OrchestratorError, match="Failed to load layer"):
            await orchestrator.load_layer(layer_id)
        
        assert orchestrator.layer_states[layer_id] == LayerState.ERROR

    @pytest.mark.asyncio
    async def test_orchestrator_load_layer_already_loading(self, orchestrator, mock_backend):
        """Test loading layer that's already being loaded."""
        layer_id = "test_layer"
        file_path = Path("/tmp/test.ts")
        
        # Simulate slow loading
        async def slow_load(*args):
            await asyncio.sleep(0.1)
            return torch.randn(2, 2)
        
        mock_backend.load_tensor.side_effect = slow_load
        
        orchestrator.register_layer(layer_id, file_path, {})
        
        # Start first load
        task1 = asyncio.create_task(orchestrator.load_layer(layer_id))
        
        # Start second load while first is in progress
        task2 = asyncio.create_task(orchestrator.load_layer(layer_id))
        
        result1, result2 = await asyncio.gather(task1, task2)
        
        # Both should return the same tensor
        assert torch.equal(result1, result2)
        # Backend should only be called once
        assert mock_backend.load_tensor.call_count == 1

    def test_orchestrator_evict_layer(self, orchestrator, mock_backend):
        """Test layer eviction."""
        layer_id = "test_layer"
        orchestrator.layer_states[layer_id] = LayerState.MEMORY
        orchestrator.lru_queue.append(layer_id)
        
        orchestrator._evict_layer(layer_id)
        
        assert orchestrator.layer_states[layer_id] == LayerState.DISK
        assert layer_id not in orchestrator.lru_queue
        mock_backend.remove_tensor.assert_called_once_with(layer_id)

    def test_orchestrator_update_lru(self, orchestrator):
        """Test LRU queue updates."""
        layer_id = "test_layer"
        
        # First access
        orchestrator._update_lru(layer_id)
        assert orchestrator.lru_queue == [layer_id]
        
        # Add another layer
        orchestrator._update_lru("layer2")
        assert orchestrator.lru_queue == [layer_id, "layer2"]
        
        # Access first layer again
        orchestrator._update_lru(layer_id)
        assert orchestrator.lru_queue == ["layer2", layer_id]

    def test_orchestrator_memory_pressure_callback(self, orchestrator, mock_backend):
        """Test memory pressure handling."""
        # Setup layers in memory
        for i in range(3):
            layer_id = f"layer{i}"
            orchestrator.layer_states[layer_id] = LayerState.MEMORY
            orchestrator.lru_queue.append(layer_id)
        
        # Trigger memory pressure
        orchestrator._handle_memory_pressure(0.9)
        
        # Should evict LRU layer
        assert orchestrator.layer_states["layer0"] == LayerState.DISK
        assert "layer0" not in orchestrator.lru_queue
        mock_backend.remove_tensor.assert_called_with("layer0")

    @pytest.mark.asyncio
    async def test_orchestrator_prefetch_layers(self, orchestrator, mock_backend):
        """Test layer prefetching."""
        # Setup layers
        for i in range(3):
            layer_id = f"layer{i}"
            file_path = Path(f"/tmp/test{i}.ts")
            orchestrator.register_layer(layer_id, file_path, {})
        
        # Mock successful loading
        mock_backend.load_tensor.return_value = torch.randn(2, 2)
        
        # Trigger prefetch
        await orchestrator._prefetch_layers(["layer1", "layer2"])
        
        # Should load both layers
        assert mock_backend.load_tensor.call_count == 2
        assert orchestrator.layer_states["layer1"] == LayerState.MEMORY
        assert orchestrator.layer_states["layer2"] == LayerState.MEMORY

    def test_orchestrator_get_statistics(self, orchestrator):
        """Test statistics gathering."""
        # Setup some layers
        orchestrator.layer_states = {
            "layer1": LayerState.MEMORY,
            "layer2": LayerState.DISK,
            "layer3": LayerState.LOADING,
            "layer4": LayerState.ERROR
        }
        
        stats = orchestrator.get_statistics()
        
        assert stats["total_layers"] == 4
        assert stats["layers_in_memory"] == 1
        assert stats["layers_on_disk"] == 1
        assert stats["layers_loading"] == 1
        assert stats["layers_in_error"] == 1
        assert "cache_hit_rate" in stats
        assert "memory_usage" in stats

    def test_orchestrator_start_stop(self, orchestrator):
        """Test orchestrator start/stop."""
        orchestrator.start()
        assert orchestrator.memory_pressure_monitor.running
        
        orchestrator.stop()
        assert not orchestrator.memory_pressure_monitor.running

    def test_orchestrator_context_manager(self, orchestrator):
        """Test orchestrator as context manager."""
        with orchestrator:
            assert orchestrator.memory_pressure_monitor.running
        
        assert not orchestrator.memory_pressure_monitor.running


class TestOrchestratorIntegration:
    """Integration tests for orchestrator components."""

    @pytest.fixture
    def real_config(self):
        """Create real configuration for integration tests."""
        return TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=100 * 1024 * 1024,  # 100MB
            prefetch_mode=PrefetchMode.SEQUENTIAL,
            backend_type=BackendType.MMAP
        )

    @pytest.mark.asyncio
    async def test_orchestrator_full_workflow(self, real_config):
        """Test complete orchestrator workflow."""
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.io import save_tensor, load_tensor
        
        backend = MmapBackend(real_config)
        orchestrator = TensorStreamOrchestrator(real_config, backend)
        
        # Create test tensor and save to disk
        test_tensor = torch.randn(100, 100)
        file_path = real_config.cache_dir / "test_tensor.ts"
        save_tensor(test_tensor, file_path)
        
        try:
            with orchestrator:
                # Register and load layer
                orchestrator.register_layer("test_layer", file_path, {})
                loaded_tensor = await orchestrator.load_layer("test_layer")
                
                # Verify tensor integrity
                assert torch.allclose(loaded_tensor, test_tensor)
                assert orchestrator.layer_states["test_layer"] == LayerState.MEMORY
                
                # Test statistics
                stats = orchestrator.get_statistics()
                assert stats["total_layers"] == 1
                assert stats["layers_in_memory"] == 1
                
        finally:
            # Cleanup
            if file_path.exists():
                file_path.unlink()

    @pytest.mark.asyncio 
    async def test_orchestrator_memory_pressure_integration(self, real_config):
        """Test memory pressure handling integration."""
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.io import save_tensor
        
        # Use very small memory limit to trigger pressure
        real_config.memory_limit = 1024  # 1KB
        
        backend = MmapBackend(real_config)
        orchestrator = TensorStreamOrchestrator(real_config, backend)
        
        # Create multiple test tensors
        tensors = []
        file_paths = []
        
        for i in range(3):
            tensor = torch.randn(10, 10)  # Each ~400 bytes
            file_path = real_config.cache_dir / f"test_tensor_{i}.ts"
            save_tensor(tensor, file_path)
            
            tensors.append(tensor)
            file_paths.append(file_path)
        
        try:
            with orchestrator:
                # Register all layers
                for i, file_path in enumerate(file_paths):
                    orchestrator.register_layer(f"layer{i}", file_path, {})
                
                # Load all layers - should trigger eviction
                for i in range(3):
                    await orchestrator.load_layer(f"layer{i}")
                    
                    # Check that we don't exceed memory limit significantly
                    stats = orchestrator.get_statistics()
                    # Allow some overhead but should evict when necessary
                    
                # Verify at least some eviction occurred
                in_memory = sum(1 for state in orchestrator.layer_states.values() 
                               if state == LayerState.MEMORY)
                assert in_memory < 3  # Not all layers should be in memory
                
        finally:
            # Cleanup
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()


@pytest.mark.performance
class TestOrchestratorPerformance:
    """Performance tests for orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_concurrent_access(self):
        """Test orchestrator under concurrent access."""
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=1024 * 1024 * 1024,  # 1GB
            backend_type=BackendType.MMAP
        )
        
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.io import save_tensor
        
        backend = MmapBackend(config)
        orchestrator = TensorStreamOrchestrator(config, backend)
        
        # Create test tensors
        file_paths = []
        for i in range(10):
            tensor = torch.randn(100, 100)
            file_path = config.cache_dir / f"tensor_{i}.ts"
            save_tensor(tensor, file_path)
            file_paths.append(file_path)
            orchestrator.register_layer(f"layer{i}", file_path, {})
        
        try:
            with orchestrator:
                # Create multiple concurrent load tasks
                tasks = []
                for i in range(10):
                    for _ in range(5):  # 5 concurrent accesses per layer
                        task = asyncio.create_task(
                            orchestrator.load_layer(f"layer{i}")
                        )
                        tasks.append(task)
                
                # Measure time
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                # Verify all succeeded
                for result in results:
                    assert not isinstance(result, Exception), f"Task failed: {result}"
                
                # Should complete in reasonable time
                assert end_time - start_time < 10.0  # 10 seconds max
                
        finally:
            # Cleanup
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()

    def test_orchestrator_lru_performance(self):
        """Test LRU queue performance with many layers."""
        config = TensorStreamConfig()
        backend = Mock(spec=BackendInterface)
        orchestrator = TensorStreamOrchestrator(config, backend)
        
        # Add many layers to LRU
        num_layers = 10000
        for i in range(num_layers):
            orchestrator._update_lru(f"layer{i}")
        
        # Measure access time
        start_time = time.time()
        for i in range(1000):
            orchestrator._update_lru(f"layer{i % num_layers}")
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0  # 1 second max
        assert len(orchestrator.lru_queue) == num_layers


if __name__ == "__main__":
    pytest.main([__file__])
