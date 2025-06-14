"""
Unit tests for TensorStream main API module.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from tensorstream.api import (
    offload, analyze_model, shard_large_layers, replace_with_proxies,
    _estimate_layer_memory, _should_offload_layer, _get_layer_file_path
)
from tensorstream.config import TensorStreamConfig, BackendType, PrefetchMode
from tensorstream.proxy import TensorStreamProxyLayer
from tensorstream.orchestrator import TensorStreamOrchestrator
from tensorstream.exceptions import TensorStreamError, ConfigError


class TestModelAnalysis:
    """Test model analysis functionality."""

    def test_estimate_layer_memory(self):
        """Test layer memory estimation."""
        # Test linear layer
        linear = nn.Linear(1000, 500)
        memory = _estimate_layer_memory(linear)
        
        # Weight: 1000 * 500 * 4 bytes + Bias: 500 * 4 bytes = 2,002,000 bytes
        expected = (1000 * 500 + 500) * 4
        assert memory == expected

    def test_estimate_layer_memory_conv2d(self):
        """Test memory estimation for Conv2D layer."""
        conv = nn.Conv2d(3, 64, kernel_size=3)
        memory = _estimate_layer_memory(conv)
        
        # Weight: 64 * 3 * 3 * 3 * 4 bytes + Bias: 64 * 4 bytes
        expected = (64 * 3 * 3 * 3 + 64) * 4
        assert memory == expected

    def test_estimate_layer_memory_no_parameters(self):
        """Test memory estimation for layers without parameters."""
        relu = nn.ReLU()
        memory = _estimate_layer_memory(relu)
        assert memory == 0

    def test_should_offload_layer(self):
        """Test layer offloading decision."""
        config = TensorStreamConfig(
            min_layer_size=1024 * 1024,  # 1MB
            offload_activations=False
        )
        
        # Large layer should be offloaded
        large_layer = nn.Linear(1000, 1000)  # ~4MB
        assert _should_offload_layer(large_layer, config)
        
        # Small layer should not be offloaded
        small_layer = nn.Linear(10, 10)  # ~400 bytes
        assert not _should_offload_layer(small_layer, config)

    def test_should_offload_layer_activation_layers(self):
        """Test offloading decision for activation layers."""
        config = TensorStreamConfig(
            min_layer_size=0,
            offload_activations=True
        )
        
        # Should offload activation when enabled
        relu = nn.ReLU()
        assert _should_offload_layer(relu, config)
        
        # Should not offload when disabled
        config.offload_activations = False
        assert not _should_offload_layer(relu, config)

    def test_get_layer_file_path(self):
        """Test layer file path generation."""
        config = TensorStreamConfig(cache_dir=Path("/tmp/cache"))
        
        path = _get_layer_file_path("test_layer", config)
        expected = Path("/tmp/cache/test_layer.ts")
        assert path == expected

    def test_analyze_model_simple(self):
        """Test model analysis with simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1000, 500)
                self.layer2 = nn.Linear(500, 100)
                self.relu = nn.ReLU()
        
        model = SimpleModel()
        config = TensorStreamConfig(min_layer_size=1024)  # 1KB
        
        analysis = analyze_model(model, config)
        
        assert "total_parameters" in analysis
        assert "total_memory" in analysis
        assert "offloadable_layers" in analysis
        assert "offloadable_memory" in analysis
        
        # Should identify linear layers as offloadable
        assert len(analysis["offloadable_layers"]) >= 2
        assert "layer1" in analysis["offloadable_layers"]
        assert "layer2" in analysis["offloadable_layers"]

    def test_analyze_model_complex(self):
        """Test model analysis with complex nested model."""
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 250)
                )
                self.head = nn.ModuleDict({
                    'classifier': nn.Linear(250, 10),
                    'regressor': nn.Linear(250, 1)
                })
        
        model = ComplexModel()
        config = TensorStreamConfig(min_layer_size=1024)
        
        analysis = analyze_model(model, config)
        
        # Should find nested layers
        offloadable = analysis["offloadable_layers"]
        assert any("backbone.0" in layer for layer in offloadable)
        assert any("backbone.2" in layer for layer in offloadable)
        assert any("head.classifier" in layer for layer in offloadable)

    def test_analyze_model_with_shared_layers(self):
        """Test model analysis with shared/reused layers."""
        shared_layer = nn.Linear(100, 50)
        
        class ModelWithSharedLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = shared_layer
                self.layer1 = shared_layer  # Same instance
                self.layer2 = nn.Linear(50, 10)
        
        model = ModelWithSharedLayers()
        config = TensorStreamConfig(min_layer_size=1024)
        
        analysis = analyze_model(model, config)
        
        # Should handle shared layers correctly
        offloadable = analysis["offloadable_layers"]
        # Shared layer should only be counted once
        shared_count = sum(1 for layer in offloadable if "shared" in layer or "layer1" in layer)
        assert shared_count <= 1


class TestLayerSharding:
    """Test layer sharding functionality."""

    def test_shard_large_layers_no_sharding_needed(self):
        """Test sharding when no layers exceed size limit."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(100, 50)
                self.layer2 = nn.Linear(50, 10)
        
        model = SimpleModel()
        config = TensorStreamConfig(max_shard_size=1024 * 1024)  # 1MB
        
        # No sharding should occur for small layers
        sharded_layers = shard_large_layers(model, config)
        assert len(sharded_layers) == 0

    def test_shard_large_layers_linear_layer(self):
        """Test sharding of large linear layer."""
        large_layer = nn.Linear(10000, 5000)  # ~200MB
        
        class ModelWithLargeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.large_layer = large_layer
        
        model = ModelWithLargeLayer()
        config = TensorStreamConfig(max_shard_size=50 * 1024 * 1024)  # 50MB
        
        sharded_layers = shard_large_layers(model, config)
        
        # Should create shards for large layer
        assert len(sharded_layers) > 0
        assert "large_layer" in sharded_layers
        
        shards = sharded_layers["large_layer"]
        assert len(shards) > 1  # Should be split into multiple shards
        
        # Verify shard shapes
        original_weight_shape = large_layer.weight.shape
        total_out_features = sum(shard["weight"].shape[0] for shard in shards)
        assert total_out_features == original_weight_shape[0]

    def test_shard_large_layers_conv2d_layer(self):
        """Test sharding of large Conv2D layer."""
        large_conv = nn.Conv2d(3, 1000, kernel_size=7)  # Large number of output channels
        
        class ModelWithLargeConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.large_conv = large_conv
        
        model = ModelWithLargeConv()
        config = TensorStreamConfig(max_shard_size=1024 * 1024)  # 1MB
        
        sharded_layers = shard_large_layers(model, config)
        
        if len(sharded_layers) > 0:  # If sharding occurred
            assert "large_conv" in sharded_layers
            shards = sharded_layers["large_conv"]
            
            # Verify Conv2D shard shapes
            original_weight_shape = large_conv.weight.shape
            total_out_channels = sum(shard["weight"].shape[0] for shard in shards)
            assert total_out_channels == original_weight_shape[0]

    def test_shard_layer_weight_and_bias(self):
        """Test that sharding preserves weight and bias relationship."""
        layer = nn.Linear(1000, 1000)
        config = TensorStreamConfig(max_shard_size=1024 * 1024)  # Small to force sharding
        
        # Calculate memory size
        memory_size = _estimate_layer_memory(layer)
        
        if memory_size > config.max_shard_size:
            # Manually test sharding logic
            from tensorstream.api import _shard_linear_layer
            
            shards = _shard_linear_layer(layer, config.max_shard_size)
            
            # Check that weight and bias are consistently sharded
            for shard in shards:
                assert "weight" in shard
                if layer.bias is not None:
                    assert "bias" in shard
                    assert shard["weight"].shape[0] == shard["bias"].shape[0]


class TestProxyReplacement:
    """Test proxy layer replacement functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock(spec=TensorStreamOrchestrator)
        orchestrator.register_layer = Mock()
        return orchestrator

    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        class SampleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(100, 50)
                self.layer2 = nn.Linear(50, 25)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
        
        return SampleModel()

    def test_replace_with_proxies_basic(self, sample_model, mock_orchestrator):
        """Test basic proxy replacement."""
        config = TensorStreamConfig(
            cache_dir=Path("/tmp/cache"),
            min_layer_size=1024  # Small threshold to catch layers
        )
        
        offloadable_layers = ["layer1", "layer2"]
        
        proxy_layers = replace_with_proxies(
            sample_model, offloadable_layers, {}, config, mock_orchestrator
        )
        
        # Should create proxy layers
        assert len(proxy_layers) == 2
        assert "layer1" in proxy_layers
        assert "layer2" in proxy_layers
        
        # Original layers should be replaced
        assert isinstance(sample_model.layer1, TensorStreamProxyLayer)
        assert isinstance(sample_model.layer2, TensorStreamProxyLayer)
        
        # Non-offloadable layers should remain unchanged
        assert not isinstance(sample_model.relu, TensorStreamProxyLayer)
        assert not isinstance(sample_model.dropout, TensorStreamProxyLayer)

    def test_replace_with_proxies_nested_layers(self, mock_orchestrator):
        """Test proxy replacement with nested layers."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 25)
                )
        
        model = NestedModel()
        config = TensorStreamConfig(cache_dir=Path("/tmp/cache"))
        offloadable_layers = ["backbone.0", "backbone.2"]
        
        proxy_layers = replace_with_proxies(
            model, offloadable_layers, {}, config, mock_orchestrator
        )
        
        # Should handle nested layer replacement
        assert len(proxy_layers) == 2
        assert isinstance(model.backbone[0], TensorStreamProxyLayer)
        assert isinstance(model.backbone[2], TensorStreamProxyLayer)
        assert not isinstance(model.backbone[1], TensorStreamProxyLayer)

    def test_replace_with_proxies_with_shards(self, mock_orchestrator):
        """Test proxy replacement with sharded layers."""
        model = nn.Sequential(nn.Linear(1000, 500))
        config = TensorStreamConfig(cache_dir=Path("/tmp/cache"))
        
        # Mock sharded layers
        sharded_layers = {
            "0": [
                {"weight": torch.randn(250, 1000), "bias": torch.randn(250)},
                {"weight": torch.randn(250, 1000), "bias": torch.randn(250)}
            ]
        }
        
        proxy_layers = replace_with_proxies(
            model, ["0"], sharded_layers, config, mock_orchestrator
        )
        
        # Should create proxy for each shard
        assert len(proxy_layers) == 2
        assert "0_shard_0" in proxy_layers
        assert "0_shard_1" in proxy_layers

    def test_replace_with_proxies_save_state_dicts(self, sample_model, mock_orchestrator):
        """Test that original state dicts are saved."""
        config = TensorStreamConfig(cache_dir=Path(tempfile.mkdtemp()))
        offloadable_layers = ["layer1"]
        
        # Get original state dict
        original_state = sample_model.layer1.state_dict().copy()
        
        with patch('tensorstream.io.save_tensor') as mock_save:
            proxy_layers = replace_with_proxies(
                sample_model, offloadable_layers, {}, config, mock_orchestrator
            )
            
            # Should save state dict to disk
            mock_save.assert_called()
            
            # Should pass original state to save function
            call_args = mock_save.call_args_list[0]
            saved_state = call_args[0][0]
            
            # Compare tensors
            assert torch.equal(saved_state['weight'], original_state['weight'])
            assert torch.equal(saved_state['bias'], original_state['bias'])


class TestMainOffloadAPI:
    """Test main offload API functionality."""

    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        class SampleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1000, 500)
                self.layer2 = nn.Linear(500, 100)
                self.layer3 = nn.Linear(100, 10)
                self.relu = nn.ReLU()
        
        return SampleModel()

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_basic(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test basic offload functionality."""
        # Setup mocks
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        # Test offload
        with patch('tensorstream.io.save_tensor'):
            result = offload(sample_model)
        
        # Should return orchestrator
        assert result == mock_orchestrator
        
        # Should create backend and orchestrator
        mock_get_backend.assert_called_once()
        mock_orchestrator_class.assert_called_once()

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_with_config(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test offload with custom configuration."""
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        custom_config = TensorStreamConfig(
            memory_limit=512 * 1024 * 1024,
            backend_type=BackendType.CUDA_CORE,
            prefetch_mode=PrefetchMode.ADAPTIVE
        )
        
        with patch('tensorstream.io.save_tensor'):
            result = offload(sample_model, config=custom_config)
        
        # Should use custom config
        call_args = mock_orchestrator_class.call_args
        passed_config = call_args[0][0]
        assert passed_config.memory_limit == custom_config.memory_limit
        assert passed_config.backend_type == custom_config.backend_type

    def test_offload_invalid_model(self):
        """Test offload with invalid model."""
        with pytest.raises(TensorStreamError, match="Model must be"):
            offload("not a model")

    def test_offload_empty_model(self):
        """Test offload with model that has no parameters."""
        empty_model = nn.Sequential()
        
        # Should handle empty model gracefully
        with patch('tensorstream.io.save_tensor'):
            result = offload(empty_model)
        
        # Should still return orchestrator
        assert result is not None

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_dry_run(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test offload dry run mode."""
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        
        with patch('tensorstream.io.save_tensor') as mock_save:
            result = offload(sample_model, dry_run=True)
        
        # Should not create orchestrator in dry run
        mock_orchestrator_class.assert_not_called()
        
        # Should not save tensors in dry run
        mock_save.assert_not_called()
        
        # Should return analysis instead
        assert isinstance(result, dict)
        assert "total_parameters" in result
        assert "offloadable_layers" in result

    def test_offload_with_invalid_config(self, sample_model):
        """Test offload with invalid configuration."""
        invalid_config = "not a config"
        
        with pytest.raises(TensorStreamError, match="Config must be"):
            offload(sample_model, config=invalid_config)

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_backend_creation_failure(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test offload when backend creation fails."""
        mock_get_backend.side_effect = Exception("Backend creation failed")
        
        with pytest.raises(TensorStreamError, match="Failed to create backend"):
            offload(sample_model)

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_orchestrator_creation_failure(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test offload when orchestrator creation fails."""
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator_class.side_effect = Exception("Orchestrator creation failed")
        
        with pytest.raises(TensorStreamError, match="Failed to create orchestrator"):
            offload(sample_model)

    @patch('tensorstream.api.save_tensor')
    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_save_failure(self, mock_get_backend, mock_orchestrator_class, mock_save, sample_model):
        """Test offload when tensor saving fails."""
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_save.side_effect = Exception("Save failed")
        
        with pytest.raises(TensorStreamError, match="Failed to save layer"):
            offload(sample_model)

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_preserves_model_state(self, mock_get_backend, mock_orchestrator_class, sample_model):
        """Test that offload preserves model training state."""
        # Set model to eval mode
        sample_model.eval()
        original_training = sample_model.training
        
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        with patch('tensorstream.io.save_tensor'):
            offload(sample_model)
        
        # Training state should be preserved
        assert sample_model.training == original_training

    @patch('tensorstream.api.TensorStreamOrchestrator')
    @patch('tensorstream.backends.get_backend')
    def test_offload_handles_device_placement(self, mock_get_backend, mock_orchestrator_class):
        """Test offload with model on different devices."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        model = nn.Linear(100, 50).to(device)
        
        mock_backend = Mock()
        mock_get_backend.return_value = mock_backend
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        with patch('tensorstream.io.save_tensor'):
            result = offload(model)
        
        # Should handle device placement
        assert result is not None


class TestOffloadIntegration:
    """Integration tests for offload API."""

    def test_offload_full_workflow(self):
        """Test complete offload workflow."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1000, 500)
                self.layer2 = nn.Linear(500, 100)
                self.relu = nn.ReLU()
        
        model = TestModel()
        
        # Use temporary directory
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            min_layer_size=1024,  # Small threshold
            backend_type=BackendType.MMAP
        )
        
        try:
            # Test dry run first
            analysis = offload(model, config=config, dry_run=True)
            assert isinstance(analysis, dict)
            assert analysis["total_parameters"] > 0
            assert len(analysis["offloadable_layers"]) > 0
            
            # Test actual offload
            orchestrator = offload(model, config=config)
            assert orchestrator is not None
            
            # Verify layers were replaced with proxies
            assert isinstance(model.layer1, TensorStreamProxyLayer)
            assert isinstance(model.layer2, TensorStreamProxyLayer)
            assert not isinstance(model.relu, TensorStreamProxyLayer)
            
            # Test model can still be used (would need async forward)
            # This is tested in integration tests
            
        finally:
            # Cleanup
            import shutil
            if config.cache_dir.exists():
                shutil.rmtree(config.cache_dir)

    def test_offload_with_sharding(self):
        """Test offload with layer sharding."""
        # Create model with large layer that will be sharded
        model = nn.Sequential(
            nn.Linear(2000, 2000),  # Large layer
            nn.ReLU(),
            nn.Linear(2000, 10)
        )
        
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            max_shard_size=4 * 1024 * 1024,  # 4MB to force sharding
            min_layer_size=1024,
            backend_type=BackendType.MMAP
        )
        
        try:
            orchestrator = offload(model, config=config)
            assert orchestrator is not None
            
            # Large layer should be replaced with proxy
            assert isinstance(model[0], TensorStreamProxyLayer)
            
            # Check if sharding occurred by examining orchestrator
            stats = orchestrator.get_statistics()
            # If sharding occurred, we'd have more registered layers than original
            
        finally:
            # Cleanup
            import shutil
            if config.cache_dir.exists():
                shutil.rmtree(config.cache_dir)


@pytest.mark.performance
class TestOffloadPerformance:
    """Performance tests for offload API."""

    def test_offload_large_model_performance(self):
        """Test offload performance with large model."""
        import time
        
        # Create large model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for i in range(10):
                    layers.append(nn.Linear(1000, 1000))
                    layers.append(nn.ReLU())
                self.layers = nn.Sequential(*layers)
                self.final = nn.Linear(1000, 10)
        
        model = LargeModel()
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            backend_type=BackendType.MMAP
        )
        
        try:
            # Measure offload time
            start_time = time.time()
            orchestrator = offload(model, config=config)
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 30.0  # 30 seconds max
            assert orchestrator is not None
            
        finally:
            # Cleanup
            import shutil
            if config.cache_dir.exists():
                shutil.rmtree(config.cache_dir)

    def test_offload_memory_efficiency(self):
        """Test that offload doesn't use excessive memory."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple models and offload them
        models = []
        orchestrators = []
        
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            backend_type=BackendType.MMAP
        )
        
        try:
            for i in range(5):
                model = nn.Sequential(
                    nn.Linear(500, 500),
                    nn.ReLU(),
                    nn.Linear(500, 100)
                )
                orchestrator = offload(model, config=config)
                models.append(model)
                orchestrators.append(orchestrator)
            
            # Memory usage shouldn't grow excessively
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Allow reasonable growth (100MB max)
            assert memory_growth < 100 * 1024 * 1024
            
        finally:
            # Cleanup
            models.clear()
            orchestrators.clear()
            gc.collect()
            
            import shutil
            if config.cache_dir.exists():
                shutil.rmtree(config.cache_dir)


if __name__ == "__main__":
    pytest.main([__file__])
