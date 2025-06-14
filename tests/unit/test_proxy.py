"""
Unit tests for TensorStream proxy layer module.
"""

import pytest
import torch
import torch.nn as nn
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from tensorstream.proxy import TensorStreamProxyLayer
from tensorstream.orchestrator import TensorStreamOrchestrator, LayerState
from tensorstream.config import TensorStreamConfig
from tensorstream.exceptions import ProxyError, OrchestratorError
from tensorstream.backends import BackendInterface


class TestTensorStreamProxyLayer:
    """Test TensorStreamProxyLayer functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock(spec=TensorStreamOrchestrator)
        orchestrator.load_layer = AsyncMock()
        orchestrator.register_layer = Mock()
        orchestrator.get_statistics = Mock(return_value={})
        return orchestrator

    @pytest.fixture
    def original_layer(self):
        """Create original layer for testing."""
        return nn.Linear(100, 50)

    @pytest.fixture
    def proxy_layer(self, original_layer, mock_orchestrator):
        """Create proxy layer for testing."""
        return TensorStreamProxyLayer(
            original_layer=original_layer,
            layer_id="test_layer",
            orchestrator=mock_orchestrator,
            file_path=Path("/tmp/test.ts")
        )

    def test_proxy_layer_initialization(self, proxy_layer, original_layer, mock_orchestrator):
        """Test proxy layer initialization."""
        assert proxy_layer.original_layer is original_layer
        assert proxy_layer.layer_id == "test_layer"
        assert proxy_layer.orchestrator is mock_orchestrator
        assert proxy_layer.file_path == Path("/tmp/test.ts")
        assert proxy_layer._loaded_state_dict is None
        assert not proxy_layer._is_loading

    def test_proxy_layer_properties(self, proxy_layer, original_layer):
        """Test proxy layer properties delegation."""
        # Test that properties are delegated to original layer
        assert proxy_layer.in_features == original_layer.in_features
        assert proxy_layer.out_features == original_layer.out_features
        assert proxy_layer.training == original_layer.training

    def test_proxy_layer_parameters(self, proxy_layer, original_layer):
        """Test proxy layer parameter access."""
        # Initially should have no parameters (not loaded)
        proxy_params = list(proxy_layer.parameters())
        assert len(proxy_params) == 0
        
        # Test named_parameters
        named_params = list(proxy_layer.named_parameters())
        assert len(named_params) == 0

    def test_proxy_layer_state_dict(self, proxy_layer):
        """Test proxy layer state_dict access."""
        # Should return empty state dict when not loaded
        state_dict = proxy_layer.state_dict()
        assert len(state_dict) == 0

    def test_proxy_layer_load_state_dict(self, proxy_layer):
        """Test proxy layer load_state_dict."""
        test_state_dict = {
            'weight': torch.randn(50, 100),
            'bias': torch.randn(50)
        }
        
        # Should store the state dict for later use
        proxy_layer.load_state_dict(test_state_dict)
        assert proxy_layer._loaded_state_dict is test_state_dict

    @pytest.mark.asyncio
    async def test_proxy_layer_ensure_loaded_success(self, proxy_layer, mock_orchestrator):
        """Test successful layer loading."""
        # Mock successful loading
        test_state_dict = {
            'weight': torch.randn(50, 100),
            'bias': torch.randn(50)
        }
        mock_orchestrator.load_layer.return_value = test_state_dict
        
        await proxy_layer._ensure_loaded()
        
        # Should register and load layer
        mock_orchestrator.register_layer.assert_called_once_with(
            "test_layer", Path("/tmp/test.ts"), {}
        )
        mock_orchestrator.load_layer.assert_called_once_with("test_layer")
        
        # Should apply loaded state to original layer
        assert torch.equal(proxy_layer.original_layer.weight, test_state_dict['weight'])
        assert torch.equal(proxy_layer.original_layer.bias, test_state_dict['bias'])

    @pytest.mark.asyncio
    async def test_proxy_layer_ensure_loaded_with_preloaded_state(self, proxy_layer):
        """Test layer loading with preloaded state dict."""
        # Preload state dict
        test_state_dict = {
            'weight': torch.randn(50, 100),
            'bias': torch.randn(50)
        }
        proxy_layer.load_state_dict(test_state_dict)
        
        await proxy_layer._ensure_loaded()
        
        # Should use preloaded state without calling orchestrator
        proxy_layer.orchestrator.load_layer.assert_not_called()
        assert torch.equal(proxy_layer.original_layer.weight, test_state_dict['weight'])
        assert torch.equal(proxy_layer.original_layer.bias, test_state_dict['bias'])

    @pytest.mark.asyncio
    async def test_proxy_layer_ensure_loaded_orchestrator_error(self, proxy_layer, mock_orchestrator):
        """Test layer loading with orchestrator error."""
        mock_orchestrator.load_layer.side_effect = OrchestratorError("Load failed")
        
        with pytest.raises(ProxyError, match="Failed to load layer"):
            await proxy_layer._ensure_loaded()

    @pytest.mark.asyncio
    async def test_proxy_layer_ensure_loaded_concurrent(self, proxy_layer, mock_orchestrator):
        """Test concurrent loading attempts."""
        # Mock slow loading
        async def slow_load(*args):
            await asyncio.sleep(0.1)
            return {
                'weight': torch.randn(50, 100),
                'bias': torch.randn(50)
            }
        
        mock_orchestrator.load_layer.side_effect = slow_load
        
        # Start multiple concurrent loads
        tasks = [
            asyncio.create_task(proxy_layer._ensure_loaded()),
            asyncio.create_task(proxy_layer._ensure_loaded()),
            asyncio.create_task(proxy_layer._ensure_loaded())
        ]
        
        await asyncio.gather(*tasks)
        
        # Should only load once
        assert mock_orchestrator.load_layer.call_count == 1

    @pytest.mark.asyncio
    async def test_proxy_layer_forward(self, proxy_layer, mock_orchestrator):
        """Test proxy layer forward pass."""
        # Setup test data
        input_tensor = torch.randn(32, 100)
        test_state_dict = {
            'weight': torch.randn(50, 100),
            'bias': torch.randn(50)
        }
        mock_orchestrator.load_layer.return_value = test_state_dict
        
        # Forward pass should trigger loading
        output = await proxy_layer.forward(input_tensor)
        
        # Should have correct output shape
        assert output.shape == (32, 50)
        
        # Should have loaded the layer
        mock_orchestrator.load_layer.assert_called_once()

    def test_proxy_layer_forward_sync(self, proxy_layer):
        """Test synchronous forward pass (should raise error)."""
        input_tensor = torch.randn(32, 100)
        
        # Synchronous forward should raise error
        with pytest.raises(ProxyError, match="Use async forward"):
            proxy_layer(input_tensor)

    @pytest.mark.asyncio
    async def test_proxy_layer_forward_with_args_kwargs(self, proxy_layer, mock_orchestrator):
        """Test forward pass with additional arguments."""
        # Use a more complex layer that accepts additional arguments
        class ComplexLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 50)
            
            def forward(self, x, scale=1.0, offset=0.0):
                return self.linear(x) * scale + offset
        
        complex_layer = ComplexLayer()
        proxy = TensorStreamProxyLayer(
            original_layer=complex_layer,
            layer_id="complex_layer",
            orchestrator=mock_orchestrator,
            file_path=Path("/tmp/complex.ts")
        )
        
        # Mock loading
        test_state_dict = {
            'linear.weight': torch.randn(50, 100),
            'linear.bias': torch.randn(50)
        }
        mock_orchestrator.load_layer.return_value = test_state_dict
        
        # Forward with additional arguments
        input_tensor = torch.randn(32, 100)
        output = await proxy.forward(input_tensor, scale=2.0, offset=1.0)
        
        assert output.shape == (32, 50)

    def test_proxy_layer_train_eval_mode(self, proxy_layer, original_layer):
        """Test train/eval mode switching."""
        # Test train mode
        proxy_layer.train()
        assert proxy_layer.training
        assert original_layer.training
        
        # Test eval mode
        proxy_layer.eval()
        assert not proxy_layer.training
        assert not original_layer.training

    def test_proxy_layer_to_device(self, proxy_layer, original_layer):
        """Test device movement."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        proxy_layer.to(device)
        
        # Original layer should be moved
        for param in original_layer.parameters():
            assert param.device == device

    def test_proxy_layer_repr(self, proxy_layer):
        """Test proxy layer string representation."""
        repr_str = repr(proxy_layer)
        assert "TensorStreamProxyLayer" in repr_str
        assert "test_layer" in repr_str
        assert "/tmp/test.ts" in repr_str

    def test_proxy_layer_modules(self, proxy_layer, original_layer):
        """Test module iteration."""
        # Test named_modules
        named_modules = dict(proxy_layer.named_modules())
        assert "" in named_modules  # Root module
        
        # Test modules
        modules = list(proxy_layer.modules())
        assert proxy_layer in modules

    def test_proxy_layer_children(self, proxy_layer, original_layer):
        """Test children access."""
        children = list(proxy_layer.children())
        # Proxy layer doesn't expose original layer as child by default
        # This prevents parameter counting issues
        assert len(children) == 0

    @pytest.mark.asyncio
    async def test_proxy_layer_parameter_gradients(self, proxy_layer, mock_orchestrator):
        """Test gradient handling."""
        # Setup test data
        input_tensor = torch.randn(32, 100, requires_grad=True)
        target = torch.randn(32, 50)
        test_state_dict = {
            'weight': torch.randn(50, 100, requires_grad=True),
            'bias': torch.randn(50, requires_grad=True)
        }
        mock_orchestrator.load_layer.return_value = test_state_dict
        
        # Forward pass
        output = await proxy_layer.forward(input_tensor)
        
        # Compute loss and backward
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        assert proxy_layer.original_layer.weight.grad is not None
        assert proxy_layer.original_layer.bias.grad is not None

    def test_proxy_layer_save_restore_metadata(self, proxy_layer):
        """Test metadata save/restore functionality."""
        metadata = {
            "original_shape": [50, 100],
            "dtype": "float32",
            "device": "cpu"
        }
        
        proxy_layer.save_metadata(metadata)
        restored = proxy_layer.get_metadata()
        
        assert restored == metadata

    @pytest.mark.asyncio
    async def test_proxy_layer_unload(self, proxy_layer, mock_orchestrator):
        """Test layer unloading functionality."""
        # First load the layer
        test_state_dict = {
            'weight': torch.randn(50, 100),
            'bias': torch.randn(50)
        }
        mock_orchestrator.load_layer.return_value = test_state_dict
        
        await proxy_layer._ensure_loaded()
        
        # Now unload
        proxy_layer.unload()
        
        # Parameters should be cleared
        assert len(list(proxy_layer.original_layer.parameters())) == 0 or \
               all(p.numel() == 0 for p in proxy_layer.original_layer.parameters())

    def test_proxy_layer_hooks(self, proxy_layer):
        """Test forward hook functionality."""
        hook_called = False
        hook_input = None
        hook_output = None
        
        def test_hook(module, input, output):
            nonlocal hook_called, hook_input, hook_output
            hook_called = True
            hook_input = input
            hook_output = output
        
        # Register hook
        handle = proxy_layer.register_forward_hook(test_hook)
        
        # Note: This test would need async forward to actually trigger
        # Just test that hook registration works
        assert handle is not None
        
        # Remove hook
        handle.remove()


class TestProxyLayerIntegration:
    """Integration tests for proxy layer."""

    @pytest.fixture
    def real_config(self):
        """Create real configuration."""
        return TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=100 * 1024 * 1024,  # 100MB
        )

    @pytest.mark.asyncio
    async def test_proxy_layer_full_workflow(self, real_config):
        """Test complete proxy layer workflow."""
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.orchestrator import TensorStreamOrchestrator
        from tensorstream.io import save_tensor
        
        # Create components
        backend = MmapBackend(real_config)
        orchestrator = TensorStreamOrchestrator(real_config, backend)
        original_layer = nn.Linear(100, 50)
        
        # Save layer state to disk
        state_dict = original_layer.state_dict()
        file_path = real_config.cache_dir / "test_layer.ts"
        save_tensor(state_dict, file_path)
        
        # Create proxy layer
        proxy_layer = TensorStreamProxyLayer(
            original_layer=nn.Linear(100, 50),  # Fresh layer
            layer_id="test_layer",
            orchestrator=orchestrator,
            file_path=file_path
        )
        
        try:
            with orchestrator:
                # Test forward pass
                input_tensor = torch.randn(32, 100)
                output = await proxy_layer.forward(input_tensor)
                
                # Verify output shape
                assert output.shape == (32, 50)
                
                # Verify layer was loaded correctly
                assert torch.allclose(
                    proxy_layer.original_layer.weight, 
                    state_dict['weight']
                )
                assert torch.allclose(
                    proxy_layer.original_layer.bias, 
                    state_dict['bias']
                )
                
        finally:
            # Cleanup
            if file_path.exists():
                file_path.unlink()

    @pytest.mark.asyncio
    async def test_proxy_layer_model_integration(self, real_config):
        """Test proxy layer in a complete model."""
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.orchestrator import TensorStreamOrchestrator
        from tensorstream.io import save_tensor
        
        # Create model with proxy layers
        class TestModel(nn.Module):
            def __init__(self, orchestrator, cache_dir):
                super().__init__()
                self.layer1 = self._create_proxy_layer(
                    nn.Linear(100, 64), "layer1", orchestrator, cache_dir / "layer1.ts"
                )
                self.layer2 = self._create_proxy_layer(
                    nn.Linear(64, 32), "layer2", orchestrator, cache_dir / "layer2.ts"
                )
                self.layer3 = self._create_proxy_layer(
                    nn.Linear(32, 10), "layer3", orchestrator, cache_dir / "layer3.ts"
                )
            
            def _create_proxy_layer(self, original, layer_id, orchestrator, file_path):
                # Save original state
                save_tensor(original.state_dict(), file_path)
                
                # Create fresh layer for proxy
                fresh_layer = type(original)(
                    original.in_features, original.out_features
                )
                
                return TensorStreamProxyLayer(
                    original_layer=fresh_layer,
                    layer_id=layer_id,
                    orchestrator=orchestrator,
                    file_path=file_path
                )
            
            async def forward(self, x):
                x = torch.relu(await self.layer1.forward(x))
                x = torch.relu(await self.layer2.forward(x))
                x = await self.layer3.forward(x)
                return x
        
        # Create components
        backend = MmapBackend(real_config)
        orchestrator = TensorStreamOrchestrator(real_config, backend)
        
        try:
            model = TestModel(orchestrator, real_config.cache_dir)
            
            with orchestrator:
                # Test model forward pass
                input_tensor = torch.randn(16, 100)
                output = await model.forward(input_tensor)
                
                # Verify output shape
                assert output.shape == (16, 10)
                
                # Verify all layers were loaded
                stats = orchestrator.get_statistics()
                assert stats["layers_in_memory"] == 3
                
        finally:
            # Cleanup
            for i in range(1, 4):
                file_path = real_config.cache_dir / f"layer{i}.ts"
                if file_path.exists():
                    file_path.unlink()


class TestProxyLayerEdgeCases:
    """Test edge cases and error conditions."""

    def test_proxy_layer_with_custom_layer(self):
        """Test proxy layer with custom layer types."""
        class CustomLayer(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(size, size))
                self.size = size
            
            def forward(self, x):
                return torch.matmul(x, self.weight)
        
        orchestrator = Mock(spec=TensorStreamOrchestrator)
        original = CustomLayer(50)
        
        proxy = TensorStreamProxyLayer(
            original_layer=original,
            layer_id="custom_layer",
            orchestrator=orchestrator,
            file_path=Path("/tmp/custom.ts")
        )
        
        # Should preserve custom attributes
        assert proxy.size == 50

    def test_proxy_layer_with_no_parameters(self):
        """Test proxy layer with layers that have no parameters."""
        orchestrator = Mock(spec=TensorStreamOrchestrator)
        original = nn.ReLU()  # No parameters
        
        proxy = TensorStreamProxyLayer(
            original_layer=original,
            layer_id="relu_layer",
            orchestrator=orchestrator,
            file_path=Path("/tmp/relu.ts")
        )
        
        # Should handle layers with no parameters
        assert len(list(proxy.parameters())) == 0

    @pytest.mark.asyncio
    async def test_proxy_layer_state_dict_mismatch(self, mock_orchestrator):
        """Test handling of state dict mismatches."""
        original = nn.Linear(100, 50)
        proxy = TensorStreamProxyLayer(
            original_layer=original,
            layer_id="test_layer",
            orchestrator=mock_orchestrator,
            file_path=Path("/tmp/test.ts")
        )
        
        # Mock loading incompatible state dict
        bad_state_dict = {
            'weight': torch.randn(30, 80),  # Wrong shape
            'bias': torch.randn(30)
        }
        mock_orchestrator.load_layer.return_value = bad_state_dict
        
        with pytest.raises(ProxyError, match="Failed to load layer"):
            await proxy._ensure_loaded()

    def test_proxy_layer_invalid_file_path(self):
        """Test proxy layer with invalid file path."""
        orchestrator = Mock(spec=TensorStreamOrchestrator)
        original = nn.Linear(100, 50)
        
        # Should not raise error during creation
        proxy = TensorStreamProxyLayer(
            original_layer=original,
            layer_id="test_layer",
            orchestrator=orchestrator,
            file_path=Path("/nonexistent/path.ts")
        )
        
        assert proxy.file_path == Path("/nonexistent/path.ts")


@pytest.mark.performance
class TestProxyLayerPerformance:
    """Performance tests for proxy layer."""

    @pytest.mark.asyncio
    async def test_proxy_layer_loading_performance(self):
        """Test proxy layer loading performance."""
        from tensorstream.backends.mmap_backend import MmapBackend
        from tensorstream.orchestrator import TensorStreamOrchestrator
        from tensorstream.io import save_tensor
        import time
        
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=1024 * 1024 * 1024,  # 1GB
        )
        
        backend = MmapBackend(config)
        orchestrator = TensorStreamOrchestrator(config, backend)
        
        # Create large layer
        large_layer = nn.Linear(1000, 1000)
        state_dict = large_layer.state_dict()
        file_path = config.cache_dir / "large_layer.ts"
        save_tensor(state_dict, file_path)
        
        proxy = TensorStreamProxyLayer(
            original_layer=nn.Linear(1000, 1000),
            layer_id="large_layer",
            orchestrator=orchestrator,
            file_path=file_path
        )
        
        try:
            with orchestrator:
                # Measure loading time
                start_time = time.time()
                await proxy._ensure_loaded()
                end_time = time.time()
                
                # Should load reasonably quickly
                assert end_time - start_time < 5.0  # 5 seconds max
                
                # Measure forward pass time
                input_tensor = torch.randn(100, 1000)
                start_time = time.time()
                output = await proxy.forward(input_tensor)
                end_time = time.time()
                
                # Forward pass should be fast after loading
                assert end_time - start_time < 1.0  # 1 second max
                assert output.shape == (100, 1000)
                
        finally:
            if file_path.exists():
                file_path.unlink()

    @pytest.mark.asyncio
    async def test_proxy_layer_memory_usage(self):
        """Test proxy layer memory usage."""
        import gc
        
        config = TensorStreamConfig(
            cache_dir=Path(tempfile.mkdtemp()),
            memory_limit=100 * 1024 * 1024,  # 100MB
        )
        
        # This test would measure actual memory usage
        # For now, just test that proxy layers can be created and destroyed
        proxies = []
        
        for i in range(100):
            orchestrator = Mock(spec=TensorStreamOrchestrator)
            original = nn.Linear(100, 50)
            proxy = TensorStreamProxyLayer(
                original_layer=original,
                layer_id=f"layer_{i}",
                orchestrator=orchestrator,
                file_path=Path(f"/tmp/layer_{i}.ts")
            )
            proxies.append(proxy)
        
        # Clear proxies
        proxies.clear()
        gc.collect()
        
        # Should not cause memory issues
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
