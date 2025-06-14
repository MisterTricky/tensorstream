"""
Integration tests for TensorStream.

These tests verify that all components work together correctly
in realistic scenarios.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, Mock

import tensorstream
from tensorstream import Config, offload, get_model_statistics
from tensorstream.config import BackendType, PrefetchStrategy
from tensorstream.proxy import TensorStreamProxyLayer
from tensorstream.exceptions import TensorStreamError


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TransformerLikeModel(nn.Module):
    """Transformer-like model for testing."""
    
    def __init__(self, vocab_size=1000, d_model=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TransformerLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_test_"))
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def simple_cnn():
    """Create simple CNN model."""
    return SimpleCNN()


@pytest.fixture
def transformer_model():
    """Create transformer-like model."""
    return TransformerLikeModel(vocab_size=100, d_model=256, num_layers=2)


class TestBasicIntegration:
    """Test basic TensorStream integration."""
    
    def test_offload_simple_model(self, simple_cnn, temp_storage):
        """Test offloading a simple CNN model."""
        config = Config(
            storage_path=temp_storage,
            vram_budget_gb=1.0,
            backend=BackendType.MMAP,
            debug_mode=True
        )
        
        # Get original output for comparison
        x = torch.randn(1, 3, 32, 32)
        original_output = simple_cnn(x)
        
        # Offload the model
        offloaded_model = offload(simple_cnn, config)
        
        # Verify model structure
        assert hasattr(offloaded_model, '_tensorstream_orchestrator')
        assert hasattr(offloaded_model, '_tensorstream_registry')
        
        # Check that some layers were replaced with proxies
        proxy_count = 0
        for module in offloaded_model.modules():
            if isinstance(module, TensorStreamProxyLayer):
                proxy_count += 1
        
        assert proxy_count > 0, "No proxy layers were created"
        
        # Test forward pass
        offloaded_output = offloaded_model(x)
        
        # Outputs should be close (allowing for small numerical differences)
        assert torch.allclose(original_output, offloaded_output, atol=1e-5)
        
        # Cleanup
        offloaded_model.cleanup_tensorstream()
    
    def test_offload_transformer_model(self, transformer_model, temp_storage):
        """Test offloading a transformer-like model."""
        config = Config(
            storage_path=temp_storage,
            vram_budget_gb=2.0,
            backend=BackendType.MMAP,
            prefetch_strategy=PrefetchStrategy.NEXT_LAYER
        )
        
        # Test input
        x = torch.randint(0, 100, (2, 10))  # batch_size=2, seq_len=10
        
        # Get original output
        original_output = transformer_model(x)
        
        # Offload the model
        offloaded_model = offload(transformer_model, config)
        
        # Test forward pass
        offloaded_output = offloaded_model(x)
        
        # Verify outputs match
        assert torch.allclose(original_output, offloaded_output, atol=1e-4)
        
        # Check statistics
        stats = get_model_statistics(offloaded_model)
        assert "vram_usage_bytes" in stats
        assert "total_layers" in stats
        assert stats["total_layers"] > 0
        
        # Cleanup
        offloaded_model.cleanup_tensorstream()
    
    def test_multiple_forward_passes(self, simple_cnn, temp_storage):
        """Test multiple forward passes with caching."""
        config = Config(
            storage_path=temp_storage,
            vram_budget_gb=1.0,
            cache_size_layers=3,
            backend=BackendType.MMAP
        )
        
        offloaded_model = offload(simple_cnn, config)
        
        # Multiple forward passes
        x = torch.randn(1, 3, 32, 32)
        outputs = []
        
        for i in range(5):
            output = offloaded_model(x)
            outputs.append(output)
        
        # All outputs should be identical
        for output in outputs[1:]:
            assert torch.allclose(outputs[0], output)
        
        # Check cache statistics
        stats = get_model_statistics(offloaded_model)
        assert stats["cache_hits"] > 0
        
        offloaded_model.cleanup_tensorstream()
    
    def test_memory_pressure_handling(self, temp_storage):
        """Test memory pressure handling with small budget."""
        # Create model with multiple large layers
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1000, 1000) for _ in range(5)
                ])
                self.relu = nn.ReLU()
            
            def forward(self, x):
                for layer in self.layers:
                    x = self.relu(layer(x))
                return x
        
        model = LargeModel()
        
        # Very small memory budget to force evictions
        config = Config(
            storage_path=temp_storage,
            vram_budget_gb=0.01,  # 10MB
            backend=BackendType.MMAP,
            debug_mode=True
        )
        
        offloaded_model = offload(model, config)
        
        # Forward pass should work despite memory pressure
        x = torch.randn(10, 1000)
        output = offloaded_model(x)
        
        assert output.shape == (10, 1000)
        
        # Check that evictions occurred
        stats = get_model_statistics(offloaded_model)
        assert stats["evictions"] > 0
        
        offloaded_model.cleanup_tensorstream()


class TestAdvancedIntegration:
    """Test advanced TensorStream features."""
    
    def test_different_backends(self, simple_cnn, temp_storage):
        """Test different backend configurations."""
        backends_to_test = [BackendType.MMAP]
        
        # Add CUDA backend if available
        if torch.cuda.is_available():
            backends_to_test.append(BackendType.CUDA_CORE)
        
        for backend in backends_to_test:
            config = Config(
                storage_path=temp_storage / f"backend_{backend.value}",
                backend=backend,
                vram_budget_gb=1.0
            )
            
            # Create fresh model for each backend
            model = SimpleCNN()
            x = torch.randn(1, 3, 32, 32)
            original_output = model(x)
            
            # Offload with specific backend
            offloaded_model = offload(model, config)
            offloaded_output = offloaded_model(x)
            
            # Verify output consistency
            assert torch.allclose(original_output, offloaded_output, atol=1e-5)
            
            offloaded_model.cleanup_tensorstream()
    
    def test_prefetch_strategies(self, transformer_model, temp_storage):
        """Test different prefetch strategies."""
        strategies = [
            PrefetchStrategy.NONE,
            PrefetchStrategy.NEXT_LAYER,
            PrefetchStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            config = Config(
                storage_path=temp_storage / f"prefetch_{strategy.value}",
                prefetch_strategy=strategy,
                vram_budget_gb=1.0,
                backend=BackendType.MMAP
            )
            
            # Fresh model for each strategy
            model = TransformerLikeModel(vocab_size=100, d_model=128, num_layers=2)
            x = torch.randint(0, 100, (1, 8))
            
            offloaded_model = offload(model, config)
            
            # Time multiple forward passes
            start_time = time.time()
            for _ in range(3):
                _ = offloaded_model(x)
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 10.0
            
            offloaded_model.cleanup_tensorstream()
    
    def test_compression(self, simple_cnn, temp_storage):
        """Test tensor compression."""
        config = Config(
            storage_path=temp_storage,
            compression_enabled=True,
            compression_level=6,
            vram_budget_gb=1.0,
            backend=BackendType.MMAP
        )
        
        x = torch.randn(1, 3, 32, 32)
        original_output = simple_cnn(x)
        
        offloaded_model = offload(simple_cnn, config)
        compressed_output = offloaded_model(x)
        
        # Output should still be close despite compression
        assert torch.allclose(original_output, compressed_output, atol=1e-4)
        
        # Check that compressed files exist
        ts_files = list(temp_storage.glob("*.ts"))
        assert len(ts_files) > 0
        
        # Verify compression metadata
        from tensorstream.io import get_ts_file_info
        for ts_file in ts_files:
            info = get_ts_file_info(ts_file)
            if info["compressed"]:
                assert info["compressed_size"] < info["data_size"]
        
        offloaded_model.cleanup_tensorstream()
    
    def test_model_state_preservation(self, simple_cnn, temp_storage):
        """Test that model state is preserved after offloading."""
        config = Config(storage_path=temp_storage, backend=BackendType.MMAP)
        
        # Set model to eval mode
        simple_cnn.eval()
        original_training = simple_cnn.training
        
        # Get original parameter count
        original_param_count = sum(p.numel() for p in simple_cnn.parameters())
        
        # Offload model
        offloaded_model = offload(simple_cnn, config)
        
        # Training state should be preserved
        assert offloaded_model.training == original_training
        
        # Parameter count should be preserved
        offloaded_param_count = sum(p.numel() for p in offloaded_model.parameters())
        assert offloaded_param_count == original_param_count
        
        # Switch to train mode
        offloaded_model.train()
        assert offloaded_model.training
        
        offloaded_model.cleanup_tensorstream()
    
    def test_device_movement(self, simple_cnn, temp_storage):
        """Test model device movement with TensorStream."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = Config(
            storage_path=temp_storage,
            backend=BackendType.MMAP,
            device=torch.device('cuda')
        )
        
        # Start on CPU
        model = simple_cnn.cpu()
        offloaded_model = offload(model, config)
        
        # Move to CUDA
        offloaded_model.cuda()
        
        # Test forward pass on CUDA
        x = torch.randn(1, 3, 32, 32).cuda()
        output = offloaded_model(x)
        
        assert output.device.type == 'cuda'
        
        # Move back to CPU
        offloaded_model.cpu()
        x_cpu = torch.randn(1, 3, 32, 32).cpu()
        output_cpu = offloaded_model(x_cpu)
        
        assert output_cpu.device.type == 'cpu'
        
        offloaded_model.cleanup_tensorstream()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model(self, temp_storage):
        """Test error handling for invalid models."""
        config = Config(storage_path=temp_storage)
        
        with pytest.raises(TensorStreamError):
            offload("not a model", config)
    
    def test_empty_model(self, temp_storage):
        """Test handling of model with no parameters."""
        empty_model = nn.Sequential()
        config = Config(storage_path=temp_storage)
        
        # Should not raise error but may warn
        offloaded_model = offload(empty_model, config)
        
        # Should have orchestrator even if no layers were offloaded
        assert hasattr(offloaded_model, '_tensorstream_orchestrator')
        
        offloaded_model.cleanup_tensorstream()
    
    def test_storage_permission_error(self, simple_cnn):
        """Test handling of storage permission errors."""
        # Try to use a read-only directory
        import os
        config = Config(storage_path=Path("/root/readonly"))
        
        with pytest.raises(TensorStreamError):
            offload(simple_cnn, config)
    
    def test_model_without_tensorstream(self, simple_cnn):
        """Test functions that require TensorStream on regular model."""
        with pytest.raises(TensorStreamError):
            get_model_statistics(simple_cnn)


class TestPerformance:
    """Performance and benchmark tests."""
    
    @pytest.mark.performance
    def test_offload_performance(self, temp_storage):
        """Test offloading performance with large model."""
        # Create large model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(2000, 2000) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = LargeModel()
        config = Config(
            storage_path=temp_storage,
            backend=BackendType.MMAP,
            num_io_threads=4
        )
        
        start_time = time.time()
        offloaded_model = offload(model, config)
        offload_time = time.time() - start_time
        
        # Offloading should complete in reasonable time
        assert offload_time < 60.0  # 60 seconds
        
        # Test inference performance
        x = torch.randn(1, 2000)
        
        start_time = time.time()
        for _ in range(5):
            _ = offloaded_model(x)
        inference_time = time.time() - start_time
        
        # Inference should be reasonably fast
        assert inference_time < 30.0  # 30 seconds for 5 passes
        
        offloaded_model.cleanup_tensorstream()
    
    @pytest.mark.performance
    def test_memory_efficiency(self, temp_storage):
        """Test memory efficiency of TensorStream."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple models
        models = []
        for i in range(3):
            model = SimpleCNN()
            config = Config(
                storage_path=temp_storage / f"model_{i}",
                backend=BackendType.MMAP
            )
            offloaded_model = offload(model, config)
            models.append(offloaded_model)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 200MB)
        assert memory_increase < 200 * 1024 * 1024
        
        # Cleanup
        for model in models:
            model.cleanup_tensorstream()


class TestConcurrency:
    """Test concurrent usage of TensorStream."""
    
    def test_concurrent_forward_passes(self, simple_cnn, temp_storage):
        """Test concurrent forward passes (thread safety)."""
        import threading
        import concurrent.futures
        
        config = Config(
            storage_path=temp_storage,
            backend=BackendType.MMAP,
            num_io_threads=4
        )
        
        offloaded_model = offload(simple_cnn, config)
        
        def run_inference():
            x = torch.randn(1, 3, 32, 32)
            return offloaded_model(x)
        
        # Run multiple inference threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_inference) for _ in range(8)]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # All results should have correct shape
        for result in results:
            assert result.shape == (1, 10)
        
        offloaded_model.cleanup_tensorstream()
    
    def test_multiple_models_same_storage(self, temp_storage):
        """Test multiple models using same storage directory."""
        models = []
        
        for i in range(3):
            model = SimpleCNN()
            config = Config(
                storage_path=temp_storage,  # Same storage for all
                backend=BackendType.MMAP
            )
            offloaded_model = offload(model, config)
            models.append(offloaded_model)
        
        # All models should work
        x = torch.randn(1, 3, 32, 32)
        for model in models:
            output = model(x)
            assert output.shape == (1, 10)
        
        # Cleanup
        for model in models:
            model.cleanup_tensorstream()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
