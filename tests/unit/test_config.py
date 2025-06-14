"""
Unit tests for TensorStream configuration management.
"""

import pytest
import tempfile
from pathlib import Path

import torch

from tensorstream.config import (
    Config, 
    BackendType, 
    PrefetchStrategy, 
    MemoryPressureMode,
    create_default_config
)
from tensorstream.exceptions import ConfigurationError


class TestConfig:
    """Test suite for Config class."""
    
    def test_basic_config_creation(self, temp_storage_dir):
        """Test basic configuration creation."""
        config = Config(storage_path=temp_storage_dir)
        
        assert config.storage_path == temp_storage_dir
        assert config.backend == BackendType.AUTO
        assert config.prefetch_strategy == PrefetchStrategy.NEXT_LAYER
        assert config.memory_pressure_mode == MemoryPressureMode.ADAPTIVE
        assert config.vram_budget_gb > 0
    
    def test_config_validation(self, temp_storage_dir):
        """Test configuration validation."""
        # Valid config
        config = Config(
            storage_path=temp_storage_dir,
            vram_budget_gb=4.0,
            backend=BackendType.MMAP,
            compression_enabled=True,
            compression_level=5
        )
        
        assert config.vram_budget_gb == 4.0
        assert config.backend == BackendType.MMAP
        assert config.compression_enabled is True
        assert config.compression_level == 5
    
    def test_invalid_storage_path(self):
        """Test invalid storage path handling."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config(storage_path="/nonexistent/deeply/nested/path/that/cannot/be/created")
        
        assert "storage_path" in str(exc_info.value)
    
    def test_invalid_vram_budget(self, temp_storage_dir):
        """Test invalid VRAM budget handling."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config(storage_path=temp_storage_dir, vram_budget_gb=-1.0)
        
        assert "vram_budget_gb" in str(exc_info.value)
    
    def test_invalid_backend_type(self, temp_storage_dir):
        """Test invalid backend type handling."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config(storage_path=temp_storage_dir, backend="invalid_backend")
        
        assert "backend" in str(exc_info.value)
    
    def test_invalid_compression_level(self, temp_storage_dir):
        """Test invalid compression level handling."""
        with pytest.raises(ConfigurationError) as exc_info:
            Config(
                storage_path=temp_storage_dir,
                compression_enabled=True,
                compression_level=15  # Invalid: should be 1-9
            )
        
        assert "compression_level" in str(exc_info.value)
    
    def test_string_backend_conversion(self, temp_storage_dir):
        """Test string to BackendType conversion."""
        config = Config(storage_path=temp_storage_dir, backend="mmap")
        assert config.backend == BackendType.MMAP
        
        config = Config(storage_path=temp_storage_dir, backend="gpudirect")
        assert config.backend == BackendType.GPUDIRECT
    
    def test_auto_vram_detection(self, temp_storage_dir):
        """Test automatic VRAM budget detection."""
        config = Config(storage_path=temp_storage_dir, vram_budget_gb=None)
        
        # Should auto-detect and set a positive value
        assert config.vram_budget_gb > 0
    
    def test_device_configuration(self, temp_storage_dir):
        """Test device configuration."""
        # Test default device
        config = Config(storage_path=temp_storage_dir)
        assert config.device is not None
        
        # Test explicit device
        config = Config(storage_path=temp_storage_dir, device="cpu")
        assert config.device == torch.device("cpu")
        
        if torch.cuda.is_available():
            config = Config(storage_path=temp_storage_dir, device="cuda:0")
            assert config.device == torch.device("cuda:0")
    
    def test_config_serialization(self, temp_storage_dir):
        """Test configuration serialization/deserialization."""
        original_config = Config(
            storage_path=temp_storage_dir,
            vram_budget_gb=8.0,
            backend=BackendType.MMAP,
            prefetch_strategy=PrefetchStrategy.ADAPTIVE,
            compression_enabled=True,
            compression_level=7,
            num_io_threads=6
        )
        
        # Convert to dict
        config_dict = original_config.to_dict()
        
        # Verify dict contents
        assert config_dict["vram_budget_gb"] == 8.0
        assert config_dict["backend"] == "mmap"
        assert config_dict["prefetch_strategy"] == "adaptive"
        assert config_dict["compression_enabled"] is True
        assert config_dict["compression_level"] == 7
        assert config_dict["num_io_threads"] == 6
        
        # Recreate from dict
        restored_config = Config.from_dict(config_dict)
        
        # Verify restoration
        assert restored_config.vram_budget_gb == original_config.vram_budget_gb
        assert restored_config.backend == original_config.backend
        assert restored_config.prefetch_strategy == original_config.prefetch_strategy
        assert restored_config.compression_enabled == original_config.compression_enabled
        assert restored_config.compression_level == original_config.compression_level
        assert restored_config.num_io_threads == original_config.num_io_threads
    
    def test_backend_priority(self, temp_storage_dir):
        """Test backend priority determination."""
        # Test auto backend priority
        config = Config(storage_path=temp_storage_dir, backend=BackendType.AUTO)
        priority = config.get_backend_priority()
        
        assert isinstance(priority, list)
        assert len(priority) > 0
        assert BackendType.MMAP in priority  # mmap should always be available
    
    def test_memory_pressure_modes(self, temp_storage_dir):
        """Test memory pressure mode configuration."""
        modes = [
            MemoryPressureMode.STRICT,
            MemoryPressureMode.ADAPTIVE,
            MemoryPressureMode.LENIENT
        ]
        
        for mode in modes:
            config = Config(storage_path=temp_storage_dir, memory_pressure_mode=mode)
            assert config.memory_pressure_mode == mode
    
    def test_prefetch_strategies(self, temp_storage_dir):
        """Test prefetch strategy configuration."""
        strategies = [
            PrefetchStrategy.NONE,
            PrefetchStrategy.NEXT_LAYER,
            PrefetchStrategy.ADAPTIVE,
            PrefetchStrategy.AGGRESSIVE
        ]
        
        for strategy in strategies:
            config = Config(storage_path=temp_storage_dir, prefetch_strategy=strategy)
            assert config.prefetch_strategy == strategy
    
    def test_temp_directory_creation(self, temp_storage_dir):
        """Test temporary directory creation."""
        config = Config(storage_path=temp_storage_dir)
        
        # Check temp directory was created
        assert config.temp_dir.exists()
        assert config.temp_dir.is_dir()
        assert config.temp_dir.parent == config.storage_path
    
    def test_custom_temp_directory(self, temp_storage_dir):
        """Test custom temporary directory."""
        custom_temp = temp_storage_dir / "custom_temp"
        config = Config(storage_path=temp_storage_dir, temp_dir=custom_temp)
        
        assert config.temp_dir == custom_temp
        assert config.temp_dir.exists()
    
    def test_metadata_handling(self, temp_storage_dir):
        """Test metadata handling."""
        metadata = {
            "user": "test_user",
            "experiment": "config_test",
            "version": "1.0"
        }
        
        config = Config(storage_path=temp_storage_dir, metadata=metadata)
        assert config.metadata == metadata
    
    def test_create_default_config(self, temp_storage_dir):
        """Test default configuration creation helper."""
        config = create_default_config(temp_storage_dir)
        
        assert isinstance(config, Config)
        assert config.storage_path == temp_storage_dir
        assert config.backend == BackendType.AUTO
        assert config.vram_budget_gb > 0


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_vram_budget(self, temp_storage_dir):
        """Test handling of very small VRAM budget."""
        config = Config(storage_path=temp_storage_dir, vram_budget_gb=0.001)
        assert config.vram_budget_gb == 0.001
    
    def test_large_vram_budget(self, temp_storage_dir):
        """Test handling of very large VRAM budget."""
        config = Config(storage_path=temp_storage_dir, vram_budget_gb=1000.0)
        assert config.vram_budget_gb == 1000.0
    
    def test_zero_io_threads(self, temp_storage_dir):
        """Test invalid number of I/O threads."""
        with pytest.raises(ConfigurationError):
            Config(storage_path=temp_storage_dir, num_io_threads=0)
    
    def test_negative_chunk_size(self, temp_storage_dir):
        """Test invalid chunk size."""
        with pytest.raises(ConfigurationError):
            Config(storage_path=temp_storage_dir, chunk_size_mb=-1)
    
    def test_invalid_cache_size(self, temp_storage_dir):
        """Test invalid cache size."""
        with pytest.raises(ConfigurationError):
            Config(storage_path=temp_storage_dir, cache_size_layers=0)
    
    def test_config_immutability_after_validation(self, temp_storage_dir):
        """Test that certain config properties can't be changed after validation."""
        config = Config(storage_path=temp_storage_dir)
        
        # These should be set during __post_init__
        original_storage_path = config.storage_path
        original_temp_dir = config.temp_dir
        
        # The paths should be resolved and not changeable
        assert original_storage_path.is_absolute()
        assert original_temp_dir.is_absolute()


@pytest.mark.slow
class TestConfigPerformance:
    """Performance-related configuration tests."""
    
    def test_config_creation_performance(self, temp_storage_dir):
        """Test that config creation is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            Config(storage_path=temp_storage_dir)
        end_time = time.time()
        
        # Should be able to create 100 configs in less than 1 second
        assert end_time - start_time < 1.0
    
    def test_config_serialization_performance(self, temp_storage_dir):
        """Test that config serialization is reasonably fast."""
        import time
        
        config = Config(storage_path=temp_storage_dir)
        
        start_time = time.time()
        for _ in range(1000):
            config_dict = config.to_dict()
            Config.from_dict(config_dict)
        end_time = time.time()
        
        # Should be able to serialize/deserialize 1000 times in less than 1 second
        assert end_time - start_time < 1.0
