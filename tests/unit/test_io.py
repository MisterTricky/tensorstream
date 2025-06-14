"""
Unit tests for TensorStream I/O operations.
"""

import pytest
import numpy as np
from pathlib import Path

import torch

from tensorstream.io import (
    save_to_ts,
    load_from_ts,
    read_ts_header,
    get_ts_file_info,
    shard_tensor,
    combine_shards,
    TSFileHeader,
    TORCH_TO_NUMPY_DTYPE,
    NUMPY_TO_TORCH_DTYPE,
)
from tensorstream.exceptions import StorageError


class TestTSFileFormat:
    """Test the .ts file format implementation."""
    
    def test_basic_tensor_save_load(self, temp_storage_dir, sample_tensor):
        """Test basic tensor save and load operations."""
        file_path = temp_storage_dir / "test_tensor.ts"
        
        # Save tensor
        save_to_ts(sample_tensor, file_path)
        
        # Load tensor
        loaded_tensor = load_from_ts(file_path)
        
        # Verify tensors are equal
        assert torch.allclose(sample_tensor, loaded_tensor)
        assert sample_tensor.shape == loaded_tensor.shape
        assert sample_tensor.dtype == loaded_tensor.dtype
    
    def test_tensor_with_metadata(self, temp_storage_dir, sample_tensor):
        """Test tensor save/load with metadata."""
        file_path = temp_storage_dir / "test_with_metadata.ts"
        metadata = {
            "layer_name": "test_layer",
            "model": "test_model",
            "version": "1.0",
            "params": {"learning_rate": 0.001}
        }
        
        # Save with metadata
        save_to_ts(sample_tensor, file_path, metadata=metadata)
        
        # Load and verify
        loaded_tensor = load_from_ts(file_path)
        header = read_ts_header(file_path)
        
        assert torch.allclose(sample_tensor, loaded_tensor)
        assert header.metadata == metadata
    
    def test_compressed_tensor(self, temp_storage_dir, large_tensor):
        """Test compressed tensor save/load."""
        file_path = temp_storage_dir / "test_compressed.ts"
        
        # Save with compression
        save_to_ts(large_tensor, file_path, compress=True, compression_level=6)
        
        # Load and verify
        loaded_tensor = load_from_ts(file_path)
        header = read_ts_header(file_path)
        
        assert torch.allclose(large_tensor, loaded_tensor)
        assert header.compressed is True
        assert header.compression_level == 6
        assert header.compressed_size < header.data_size
    
    def test_checksum_verification(self, temp_storage_dir, sample_tensor):
        """Test checksum verification."""
        file_path = temp_storage_dir / "test_checksum.ts"
        
        # Save with checksum
        save_to_ts(sample_tensor, file_path, verify_checksum=True)
        
        # Load with verification
        loaded_tensor = load_from_ts(file_path, verify_checksum=True)
        
        assert torch.allclose(sample_tensor, loaded_tensor)
        
        # Verify checksum is stored
        header = read_ts_header(file_path)
        assert header.checksum is not None
    
    def test_different_dtypes(self, temp_storage_dir):
        """Test different tensor data types."""
        dtypes = [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]
        
        if torch.cuda.is_available():
            dtypes.append(torch.bfloat16)
        
        for dtype in dtypes:
            tensor = torch.randn(10, 10).to(dtype)
            if dtype == torch.bool:
                tensor = tensor > 0  # Convert to boolean
            
            file_path = temp_storage_dir / f"test_{str(dtype).split('.')[-1]}.ts"
            
            # Save and load
            save_to_ts(tensor, file_path)
            loaded_tensor = load_from_ts(file_path)
            
            if dtype == torch.bfloat16:
                # bfloat16 has limited precision
                assert torch.allclose(tensor.float(), loaded_tensor.float(), rtol=1e-2)
            else:
                assert torch.allclose(tensor, loaded_tensor)
            assert tensor.dtype == loaded_tensor.dtype
    
    def test_different_shapes(self, temp_storage_dir):
        """Test different tensor shapes."""
        shapes = [
            (100,),           # 1D
            (50, 50),         # 2D
            (10, 10, 10),     # 3D
            (5, 5, 5, 5),     # 4D
            (1,),             # Single element
            (1000, 1),        # Tall and narrow
            (1, 1000),        # Wide and short
        ]
        
        for i, shape in enumerate(shapes):
            tensor = torch.randn(shape)
            file_path = temp_storage_dir / f"test_shape_{i}.ts"
            
            save_to_ts(tensor, file_path)
            loaded_tensor = load_from_ts(file_path)
            
            assert tensor.shape == loaded_tensor.shape
            assert torch.allclose(tensor, loaded_tensor)
    
    def test_device_handling(self, temp_storage_dir, sample_tensor):
        """Test device handling during save/load."""
        file_path = temp_storage_dir / "test_device.ts"
        
        # Save tensor (should work regardless of device)
        save_to_ts(sample_tensor, file_path)
        
        # Load to CPU
        cpu_tensor = load_from_ts(file_path, device=torch.device('cpu'))
        assert cpu_tensor.device.type == 'cpu'
        assert torch.allclose(sample_tensor.cpu(), cpu_tensor)
        
        # Load to CUDA if available
        if torch.cuda.is_available():
            cuda_tensor = load_from_ts(file_path, device=torch.device('cuda:0'))
            assert cuda_tensor.device.type == 'cuda'
            assert torch.allclose(sample_tensor.cpu(), cuda_tensor.cpu())


class TestTSFileHeader:
    """Test the TSFileHeader class."""
    
    def test_header_serialization(self):
        """Test header serialization and deserialization."""
        header = TSFileHeader()
        header.tensor_dtype = "torch.float32"
        header.tensor_shape = [100, 50]
        header.compressed = True
        header.compression_level = 6
        header.data_size = 20000
        header.compressed_size = 15000
        header.metadata = {"test": "value"}
        
        # Serialize
        header_bytes = header.to_bytes()
        
        # Deserialize
        restored_header = TSFileHeader.from_bytes(header_bytes)
        
        # Verify
        assert restored_header.tensor_dtype == header.tensor_dtype
        assert restored_header.tensor_shape == header.tensor_shape
        assert restored_header.compressed == header.compressed
        assert restored_header.compression_level == header.compression_level
        assert restored_header.data_size == header.data_size
        assert restored_header.compressed_size == header.compressed_size
        assert restored_header.metadata == header.metadata
    
    def test_header_with_checksum(self):
        """Test header with checksum."""
        header = TSFileHeader()
        header.tensor_dtype = "torch.float32"
        header.tensor_shape = [10, 10]
        header.checksum = b'0' * 32  # Dummy checksum
        header.data_size = 400
        
        header_bytes = header.to_bytes()
        restored_header = TSFileHeader.from_bytes(header_bytes)
        
        assert restored_header.checksum == header.checksum


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_get_file_info(self, temp_storage_dir, sample_tensor):
        """Test getting file information."""
        file_path = temp_storage_dir / "test_info.ts"
        metadata = {"layer": "test", "size": "medium"}
        
        save_to_ts(sample_tensor, file_path, metadata=metadata, compress=True)
        
        info = get_ts_file_info(file_path)
        
        assert info["dtype"] == str(sample_tensor.dtype)
        assert info["shape"] == list(sample_tensor.shape)
        assert info["compressed"] is True
        assert info["metadata"] == metadata
        assert info["data_size"] > 0
        assert info["compressed_size"] > 0
        assert info["compressed_size"] < info["data_size"]
        assert info["file_size"] > 0
    
    def test_nonexistent_file(self, temp_storage_dir):
        """Test handling of nonexistent files."""
        file_path = temp_storage_dir / "nonexistent.ts"
        
        with pytest.raises(StorageError):
            read_ts_header(file_path)
        
        with pytest.raises(StorageError):
            load_from_ts(file_path)
        
        with pytest.raises(StorageError):
            get_ts_file_info(file_path)
    
    def test_invalid_file_format(self, temp_storage_dir):
        """Test handling of invalid file format."""
        file_path = temp_storage_dir / "invalid.ts"
        
        # Write invalid data
        with open(file_path, 'wb') as f:
            f.write(b"invalid file content")
        
        with pytest.raises(StorageError):
            read_ts_header(file_path)
        
        with pytest.raises(StorageError):
            load_from_ts(file_path)


class TestTensorSharding:
    """Test tensor sharding utilities."""
    
    def test_shard_and_combine(self, large_tensor):
        """Test tensor sharding and combining."""
        num_shards = 5
        
        # Shard tensor
        shards = shard_tensor(large_tensor, num_shards)
        
        assert len(shards) <= num_shards  # Might be fewer if tensor is small
        assert all(isinstance(shard, torch.Tensor) for shard in shards)
        
        # Verify total elements
        total_elements = sum(shard.numel() for shard in shards)
        assert total_elements == large_tensor.numel()
        
        # Combine shards
        combined_tensor = combine_shards(shards)
        
        # Verify reconstruction
        assert torch.allclose(large_tensor, combined_tensor)
        assert large_tensor.shape == combined_tensor.shape
    
    def test_single_shard(self, sample_tensor):
        """Test sharding with single shard."""
        shards = shard_tensor(sample_tensor, 1)
        
        assert len(shards) == 1
        assert torch.allclose(sample_tensor, shards[0])
        
        combined = combine_shards(shards)
        assert torch.allclose(sample_tensor, combined)
    
    def test_more_shards_than_elements(self):
        """Test sharding with more shards than first dimension."""
        tensor = torch.randn(3, 10)  # Only 3 elements in first dim
        
        shards = shard_tensor(tensor, 10)  # Request 10 shards
        
        assert len(shards) <= 3  # Should get at most 3 shards
        
        combined = combine_shards(shards)
        assert torch.allclose(tensor, combined)
    
    def test_empty_tensor_sharding(self):
        """Test sharding empty tensor."""
        empty_tensor = torch.empty(0, 10)
        
        shards = shard_tensor(empty_tensor, 3)
        
        assert len(shards) == 0 or (len(shards) == 1 and shards[0].numel() == 0)


class TestDataTypeConversion:
    """Test data type conversion utilities."""
    
    def test_torch_to_numpy_mapping(self):
        """Test torch to numpy dtype mapping."""
        # Test that all mappings work
        for torch_dtype, numpy_dtype in TORCH_TO_NUMPY_DTYPE.items():
            tensor = torch.tensor([1.0], dtype=torch_dtype)
            # Should not raise an exception
            if torch_dtype == torch.bfloat16:
                # bfloat16 requires special handling
                np_array = tensor.float().cpu().numpy()
            else:
                np_array = tensor.cpu().numpy()
    
    def test_numpy_to_torch_mapping(self):
        """Test numpy to torch dtype mapping."""
        # Test reverse mapping
        for numpy_dtype, torch_dtype in NUMPY_TO_TORCH_DTYPE.items():
            np_array = np.array([1.0], dtype=numpy_dtype)
            tensor = torch.from_numpy(np_array)
            # Verify dtype compatibility
            assert tensor.dtype in TORCH_TO_NUMPY_DTYPE


@pytest.mark.slow
class TestIOPerformance:
    """Performance tests for I/O operations."""
    
    def test_large_tensor_performance(self, temp_storage_dir):
        """Test performance with large tensors."""
        import time
        
        # Create a large tensor (100MB)
        large_tensor = torch.randn(5000, 5000, dtype=torch.float32)
        file_path = temp_storage_dir / "large_tensor.ts"
        
        # Measure save time
        start_time = time.time()
        save_to_ts(large_tensor, file_path)
        save_time = time.time() - start_time
        
        # Measure load time
        start_time = time.time()
        loaded_tensor = load_from_ts(file_path)
        load_time = time.time() - start_time
        
        # Verify correctness
        assert torch.allclose(large_tensor, loaded_tensor)
        
        # Performance assertions (adjust based on hardware)
        assert save_time < 10.0  # Should save 100MB in less than 10 seconds
        assert load_time < 10.0  # Should load 100MB in less than 10 seconds
    
    def test_compression_performance(self, temp_storage_dir):
        """Test compression performance vs file size."""
        # Create a tensor with patterns (compresses well)
        pattern_tensor = torch.zeros(1000, 1000)
        pattern_tensor[::2] = 1.0  # Checkerboard pattern
        
        uncompressed_path = temp_storage_dir / "uncompressed.ts"
        compressed_path = temp_storage_dir / "compressed.ts"
        
        # Save without compression
        save_to_ts(pattern_tensor, uncompressed_path, compress=False)
        
        # Save with compression
        save_to_ts(pattern_tensor, compressed_path, compress=True)
        
        # Compare file sizes
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        
        # Compression should reduce file size significantly for patterned data
        compression_ratio = compressed_size / uncompressed_size
        assert compression_ratio < 0.5  # At least 50% compression
        
        # Verify both load correctly
        uncompressed_loaded = load_from_ts(uncompressed_path)
        compressed_loaded = load_from_ts(compressed_path)
        
        assert torch.allclose(pattern_tensor, uncompressed_loaded)
        assert torch.allclose(pattern_tensor, compressed_loaded)
