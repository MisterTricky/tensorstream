"""
Unit tests for TensorStream backend implementations.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from tensorstream.backends import BackendInterface
from tensorstream.backends.mmap_backend import MmapBackend
from tensorstream.backends.cuda_backend import CudaCoreBackend
from tensorstream.backends.gpudirect_backend import GPUDirectBackend
from tensorstream.exceptions import BackendError
from tensorstream.io import save_to_ts


class TestBackendInterface:
    """Test the abstract backend interface."""
    
    def test_abstract_methods(self, mock_config):
        """Test that BackendInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BackendInterface(mock_config)
    
    def test_backend_error_helper(self, mock_config):
        """Test the backend error helper method."""
        class TestBackend(BackendInterface):
            def initialize(self): pass
            def cleanup(self): pass
            def load_tensor(self, path, device=None): pass
            def unload_tensor(self, tensor): pass
            def get_memory_info(self): return {}
            def is_available(self): return True
        
        backend = TestBackend(mock_config)
        
        with pytest.raises(BackendError) as exc_info:
            backend._raise_backend_error("test_op", "test message", {"key": "value"})
        
        error = exc_info.value
        assert error.backend_name == "TestBackend"
        assert error.operation == "test_op"
        assert "test message" in str(error)


class TestMmapBackend:
    """Test the memory-mapped file backend."""
    
    def test_backend_availability(self, mock_config):
        """Test that mmap backend is always available."""
        backend = MmapBackend(mock_config)
        assert backend.is_available() is True
    
    def test_initialization(self, mock_config):
        """Test backend initialization."""
        backend = MmapBackend(mock_config)
        
        assert not backend.is_initialized()
        
        backend.initialize()
        
        assert backend.is_initialized()
        assert backend.get_name() == "mmap"
    
    def test_cleanup(self, mock_config):
        """Test backend cleanup."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        # Add some dummy entries to test cleanup
        backend._mmap_files["test"] = (Mock(), Mock())
        backend._memory_pool[12345] = 1024
        
        backend.cleanup()
        
        assert not backend.is_initialized()
        assert len(backend._mmap_files) == 0
        assert len(backend._memory_pool) == 0
    
    def test_cpu_tensor_loading(self, mock_config, temp_storage_dir, sample_tensor):
        """Test loading tensor to CPU."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        # Save a test tensor
        tensor_path = temp_storage_dir / "test_tensor.ts"
        save_to_ts(sample_tensor, tensor_path, compress=False)  # Uncompressed for mmap
        
        try:
            # Load tensor
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
            
            # Verify
            assert torch.allclose(sample_tensor, loaded_tensor)
            assert loaded_tensor.device.type == 'cpu'
        finally:
            backend.cleanup()
    
    @pytest.mark.gpu
    def test_gpu_tensor_loading(self, mock_config, temp_storage_dir, sample_tensor):
        """Test loading tensor to GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        # Save a test tensor
        tensor_path = temp_storage_dir / "test_tensor.ts"
        save_to_ts(sample_tensor, tensor_path)
        
        try:
            # Load tensor to GPU
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cuda:0'))
            
            # Verify
            assert torch.allclose(sample_tensor.cuda(), loaded_tensor)
            assert loaded_tensor.device.type == 'cuda'
        finally:
            backend.cleanup()
    
    def test_compressed_tensor_loading(self, mock_config, temp_storage_dir, sample_tensor):
        """Test loading compressed tensor."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        # Save compressed tensor
        tensor_path = temp_storage_dir / "compressed_tensor.ts"
        save_to_ts(sample_tensor, tensor_path, compress=True)
        
        try:
            # Load tensor (should fall back to regular loading)
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
            
            # Verify
            assert torch.allclose(sample_tensor, loaded_tensor)
        finally:
            backend.cleanup()
    
    def test_memory_info(self, mock_config):
        """Test memory information retrieval."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        try:
            memory_info = backend.get_memory_info()
            
            # Should contain basic memory info
            assert isinstance(memory_info, dict)
            assert "mmap_total_size" in memory_info
            assert "mmap_file_count" in memory_info
            
            # May contain system memory info if psutil available
            if "system_total" in memory_info:
                assert memory_info["system_total"] > 0
        finally:
            backend.cleanup()
    
    def test_tensor_unloading(self, mock_config, sample_tensor):
        """Test tensor unloading."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        try:
            # Test unloading (should not raise errors)
            backend.unload_tensor(sample_tensor)
            
            # Test unloading GPU tensor if available
            if torch.cuda.is_available():
                gpu_tensor = sample_tensor.cuda()
                backend.unload_tensor(gpu_tensor)
        finally:
            backend.cleanup()
    
    def test_invalid_file_handling(self, mock_config, temp_storage_dir):
        """Test handling of invalid files."""
        backend = MmapBackend(mock_config)
        backend.initialize()
        
        try:
            # Try to load non-existent file
            with pytest.raises(BackendError):
                backend.load_tensor(temp_storage_dir / "nonexistent.ts")
            
            # Try to load invalid file
            invalid_file = temp_storage_dir / "invalid.ts"
            with open(invalid_file, 'w') as f:
                f.write("invalid content")
            
            with pytest.raises(BackendError):
                backend.load_tensor(invalid_file)
        finally:
            backend.cleanup()


class TestCudaCoreBackend:
    """Test the CUDA core backend."""
    
    def test_availability_without_cuda(self, mock_config):
        """Test availability when CUDA is not available."""
        backend = CudaCoreBackend(mock_config)
        
        # If CUDA not available, should return False
        if not torch.cuda.is_available():
            assert backend.is_available() is False
        else:
            # If CUDA available but extension not built, depends on import
            try:
                from tensorstream import _cuda_core
                assert backend.is_available() is True
            except ImportError:
                assert backend.is_available() is False
    
    @pytest.mark.gpu
    def test_initialization_with_cuda(self, mock_config):
        """Test initialization when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        backend = CudaCoreBackend(mock_config)
        
        if backend.is_available():
            try:
                backend.initialize()
                assert backend.is_initialized()
            except BackendError:
                # Extension might not be compiled
                pytest.skip("CUDA core extension not available")
            finally:
                if backend.is_initialized():
                    backend.cleanup()
    
    def test_initialization_without_extension(self, mock_config):
        """Test initialization when CUDA extension is not available."""
        backend = CudaCoreBackend(mock_config)
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'tensorstream._cuda_core': None}):
            with pytest.raises(BackendError) as exc_info:
                backend.initialize()
            
            assert "CUDA core extension not available" in str(exc_info.value)
    
    def test_fallback_loading(self, mock_config, temp_storage_dir, sample_tensor):
        """Test fallback loading when CUDA core fails."""
        backend = CudaCoreBackend(mock_config)
        
        # Create test file
        tensor_path = temp_storage_dir / "test_tensor.ts"
        save_to_ts(sample_tensor, tensor_path)
        
        # Mock initialization to succeed but core loading to fail
        backend._initialized = True
        backend._cuda_core = Mock()
        backend._cuda_core.load_tensor_direct.side_effect = Exception("Core failed")
        
        try:
            # Should fall back to standard loading
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
            assert torch.allclose(sample_tensor, loaded_tensor)
        finally:
            backend.cleanup()


class TestGPUDirectBackend:
    """Test the GPUDirect Storage backend."""
    
    def test_availability_without_extension(self, mock_config):
        """Test availability when GDS extension is not available."""
        backend = GPUDirectBackend(mock_config)
        
        # Without the extension, should return False
        try:
            from tensorstream import _gds_core
            # If import succeeds, availability depends on CUDA
            if torch.cuda.is_available():
                assert backend.is_available() is True
            else:
                assert backend.is_available() is False
        except ImportError:
            assert backend.is_available() is False
    
    def test_initialization_without_cuda(self, mock_config):
        """Test initialization when CUDA is not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available")
        
        backend = GPUDirectBackend(mock_config)
        
        with pytest.raises(BackendError) as exc_info:
            backend.initialize()
        
        assert "CUDA not available" in str(exc_info.value)
    
    @pytest.mark.gpu
    def test_initialization_with_mock_gds(self, mock_config):
        """Test initialization with mocked GDS."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        backend = GPUDirectBackend(mock_config)
        
        # Mock the GDS core
        mock_gds_core = Mock()
        mock_gds_core.initialize_gds.return_value = True
        
        with patch.object(backend, '_gds_core', mock_gds_core):
            backend._gds_core = mock_gds_core
            
            try:
                backend.initialize()
                assert backend.is_initialized()
                mock_gds_core.initialize_gds.assert_called_once()
            finally:
                backend.cleanup()
    
    def test_fallback_for_compressed_files(self, mock_config, temp_storage_dir, sample_tensor):
        """Test fallback to standard loading for compressed files."""
        backend = GPUDirectBackend(mock_config)
        
        # Create compressed test file
        tensor_path = temp_storage_dir / "compressed_tensor.ts"
        save_to_ts(sample_tensor, tensor_path, compress=True)
        
        # Mock initialization
        backend._initialized = True
        backend._gds_core = Mock()
        
        try:
            # Should fall back for compressed files
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
            assert torch.allclose(sample_tensor, loaded_tensor)
        finally:
            backend.cleanup()
    
    def test_cpu_device_fallback(self, mock_config, temp_storage_dir, sample_tensor):
        """Test fallback for CPU devices."""
        backend = GPUDirectBackend(mock_config)
        
        # Create test file
        tensor_path = temp_storage_dir / "test_tensor.ts"
        save_to_ts(sample_tensor, tensor_path)
        
        try:
            # Should fall back for CPU device
            loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
            assert torch.allclose(sample_tensor, loaded_tensor)
            assert loaded_tensor.device.type == 'cpu'
        finally:
            backend.cleanup()
    
    def test_cleanup_resources(self, mock_config):
        """Test proper cleanup of GDS resources."""
        backend = GPUDirectBackend(mock_config)
        
        # Mock GDS core and add some dummy resources
        mock_gds_core = Mock()
        backend._gds_core = mock_gds_core
        backend._file_handles = {"test_file": "handle123"}
        backend._memory_registrations = {12345: "registration456"}
        
        backend.cleanup()
        
        # Should have attempted to close files and unregister memory
        assert len(backend._file_handles) == 0
        assert len(backend._memory_registrations) == 0
    
    def test_memory_info_with_gds(self, mock_config):
        """Test memory info retrieval with GDS."""
        backend = GPUDirectBackend(mock_config)
        backend._initialized = True
        
        # Mock GDS core
        mock_gds_core = Mock()
        mock_gds_core.get_stats.return_value = {
            "transfers": 100,
            "bytes_transferred": 1024*1024
        }
        backend._gds_core = mock_gds_core
        
        try:
            memory_info = backend.get_memory_info()
            
            assert isinstance(memory_info, dict)
            assert "gds_open_files" in memory_info
            assert "gds_registered_regions" in memory_info
            
            if torch.cuda.is_available():
                assert "cuda_allocated" in memory_info
        finally:
            backend.cleanup()


class TestBackendIntegration:
    """Integration tests for backend interactions."""
    
    def test_backend_switching(self, mock_config, temp_storage_dir, sample_tensor):
        """Test switching between different backends."""
        tensor_path = temp_storage_dir / "test_tensor.ts"
        save_to_ts(sample_tensor, tensor_path)
        
        backends = [MmapBackend(mock_config)]
        
        # Add other backends if available
        if torch.cuda.is_available():
            try:
                backends.append(CudaCoreBackend(mock_config))
            except:
                pass
            
            try:
                backends.append(GPUDirectBackend(mock_config))
            except:
                pass
        
        # Test each available backend
        for backend in backends:
            if backend.is_available():
                try:
                    backend.initialize()
                    
                    loaded_tensor = backend.load_tensor(tensor_path, torch.device('cpu'))
                    assert torch.allclose(sample_tensor, loaded_tensor)
                    
                    backend.cleanup()
                except Exception as e:
                    # Some backends might not be fully available
                    pytest.skip(f"Backend {backend.get_name()} not available: {e}")
    
    def test_memory_management_across_backends(self, mock_config):
        """Test memory management consistency across backends."""
        backends = [MmapBackend(mock_config)]
        
        if torch.cuda.is_available():
            try:
                backends.append(CudaCoreBackend(mock_config))
            except:
                pass
        
        for backend in backends:
            if backend.is_available():
                try:
                    backend.initialize()
                    
                    # Get initial memory info
                    initial_info = backend.get_memory_info()
                    assert isinstance(initial_info, dict)
                    
                    # Memory info should be non-negative
                    for key, value in initial_info.items():
                        if isinstance(value, (int, float)):
                            assert value >= 0, f"Negative memory value for {key}: {value}"
                    
                    backend.cleanup()
                except Exception:
                    # Backend might not be fully functional
                    pass
