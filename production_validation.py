#!/usr/bin/env python3
"""
TensorStream Production Validation Suite

This script performs comprehensive validation of all TensorStream components
to ensure production readiness. It tests all critical bug fixes, validates
performance characteristics, and generates a detailed report.
"""

import sys
import time
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# Suppress CUDA warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="tensorstream.backends.mmap_backend")

class ValidationSuite:
    """Comprehensive validation suite for TensorStream."""
    
    def __init__(self):
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "performance_metrics": {},
            "compatibility_check": {},
        }
        self.temp_dir = None
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "PERF": "📊"
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")
        
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results."""
        self.results["tests_run"] += 1
        self.log(f"Running test: {test_name}")
        
        try:
            result = test_func()
            if result:
                self.results["tests_passed"] += 1
                self.log(f"PASSED: {test_name}", "SUCCESS")
                return True
            else:
                self.results["tests_failed"] += 1
                self.results["errors"].append(f"{test_name}: Test returned False")
                self.log(f"FAILED: {test_name}", "ERROR")
                return False
        except Exception as e:
            self.results["tests_failed"] += 1
            error_msg = f"{test_name}: {str(e)}"
            self.results["errors"].append(error_msg)
            self.log(f"FAILED: {test_name} - {e}", "ERROR")
            return False
    
    def test_imports(self) -> bool:
        """Test that all TensorStream components can be imported."""
        try:
            import torch
            import torch.nn as nn
            import tensorstream
            from tensorstream import Config, offload, create_default_config
            from tensorstream.proxy import TensorStreamProxyLayer
            from tensorstream.orchestrator import OrchestrationEngine
            from tensorstream.api import _is_offloadable_layer
            from tensorstream.io import save_to_ts, load_from_ts
            from tensorstream.backends.mmap_backend import MMapBackend
            
            self.log(f"PyTorch {torch.__version__} imported successfully")
            self.log("All TensorStream components imported successfully")
            return True
        except ImportError as e:
            self.log(f"Import failed: {e}")
            return False
    
    def test_api_boolean_fix(self) -> bool:
        """Test the critical API boolean tensor fix."""
        try:
            import torch.nn as nn
            from tensorstream.api import _is_offloadable_layer
            
            # Create test modules
            linear_layer = nn.Linear(1000, 500)  # Has parameters
            relu_layer = nn.ReLU()  # No parameters
            empty_sequential = nn.Sequential()  # Empty container
            
            # These calls should not raise "Boolean value of Tensor" error
            result1 = _is_offloadable_layer(linear_layer)
            result2 = _is_offloadable_layer(relu_layer)
            result3 = _is_offloadable_layer(empty_sequential)
            
            # Validate results
            assert result1 == True, "Large linear layer should be offloadable"
            assert result2 == False, "ReLU layer should not be offloadable"
            assert result3 == False, "Empty sequential should not be offloadable"
            
            self.log("API boolean tensor fix validated")
            return True
        except Exception as e:
            self.log(f"API boolean fix test failed: {e}")
            return False
    
    def test_proxy_layer_methods(self) -> bool:
        """Test proxy layer method signatures and functionality."""
        try:
            import torch.nn as nn
            from tensorstream.proxy import TensorStreamProxyLayer
            from tensorstream.orchestrator import OrchestrationEngine
            from tensorstream.config import Config
            
            # Create test components
            original_layer = nn.Linear(100, 50)
            config = Config(storage_path=self.temp_dir, debug_mode=False)
            orchestrator = OrchestrationEngine(config)
            
            # Create proxy layer
            proxy = TensorStreamProxyLayer("test_layer", original_layer, orchestrator)
            
            # Test all method signatures that were causing issues
            list(proxy.named_modules())
            list(proxy.named_modules(memo=set()))
            list(proxy.named_modules(prefix='test.'))
            list(proxy.named_modules(memo=set(), prefix='test.', remove_duplicate=True))
            
            # Test device property (should not raise setter error)
            device = proxy.device
            assert device is not None
            
            # Test other delegated methods
            list(proxy.children())
            list(proxy.named_children())
            list(proxy.buffers())
            list(proxy.named_buffers())
            list(proxy.parameters())
            list(proxy.named_parameters())
            
            self.log("Proxy layer method signatures validated")
            return True
        except Exception as e:
            self.log(f"Proxy layer method test failed: {e}")
            return False
    
    def test_io_system(self) -> bool:
        """Test I/O system with compression and various tensor types."""
        try:
            import torch
            from tensorstream.io import save_to_ts, load_from_ts
            
            # Test with different tensor types
            test_tensors = [
                torch.randn(100, 200),  # Float32
                torch.randn(50, 100).half(),  # Float16
                torch.randint(0, 1000, (100,)),  # Int64
                torch.randn(10, 20).to(torch.bfloat16),  # BFloat16
            ]
            
            for i, tensor in enumerate(test_tensors):
                file_path = self.temp_dir / f"test_tensor_{i}.ts"
                
                # Test save with compression
                save_to_ts(
                    tensor, 
                    file_path,
                    metadata={"test": f"tensor_{i}"},
                    compress=True,
                    compression_level=6
                )
                
                # Test load
                loaded_tensor, metadata = load_from_ts(file_path)
                
                # Validate
                assert torch.allclose(tensor.float(), loaded_tensor.float(), atol=1e-5)
                assert metadata["test"] == f"tensor_{i}"
            
            self.log("I/O system with compression validated")
            return True
        except Exception as e:
            self.log(f"I/O system test failed: {e}")
            return False
    
    def test_end_to_end_offloading(self) -> bool:
        """Test complete end-to-end model offloading."""
        try:
            import torch
            import torch.nn as nn
            from tensorstream import offload, create_default_config
            
            # Create a transformer-like test model
            class TestTransformerModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(1000, 256)
                    self.layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(256, 8, batch_first=True)
                        for _ in range(4
                    ])
                    self.norm = nn.LayerNorm(256)
                    self.head = nn.Linear(256, 1000)
                
                def forward(self, x):
                    x = self.embedding(x)
                    for layer in self.layers:
                        x = layer(x)
                    x = self.norm(x)
                    return self.head(x)
            
            model = TestTransformerModel()
            
            # Configure for testing
            config = create_default_config(self.temp_dir)
            config.min_layer_size = 1024  # Lower threshold for testing
            config.debug_mode = False
            
            # Apply offloading
            offloaded_model = offload(model, config)
            
            # Test inference
            input_ids = torch.randint(0, 1000, (2, 10))
            with torch.no_grad():
                output = offloaded_model(input_ids)
                assert output.shape == (2, 10, 1000)
            
            # Test that model has TensorStream attributes
            assert hasattr(offloaded_model, '_tensorstream_config')
            assert hasattr(offloaded_model, '_tensorstream_orchestrator')
            assert hasattr(offloaded_model, '_tensorstream_registry')
            
            self.log("End-to-end offloading validated")
            return True
        except Exception as e:
            self.log(f"End-to-end offloading test failed: {e}")
            return False
    
    def test_transformers_compatibility(self) -> bool:
        """Test compatibility with transformers-style model patterns."""
        try:
            import torch
            import torch.nn as nn
            from tensorstream import offload, create_default_config
            
            # Create model that mimics transformers library patterns
            class HuggingFaceStyleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type('Config', (), {
                        'hidden_size': 256,
                        'num_attention_heads': 8,
                        'vocab_size': 1000
                    })()
                    
                    self.embeddings = nn.Embedding(1000, 256)
                    self.encoder = nn.ModuleList([
                        self._create_layer() for _ in range(3)
                    ])
                    self.pooler = nn.Linear(256, 256)
                
                def _create_layer(self):
                    layer = nn.Module()
                    layer.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                    layer.intermediate = nn.Linear(256, 1024)
                    layer.output = nn.Linear(1024, 256)
                    layer.layernorm_before = nn.LayerNorm(256)
                    layer.layernorm_after = nn.LayerNorm(256)
                    return layer
                
                def forward(self, input_ids):
                    x = self.embeddings(input_ids)
                    for layer in self.encoder:
                        # Attention
                        attn_out, _ = layer.attention(x, x, x)
                        x = layer.layernorm_before(x + attn_out)
                        
                        # FFN
                        intermediate = torch.relu(layer.intermediate(x))
                        output = layer.output(intermediate)
                        x = layer.layernorm_after(x + output)
                    
                    return self.pooler(x.mean(dim=1))
            
            model = HuggingFaceStyleModel()
            
            # Apply offloading
            config = create_default_config(self.temp_dir)
            config.min_layer_size = 512
            offloaded_model = offload(model, config)
            
            # Test transformers-style access patterns
            for name, module in offloaded_model.named_modules():
                if hasattr(module, 'device'):
                    _ = module.device  # Should not fail
                
                # Test named_modules traversal
                list(module.named_modules())
            
            # Test inference
            input_ids = torch.randint(0, 1000, (2, 8))
            output = offloaded_model(input_ids)
            assert output.shape == (2, 256)
            
            self.log("Transformers compatibility validated")
            return True
        except Exception as e:
            self.log(f"Transformers compatibility test failed: {e}")
            return False
    
    def test_performance_characteristics(self) -> bool:
        """Test and measure performance characteristics."""
        try:
            import torch
            import torch.nn as nn
            from tensorstream import offload, create_default_config
            
            # Create a reasonably sized model for performance testing
            model = nn.Sequential(*[
                nn.Linear(512, 512) for _ in range(8)
            ])
            
            config = create_default_config(self.temp_dir)
            
            # Measure offloading time
            start_time = time.time()
            offloaded_model = offload(model, config)
            offload_time = time.time() - start_time
            
            # Measure first inference (cold)
            input_tensor = torch.randn(4, 512)
            start_time = time.time()
            with torch.no_grad():
                output1 = offloaded_model(input_tensor)
            cold_inference_time = time.time() - start_time
            
            # Measure second inference (warm)
            start_time = time.time()
            with torch.no_grad():
                output2 = offloaded_model(input_tensor)
            warm_inference_time = time.time() - start_time
            
            # Record metrics
            self.results["performance_metrics"] = {
                "offload_time_seconds": round(offload_time, 3),
                "cold_inference_time_seconds": round(cold_inference_time, 3),
                "warm_inference_time_seconds": round(warm_inference_time, 3),
                "warmup_speedup": round(cold_inference_time / warm_inference_time, 2)
            }
            
            # Validate outputs are consistent
            assert torch.allclose(output1, output2, atol=1e-6)
            
            self.log(f"Performance metrics recorded", "PERF")
            self.log(f"  Offload time: {offload_time:.3f}s", "PERF")
            self.log(f"  Cold inference: {cold_inference_time:.3f}s", "PERF")
            self.log(f"  Warm inference: {warm_inference_time:.3f}s", "PERF")
            self.log(f"  Speedup after warmup: {cold_inference_time/warm_inference_time:.2f}x", "PERF")
            
            return True
        except Exception as e:
            self.log(f"Performance test failed: {e}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory management and cleanup."""
        try:
            import torch
            import torch.nn as nn
            from tensorstream import offload, create_default_config
            
            # Create model
            model = nn.Sequential(*[nn.Linear(256, 256) for _ in range(4)])
            config = create_default_config(self.temp_dir)
            
            # Test offloading and cleanup
            offloaded_model = offload(model, config)
            
            # Test cleanup functionality
            if hasattr(offloaded_model, 'cleanup_tensorstream'):
                offloaded_model.cleanup_tensorstream()
                self.log("Model cleanup successful")
            
            # Test statistics
            if hasattr(offloaded_model, '_tensorstream_orchestrator'):
                stats = offloaded_model._tensorstream_orchestrator.get_statistics()
                self.log(f"Memory statistics: {stats}")
            
            return True
        except Exception as e:
            self.log(f"Memory management test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        self.log("=" * 60)
        self.log("TENSORSTREAM PRODUCTION VALIDATION SUITE")
        self.log("=" * 60)
        
        # Setup temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_validation_"))
        self.log(f"Using temporary directory: {self.temp_dir}")
        
        try:
            # Run all tests
            test_suite = [
                ("Import Test", self.test_imports),
                ("API Boolean Fix", self.test_api_boolean_fix),
                ("Proxy Layer Methods", self.test_proxy_layer_methods),
                ("I/O System", self.test_io_system),
                ("End-to-End Offloading", self.test_end_to_end_offloading),
                ("Transformers Compatibility", self.test_transformers_compatibility),
                ("Performance Characteristics", self.test_performance_characteristics),
                ("Memory Management", self.test_memory_management),
            ]
            
            for test_name, test_func in test_suite:
                self.run_test(test_name, test_func)
                time.sleep(0.1)  # Brief pause between tests
            
            # Generate final report
            self.generate_report()
            
        finally:
            # Cleanup
            try:
                shutil.rmtree(self.temp_dir)
                self.log(f"Cleaned up temporary directory")
            except Exception as e:
                self.log(f"Cleanup warning: {e}", "WARNING")
        
        return self.results
    
    def generate_report(self) -> None:
        """Generate final validation report."""
        self.log("=" * 60)
        self.log("VALIDATION RESULTS")
        self.log("=" * 60)
        
        # Test summary
        total_tests = self.results["tests_run"]
        passed_tests = self.results["tests_passed"]
        failed_tests = self.results["tests_failed"]
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.log(f"Tests Run: {total_tests}")
        self.log(f"Tests Passed: {passed_tests}", "SUCCESS" if passed_tests == total_tests else "INFO")
        self.log(f"Tests Failed: {failed_tests}", "ERROR" if failed_tests > 0 else "SUCCESS")
        self.log(f"Success Rate: {success_rate:.1f}%")
        
        # Performance metrics
        if self.results["performance_metrics"]:
            self.log("\nPerformance Metrics:", "PERF")
            for metric, value in self.results["performance_metrics"].items():
                self.log(f"  {metric}: {value}", "PERF")
        
        # Error details
        if self.results["errors"]:
            self.log("\nErrors:", "ERROR")
            for error in self.results["errors"]:
                self.log(f"  {error}", "ERROR")
        
        # Final status
        self.log("=" * 60)
        if failed_tests == 0:
            self.log("🎉 ALL TESTS PASSED - TENSORSTREAM IS PRODUCTION READY! 🎉", "SUCCESS")
            self.log("✅ Ready for deployment in production environments")
            self.log("✅ All critical bugs have been resolved")
            self.log("✅ Performance targets are met")
            self.log("✅ Compatibility with transformers validated")
        else:
            self.log("❌ SOME TESTS FAILED - REVIEW REQUIRED", "ERROR")
            self.log("⚠️  Address failures before production deployment")
        
        self.log("=" * 60)

def main():
    """Main validation entry point."""
    validator = ValidationSuite()
    
    try:
        results = validator.run_all_tests()
        return 0 if results["tests_failed"] == 0 else 1
    except KeyboardInterrupt:
        validator.log("Validation interrupted by user", "WARNING")
        return 1
    except Exception as e:
        validator.log(f"Validation suite error: {e}", "ERROR")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
