#!/usr/bin/env python3
"""
Basic performance test for TensorStream library.

This script tests the core functionality without requiring GPU or
external model dependencies.
"""

import time
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

import tensorstream
from tensorstream.config import Config, BackendType


class SimpleTestModel(nn.Module):
    """Simple model for testing TensorStream functionality."""
    
    def __init__(self, input_dim=512, hidden_dim=1024, num_layers=3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, input_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


def test_basic_functionality():
    """Test basic TensorStream functionality."""
    print("🧪 Testing Basic TensorStream Functionality")
    print("=" * 50)
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_test_"))
    
    try:
        # Create test model
        print("📦 Creating test model...")
        model = SimpleTestModel(input_dim=256, hidden_dim=512, num_layers=4)
        
        # Create test input
        test_input = torch.randn(2, 256)  # batch_size=2, input_dim=256
        
        # Get baseline output
        print("🎯 Getting baseline output...")
        with torch.no_grad():
            baseline_output = model(test_input)
        
        # Create TensorStream config
        print("⚙️  Creating TensorStream configuration...")
        config = Config(
            storage_path=temp_dir,
            vram_budget_gb=0.5,  # Very conservative
            backend=BackendType.MMAP,  # Use fallback backend
            debug_mode=True
        )
        
        # Apply TensorStream
        print("🚀 Applying TensorStream...")
        offloaded_model = tensorstream.offload(model, config)
        
        # Test inference
        print("🔬 Testing inference...")
        start_time = time.time()
        
        with torch.no_grad():
            tensorstream_output = offloaded_model(test_input)
        
        inference_time = time.time() - start_time
        
        # Verify outputs match
        print("✅ Verifying output consistency...")
        if torch.allclose(baseline_output, tensorstream_output, atol=1e-5):
            print("   ✓ Outputs match!")
        else:
            print("   ❌ Outputs don't match!")
            print(f"   Max difference: {torch.max(torch.abs(baseline_output - tensorstream_output))}")
            return False
        
        # Print performance stats
        print(f"📊 Performance Results:")
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test cleanup
        print("🧹 Testing cleanup...")
        offloaded_model.cleanup_tensorstream()
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_file_operations():
    """Test file I/O operations."""
    print("\n📁 Testing File I/O Operations")
    print("=" * 50)
    
    temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_io_test_"))
    
    try:
        # Test basic tensor save/load
        print("💾 Testing tensor save/load...")
        
        # Create test tensors of various types
        test_tensors = {
            "float32": torch.randn(100, 50, dtype=torch.float32),
            "float16": torch.randn(100, 50, dtype=torch.float16),
            "int32": torch.randint(0, 100, (50, 25), dtype=torch.int32),
            "bool": torch.randint(0, 2, (20, 10), dtype=torch.bool),
        }
        
        for dtype_name, tensor in test_tensors.items():
            print(f"   Testing {dtype_name}...")
            
            # Save tensor
            file_path = temp_dir / f"test_{dtype_name}.ts"
            tensorstream.io.save_to_ts(tensor, file_path, compress=True)
            
            # Load tensor
            loaded_tensor = tensorstream.io.load_from_ts(file_path)
            
            # Verify
            if torch.equal(tensor, loaded_tensor):
                print(f"   ✓ {dtype_name} tensor save/load successful")
            else:
                print(f"   ❌ {dtype_name} tensor save/load failed")
                return False
        
        # Test metadata
        print("📝 Testing metadata...")
        metadata = {"test": "value", "number": 42}
        tensorstream.io.save_to_ts(
            test_tensors["float32"], 
            temp_dir / "test_metadata.ts", 
            metadata=metadata
        )
        
        file_info = tensorstream.io.get_ts_file_info(temp_dir / "test_metadata.ts")
        if file_info.get("metadata") == metadata:
            print("   ✓ Metadata save/load successful")
        else:
            print("   ❌ Metadata save/load failed")
            return False
        
        print("✅ All file operations passed!")
        return True
        
    except Exception as e:
        print(f"❌ File operations failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_configuration():
    """Test configuration system."""
    print("\n⚙️  Testing Configuration System")
    print("=" * 50)
    
    try:
        # Test default config creation
        print("📋 Testing default configuration...")
        temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_config_test_"))
        
        config = tensorstream.config.create_default_config(temp_dir)
        
        # Verify config properties
        assert config.storage_path == temp_dir
        assert config.vram_budget_gb > 0
        assert config.backend in [BackendType.MMAP, BackendType.CUDA, BackendType.GPUDIRECT]
        
        print("   ✓ Default configuration created successfully")
        
        # Test custom config
        print("🔧 Testing custom configuration...")
        custom_config = Config(
            storage_path=temp_dir,
            vram_budget_gb=2.0,
            backend=BackendType.MMAP,
            compression_enabled=True,
            compression_level=5,
            debug_mode=True
        )
        
        # Verify custom properties
        assert custom_config.storage_path == temp_dir
        assert custom_config.vram_budget_gb == 2.0
        assert custom_config.backend == BackendType.MMAP
        assert custom_config.compression_enabled is True
        assert custom_config.compression_level == 5
        
        print("   ✓ Custom configuration created successfully")
        
        # Test validation
        print("🔍 Testing configuration validation...")
        try:
            invalid_config = Config(
                storage_path="/nonexistent/path/that/cannot/be/created",
                vram_budget_gb=-1.0  # Invalid
            )
            print("   ❌ Configuration validation failed - invalid config was accepted")
            return False
        except tensorstream.ConfigurationError:
            print("   ✓ Configuration validation working correctly")
        
        print("✅ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Run all performance tests."""
    print("🎯 TensorStream Performance Test Suite")
    print("=" * 60)
    
    # Print system info
    print(f"🖥️  System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   TensorStream version: {tensorstream.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA backend available: {tensorstream.HAS_CUDA}")
    print(f"   GDS backend available: {tensorstream.HAS_GDS}")
    
    # Run tests
    tests = [
        ("Configuration System", test_configuration),
        ("File I/O Operations", test_file_operations),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results.append({
                'name': test_name,
                'success': success,
                'duration': duration
            })
            
            if success:
                print(f"✅ {test_name} completed successfully in {duration:.2f}s")
            else:
                print(f"❌ {test_name} failed after {duration:.2f}s")
                
        except Exception as e:
            print(f"💥 {test_name} crashed with error: {e}")
            results.append({
                'name': test_name,
                'success': False,
                'duration': 0
            })
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("📋 Test Summary")
    print('=' * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    total_time = sum(r['duration'] for r in results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {status:8} {result['name']:30} ({result['duration']:.2f}s)")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    print(f"⏱️  Total time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! TensorStream is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
