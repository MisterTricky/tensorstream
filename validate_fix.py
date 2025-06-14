#!/usr/bin/env python3
"""
Quick validation script to test the TensorStream API fix.
"""

import torch
import torch.nn as nn
import tempfile
from pathlib import Path

# Test the fix by importing and testing the problematic function
import tensorstream
from tensorstream.api import _is_offloadable_layer
from tensorstream.config import Config, BackendType

def test_is_offloadable_layer():
    """Test the _is_offloadable_layer function that was causing issues."""
    print("🧪 Testing _is_offloadable_layer function...")
    
    # Create a simple layer
    layer = nn.Linear(100, 50)
    
    # This should not crash with "Boolean value of Tensor with more than one value is ambiguous"
    result = _is_offloadable_layer(layer)
    print(f"   ✓ Linear layer offloadable: {result}")
    
    # Test with a layer that has no parameters
    layer_no_params = nn.ReLU()
    result_no_params = _is_offloadable_layer(layer_no_params)
    print(f"   ✓ ReLU layer offloadable: {result_no_params}")
    
    # Test with a module that has child modules
    class ParentModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.child1 = nn.Linear(10, 10)
            self.child2 = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.child2(self.child1(x))
    
    parent = ParentModule()
    result_parent = _is_offloadable_layer(parent)
    print(f"   ✓ Parent module offloadable: {result_parent}")
    
    print("✅ _is_offloadable_layer function test passed!")
    return True

def test_basic_config():
    """Test basic configuration creation."""
    print("🧪 Testing basic configuration...")
    
    temp_dir = Path(tempfile.mkdtemp(prefix="tensorstream_test_"))
    
    try:
        config = Config(
            storage_path=temp_dir,
            vram_budget_gb=1.0,
            backend=BackendType.MMAP,
            debug_mode=True
        )
        
        print(f"   ✓ Config created: {config.storage_path}")
        print("✅ Basic configuration test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Config creation failed: {e}")
        return False
    
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    """Run validation tests."""
    print("🔧 TensorStream API Fix Validation")
    print("=" * 40)
    
    try:
        # Test the function that was causing issues
        success1 = test_is_offloadable_layer()
        
        # Test basic configuration
        success2 = test_basic_config()
        
        if success1 and success2:
            print("\n🎉 All validation tests passed! The API fix is working.")
            return True
        else:
            print("\n❌ Some validation tests failed.")
            return False
            
    except Exception as e:
        print(f"\n💥 Validation crashed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
