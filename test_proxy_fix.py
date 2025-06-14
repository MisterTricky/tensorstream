#!/usr/bin/env python3
"""
Simple test to verify the proxy layer fix.
"""

import torch
import torch.nn as nn

# Import our proxy layer
from tensorstream.proxy import TensorStreamProxyLayer
from tensorstream.orchestrator import OrchestrationEngine
from tensorstream.config import Config
import tempfile
from pathlib import Path

def test_proxy_methods():
    """Test that proxy layer has correct method signatures."""
    print("Testing proxy layer method signatures...")
    
    # Create a simple layer
    original_layer = nn.Linear(10, 5)
    
    # Create minimal orchestrator for testing
    temp_dir = Path(tempfile.mkdtemp())
    config = Config(storage_path=temp_dir, debug_mode=True)
    orchestrator = OrchestrationEngine(config)
    
    # Create proxy layer
    proxy = TensorStreamProxyLayer("test_layer", original_layer, orchestrator)
    
    # Test the problematic method
    try:
        # This should work without error now
        modules_list = list(proxy.named_modules())
        print(f"✓ named_modules() works: found {len(modules_list)} modules")
        
        # Test with all parameters
        modules_with_memo = list(proxy.named_modules(memo=set(), prefix='test.', remove_duplicate=True))
        print(f"✓ named_modules() with all params works: found {len(modules_with_memo)} modules")
        
        # Test device property
        device = proxy.device
        print(f"✓ device property works: {device}")
        
        # Test children method
        children_list = list(proxy.children())
        print(f"✓ children() works: found {len(children_list)} children")
        
        print("✅ All proxy layer methods working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Proxy layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = test_proxy_methods()
    if success:
        print("\n🎉 Proxy layer fix successful!")
    else:
        print("\n💥 Proxy layer still has issues.")
    exit(0 if success else 1)
