#!/usr/bin/env python3
"""Comprehensive test script that writes results to a file."""

import sys
import traceback
from pathlib import Path

def write_log(message):
    """Write message to both stdout and a log file."""
    print(message)
    with open('/home/lukas/MLWork/tensorstream/test_results.log', 'a') as f:
        f.write(message + '\n')

def test_imports():
    """Test basic imports."""
    write_log("=== Testing Imports ===")
    try:
        import torch
        write_log(f"✓ PyTorch {torch.__version__} imported")
        
        import torch.nn as nn
        write_log("✓ torch.nn imported")
        
        import tensorstream
        write_log("✓ TensorStream imported")
        
        from tensorstream.proxy import TensorStreamProxyLayer
        write_log("✓ TensorStreamProxyLayer imported")
        
        from tensorstream.orchestrator import OrchestrationEngine
        write_log("✓ OrchestrationEngine imported")
        
        from tensorstream.config import Config
        write_log("✓ Config imported")
        
        return True
    except Exception as e:
        write_log(f"✗ Import failed: {e}")
        write_log(traceback.format_exc())
        return False

def test_proxy_creation():
    """Test proxy layer creation."""
    write_log("\n=== Testing Proxy Creation ===")
    try:
        import torch.nn as nn
        import tempfile
        from tensorstream.proxy import TensorStreamProxyLayer
        from tensorstream.orchestrator import OrchestrationEngine
        from tensorstream.config import Config
        
        # Create a simple layer
        original_layer = nn.Linear(10, 5)
        write_log("✓ Original layer created")
        
        # Create minimal orchestrator for testing
        temp_dir = Path(tempfile.mkdtemp())
        config = Config(storage_path=temp_dir, debug_mode=False)
        orchestrator = OrchestrationEngine(config)
        write_log("✓ Orchestrator created")
        
        # Create proxy layer
        proxy = TensorStreamProxyLayer("test_layer", original_layer, orchestrator)
        write_log("✓ Proxy layer created successfully")
        
        return proxy
    except Exception as e:
        write_log(f"✗ Proxy creation failed: {e}")
        write_log(traceback.format_exc())
        return None

def test_proxy_methods(proxy):
    """Test proxy layer methods."""
    write_log("\n=== Testing Proxy Methods ===")
    if proxy is None:
        write_log("✗ Cannot test methods - proxy is None")
        return False
    
    try:
        # Test device property
        device = proxy.device
        write_log(f"✓ Device property works: {device}")
        
        # Test named_modules method
        modules_list = list(proxy.named_modules())
        write_log(f"✓ named_modules() works: found {len(modules_list)} modules")
        
        # Test with all parameters
        modules_with_memo = list(proxy.named_modules(memo=set(), prefix='test.', remove_duplicate=True))
        write_log(f"✓ named_modules() with all params works: found {len(modules_with_memo)} modules")
        
        # Test children method
        children_list = list(proxy.children())
        write_log(f"✓ children() works: found {len(children_list)} children")
        
        # Test named_children method
        named_children_list = list(proxy.named_children())
        write_log(f"✓ named_children() works: found {len(named_children_list)} named children")
        
        return True
    except Exception as e:
        write_log(f"✗ Method testing failed: {e}")
        write_log(traceback.format_exc())
        return False

def main():
    """Main test function."""
    # Clear the log file
    with open('/home/lukas/MLWork/tensorstream/test_results.log', 'w') as f:
        f.write("TensorStream Comprehensive Test Results\n")
        f.write("=" * 50 + "\n")
    
    write_log("Starting comprehensive TensorStream tests...")
    
    # Test imports
    if not test_imports():
        write_log("❌ Import tests failed - aborting")
        return False
    
    # Test proxy creation
    proxy = test_proxy_creation()
    if proxy is None:
        write_log("❌ Proxy creation failed - aborting")
        return False
    
    # Test proxy methods
    if not test_proxy_methods(proxy):
        write_log("❌ Proxy method tests failed")
        return False
    
    write_log("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        write_log(f"Fatal error: {e}")
        write_log(traceback.format_exc())
        sys.exit(1)
