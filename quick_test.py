#!/usr/bin/env python3
"""Quick test for proxy layer."""

import torch
import torch.nn as nn
from tensorstream.proxy import TensorStreamProxyLayer
from tensorstream.orchestrator import OrchestrationEngine
from tensorstream.config import Config
import tempfile
from pathlib import Path

# Create a simple layer
print("Creating test layer...")
original_layer = nn.Linear(10, 5)

# Create minimal orchestrator for testing
print("Creating orchestrator...")
temp_dir = Path(tempfile.mkdtemp())
config = Config(storage_path=temp_dir, debug_mode=False)  # Turn off debug to reduce output
orchestrator = OrchestrationEngine(config)

# Create proxy layer
print("Creating proxy layer...")
try:
    proxy = TensorStreamProxyLayer("test_layer", original_layer, orchestrator)
    print("✓ Proxy layer created successfully!")
    
    # Test device property
    device = proxy.device
    print(f"✓ Device property works: {device}")
    
    # Test named_modules method
    modules_list = list(proxy.named_modules())
    print(f"✓ named_modules() works: found {len(modules_list)} modules")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
