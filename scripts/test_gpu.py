#!/usr/bin/env python3
"""GPU Detection Test for AMD RX 6600 XT with PyTorch ROCm"""

import sys

print("=" * 60)
print("🔥 PyTorch + ROCm GPU Detection Test")
print("=" * 60)

# Test PyTorch import
try:
    import torch
    print(f"✅ PyTorch Version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    sys.exit(1)

# Test CUDA/ROCm availability
cuda_available = torch.cuda.is_available()
print(f"GPU Available: {'✅ YES' if cuda_available else '❌ NO'}")

if cuda_available:
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    # Test simple tensor operation on GPU
    print("\nTesting GPU tensor operation...")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU tensor operation successful!")
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
else:
    print("\n⚠️  GPU not detected. Check:")
    print("  1. HSA_OVERRIDE_GFX_VERSION=10.3.0 is set")
    print("  2. User is in 'video' and 'render' groups")
    print("  3. System has been rebooted after group changes")

print("=" * 60)
