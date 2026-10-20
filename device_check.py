"""
Hardware Acceleration Diagnostic Script for Mac Mini (Apple Silicon).
Checks Metal Performance Shaders (MPS) for PyTorch and Metal/CPU devices for JAX.
"""
import sys
import platform

print("=" * 60)
print(f"Platform: {platform.system()} ({platform.machine()})")
print(f"Python:   {sys.version.split()[0]}")
print("=" * 60)

# 1. PyTorch Check
print("\n[1] Checking PyTorch...")
try:
    import torch
    print(f"  PyTorch Version: {torch.__version__}")
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    print(f"  MPS (Metal Performance Shaders) Built:     {mps_built}")
    print(f"  MPS Available for Acceleration:            {mps_available}")
    if mps_available:
        x = torch.ones(5, device="mps")
        print(f"  Successfully allocated tensor on GPU:     {x.device}")
    else:
        print("  Running on CPU.")
except ImportError:
    print("  PyTorch is not installed in this environment yet.")

# 2. JAX Check
print("\n[2] Checking JAX...")
try:
    import jax
    print(f"  JAX Version: {jax.__version__}")
    devices = jax.devices()
    default_backend = jax.default_backend()
    print(f"  Default Backend: {default_backend}")
    print(f"  Available Devices: {devices}")
    
    # Quick test computation
    import jax.numpy as jnp
    y = jnp.ones((5,))
    print(f"  Successfully created JAX array:            device={y.devices()}")
except ImportError:
    print("  JAX is not installed in this environment yet.")

print("\n" + "=" * 60)
