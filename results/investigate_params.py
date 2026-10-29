"""
investigate_params.py - Deep dive into SimpleCNN param discrepancy
"""
import sys, torch
sys.path.insert(0, '.')
from finetune_cifar10 import SimpleCNN

# Load checkpoint
ckpt = torch.load(r"D:\FYP\ShiftGuard\simplecnn_cifar10_binary_best.pth", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]

# Check what layers exist in the checkpoint
print("=== SimpleCNN checkpoint state_dict keys ===")
for k, v in sd.items():
    print(f"  {k}: {list(v.shape)} ({v.numel():,} params)")

print()

# Build model and compare
m = SimpleCNN(n_classes=2)
msd = m.state_dict()
print("=== SimpleCNN model state_dict keys ===")
for k, v in msd.items():
    print(f"  {k}: {list(v.shape)} ({v.numel():,} params)")

print()

# Check what's extra/different
sd_keys = set(sd.keys())
msd_keys = set(msd.keys())
print(f"Extra keys in checkpoint: {sd_keys - msd_keys}")
print(f"Missing keys in checkpoint: {msd_keys - sd_keys}")
