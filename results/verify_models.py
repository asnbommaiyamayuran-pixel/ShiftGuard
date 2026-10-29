"""
verify_models.py - Verify SimpleCNN architecture and parameter counts
"""
import sys, torch
sys.path.insert(0, '.')
from finetune_cifar10 import SimpleCNN

# Verify SimpleCNN architecture
m = SimpleCNN(n_classes=2)
p = sum(x.numel() for x in m.parameters())
print(f"SimpleCNN actual params (from class definition): {p:,}")

# Load checkpoint
ckpt = torch.load(r"D:\FYP\ShiftGuard\simplecnn_cifar10_binary_best.pth", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
p_ckpt = sum(x.numel() for x in sd.values())
print(f"SimpleCNN params (from checkpoint state_dict): {p_ckpt:,}")
print(f"Val acc in checkpoint: {ckpt.get('val_acc')}")
print(f"Checkpoint keys: {list(ckpt.keys())}")
