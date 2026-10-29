"""
verify_checkpoints.py
Loads and inspects all three checkpoints for verification purposes.
Does NOT modify any files.
"""

import sys
import torch

sys.path.insert(0, ".")

CHECKPOINTS = [
    ("shiftguard_cifar10_exp1_lr1e4.pth",   r"D:\FYP\ShiftGuard\shiftguard_cifar10_exp1_lr1e4.pth"),
    ("shiftguard_cifar10_binary_best.pth",   r"D:\FYP\ShiftGuard\shiftguard_cifar10_binary_best.pth"),
    ("simplecnn_cifar10_binary_best.pth",    r"D:\FYP\ShiftGuard\simplecnn_cifar10_binary_best.pth"),
]

for name, path in CHECKPOINTS:
    print(f"=== Checkpoint: {name} ===")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  Keys in checkpoint: {list(ckpt.keys())}")
        print(f"  Epoch  : {ckpt.get('epoch', 'N/A')}")
        print(f"  Stage  : {ckpt.get('stage', 'N/A')}")
        print(f"  Val acc: {ckpt.get('val_acc', 'N/A')}")
        sd = ckpt.get("model_state_dict", ckpt)
        n_params = sum(p.numel() for p in sd.values())
        print(f"  Params : {n_params:,}")
        if "training_config" in ckpt:
            print(f"  Config : {ckpt['training_config']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
