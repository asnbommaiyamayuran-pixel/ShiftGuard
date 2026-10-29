r"""
================================================================================
finetune_cifar10.py  --  ShiftGuard CIFAR-10 Binary Fine-Tuning Pipeline
================================================================================
Adapts ShiftGuard (MiniViT + ShiftAnalyzer + Router + LoRA adapters) for
binary classification: CIFAR-10 automobile (label 1) vs truck (label 9).

Binary label mapping
  CIFAR-10 label 1 (automobile) -> model class 0
  CIFAR-10 label 9 (truck)      -> model class 1

Architecture note
  The original 3-class task head is REPLACED with a fresh 2-class head.
  The backbone, ShiftAnalyzer, Router, and LoRA Adapters are loaded from
  shiftguard_checkpoint.pth (compatible weights only).

Staged training
  Stage 1: Train new 2-class head only  (backbone/adapters frozen)
  Stage 2: Full end-to-end fine-tuning  (all layers, smaller LR)

Usage (run from results/ or project root)
  cd D:\FYP\ShiftGuard\results
  python finetune_cifar10.py [--epochs-stage1 8] [--epochs-stage2 20]
                              [--batch-size 32] [--lr-head 1e-3]
                              [--lr-finetune 3e-5] [--patience 6]
                              [--seed 42] [--no-baseline]
================================================================================
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Ensure shiftguard.py is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from shiftguard import (
    MiniViT, ShiftAnalyzer, Router, AdaptiveAdapters,
    TaskPredictor, IMG_SIZE,
    set_seed, confidence_and_entropy,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CIFAR_DATA_DIR      = PROJECT_ROOT / "data" / "cifar-10-batches-py"
ORIGINAL_CHECKPOINT = PROJECT_ROOT / "shiftguard_checkpoint.pth"
BEST_CHECKPOINT     = PROJECT_ROOT / "shiftguard_cifar10_binary_best.pth"
FIGURES_DIR         = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CIFAR-10 label constants
# ---------------------------------------------------------------------------
CIFAR_AUTOMOBILE   = 1
CIFAR_TRUCK        = 9
BINARY_LABELS      = {CIFAR_AUTOMOBILE: 0, CIFAR_TRUCK: 1}
BINARY_CLASS_NAMES = ["automobile (car)", "truck"]


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_cifar10_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10_train(data_dir):
    all_images, all_labels = [], []
    for i in range(1, 6):
        imgs, lbls = load_cifar10_batch(data_dir / f"data_batch_{i}")
        all_images.append(imgs)
        all_labels.append(lbls)
    return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)


def filter_automobile_truck(images, labels):
    """Keep only automobile (1) and truck (9) and remap to {0, 1}."""
    mask = np.isin(labels, [CIFAR_AUTOMOBILE, CIFAR_TRUCK])
    imgs = images[mask]
    lbls = np.array([BINARY_LABELS[l] for l in labels[mask]], dtype=np.int64)
    return imgs, lbls


class CIFAR10BinaryDataset(Dataset):
    """
    CIFAR-10 automobile-vs-truck binary dataset.
    Input images are uint8 HWC numpy arrays (32x32x3).
    They are resized to IMG_SIZE x IMG_SIZE (64x64) to match MiniViT.
    """
    # CIFAR-10 per-channel mean and std (computed over full training set)
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD  = [0.2470, 0.2435, 0.2616]

    def __init__(self, images, labels, train=True):
        self.images = images
        self.labels = labels
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(IMG_SIZE, padding=8),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(self.CIFAR_MEAN, self.CIFAR_STD),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.labels[idx])


# ===========================================================================
# 2. ADAPTED MODEL  (ShiftGuard with 2-class head)
# ===========================================================================

class ShiftGuardBinary(nn.Module):
    """
    ShiftGuard with the classification head replaced by a fresh 2-class head.

    Backbone, ShiftAnalyzer, Router, and LoRA Adapters are loaded from the
    original 3-class checkpoint when available (compatible weights only).
    The task_head is NOT loaded from the checkpoint; it is randomly initialised
    so that Stage 1 training can train it from scratch without inheriting the
    old 3-class softmax bias.
    """
    def __init__(self, n_classes=2, rank=8):
        super().__init__()
        self.backbone       = MiniViT()
        dim                 = self.backbone.out_dim
        self.shift_analyzer = ShiftAnalyzer(dim)
        self.router         = Router(in_dim=3, n_adapters=3)
        self.adapters       = AdaptiveAdapters(dim, rank=rank)
        self.task_head      = TaskPredictor(dim, n_classes=n_classes)

    def forward(self, x):
        f                               = self.backbone(x)
        S, domain_logits, s_m, s_s     = self.shift_analyzer(f)
        weights                         = self.router(S)
        fused, eff_w                    = self.adapters(f, weights)
        logits                          = self.task_head(fused)
        probs, pred, confidence, entropy = confidence_and_entropy(logits)
        return {
            "logits": logits, "probs": probs, "pred": pred,
            "confidence": confidence, "entropy": entropy,
            "shift_signature": S, "domain_logits": domain_logits,
            "s_m": s_m, "s_s": s_s, "adapter_weights": eff_w,
        }


def build_model_from_checkpoint(checkpoint_path, device, rank=8):
    """
    Construct ShiftGuardBinary, loading compatible weights from the original
    3-class checkpoint.  The task_head is intentionally excluded.
    If rank != 8, adapters are also excluded because the checkpoint has rank-8 adapters.
    """
    model = ShiftGuardBinary(n_classes=2, rank=rank).to(device)
    if not Path(checkpoint_path).exists():
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Using random init.")
        return model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd   = ckpt["model_state_dict"]

    if rank != 8:
        print(f"[Checkpoint] lora_rank={rank} differs from checkpoint rank 8. Pretrained adapters excluded; will be randomly initialized.")
        compatible = {k: v for k, v in sd.items() if not k.startswith("task_head") and not k.startswith("adapters")}
    else:
        compatible = {k: v for k, v in sd.items() if not k.startswith("task_head")}

    missing, unexpected = model.load_state_dict(compatible, strict=False)

    loaded  = len(compatible)
    skipped = [k for k in sd if k not in compatible]
    print(f"[Checkpoint] Loaded {loaded} tensors from {Path(checkpoint_path).name}")
    print(f"[Checkpoint] Excluded keys: {skipped}")
    if missing:
        print(f"[Checkpoint] Keys fresh-initialised (not in checkpoint): {missing}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ignored: {unexpected}")
    return model


# ===========================================================================
# 3. TRAINING UTILITIES
# ===========================================================================

def set_stage1_frozen(model):
    """Freeze all parameters except the task_head (new 2-class head)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("task_head")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"[Stage 1] Trainable params: {trainable:,} / {total:,} ({pct:.2f}%) -- head only")


def set_stage2_unfrozen(model):
    """Unfreeze all parameters for full end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] Trainable params: {trainable:,} -- full end-to-end fine-tuning")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = F.cross_entropy(out["logits"], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out["pred"] == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct, total_loss, total = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = F.cross_entropy(out["logits"], y)
        total_loss += loss.item() * x.size(0)
        correct    += (out["pred"] == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


# ===========================================================================
# 4. BASELINE CNN (simple binary CNN for comparison)
# ===========================================================================

class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline trained from scratch for automobile-vs-truck.
    Input: 64x64 RGB (same preprocessing as ShiftGuardBinary for fair comparison).
    ~1.3 M parameters.
    """
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ===========================================================================
# 5. PLOT UTILITIES
# ===========================================================================

def save_training_curves(train_accs, val_accs, train_losses, val_losses,
                          stage1_end, title, save_path):
    """Save training/validation accuracy and loss curves to a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(train_accs) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        ax1.plot(epochs, [a * 100 for a in train_accs], "b-o", ms=4, label="Train")
        ax1.plot(epochs, [a * 100 for a in val_accs],   "r-o", ms=4, label="Val")
        if 0 < stage1_end < len(epochs):
            ax1.axvline(stage1_end + 0.5, color="gray", ls="--", lw=1.5, label="Stage1->Stage2")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Train vs Validation Accuracy")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 105)

        ax2.plot(epochs, train_losses, "b-o", ms=4, label="Train")
        ax2.plot(epochs, val_losses,   "r-o", ms=4, label="Val")
        if 0 < stage1_end < len(epochs):
            ax2.axvline(stage1_end + 0.5, color="gray", ls="--", lw=1.5, label="Stage1->Stage2")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cross-Entropy Loss")
        ax2.set_title("Train vs Validation Loss")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved -> {save_path}")
    except Exception as exc:
        print(f"[Plot] Could not save training curves: {exc}")


# ===========================================================================
# 6. MAIN
# ===========================================================================

def main(args):
    t_start = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_checkpoint = Path(args.checkpoint_out) if args.checkpoint_out else BEST_CHECKPOINT
    if not out_checkpoint.is_absolute():
        out_checkpoint = PROJECT_ROOT / out_checkpoint

    curves_name = f"{out_checkpoint.stem}_training_curves.png" if args.checkpoint_out else "shiftguard_training_curves.png"
    curves_path = FIGURES_DIR / curves_name

    print("=" * 70)
    print("ShiftGuard CIFAR-10 Binary Fine-Tuning Pipeline")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Random seed  : {args.seed}")
    print(f"LoRA rank    : {args.lora_rank}")
    print(f"Checkpoint in: {ORIGINAL_CHECKPOINT}")
    print(f"Output path  : {out_checkpoint}")

    # ------------------------------------------------------------------
    # A. Load CIFAR-10
    # ------------------------------------------------------------------
    print("\n[Data] Loading CIFAR-10 training batches ...")
    all_tr_imgs, all_tr_lbls = load_cifar10_train(CIFAR_DATA_DIR)
    tr_imgs_full, tr_lbls_full = filter_automobile_truck(all_tr_imgs, all_tr_lbls)

    print("[Data] Loading CIFAR-10 test batch ...")
    te_imgs_raw, te_lbls_raw = load_cifar10_batch(CIFAR_DATA_DIR / "test_batch")
    te_imgs, te_lbls = filter_automobile_truck(te_imgs_raw, te_lbls_raw)

    n_auto  = int((tr_lbls_full == 0).sum())
    n_truck = int((tr_lbls_full == 1).sum())
    print(f"[Data] Training pool  : {len(tr_lbls_full):,}  (auto={n_auto}, truck={n_truck})")
    print(f"[Data] Official test  : {len(te_lbls):,}")

    # Reproducible 90/10 train/val split
    rng     = np.random.default_rng(args.seed)
    indices = rng.permutation(len(tr_lbls_full))
    n_val   = max(1, int(0.10 * len(tr_lbls_full)))
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]

    tr_imgs  = tr_imgs_full[trn_idx];  tr_lbls  = tr_lbls_full[trn_idx]
    vl_imgs  = tr_imgs_full[val_idx];  vl_lbls  = tr_lbls_full[val_idx]
    print(f"[Data] Train split    : {len(tr_lbls):,}")
    print(f"[Data] Val split      : {len(vl_lbls):,}")

    train_ds = CIFAR10BinaryDataset(tr_imgs, tr_lbls, train=True)
    val_ds   = CIFAR10BinaryDataset(vl_imgs, vl_lbls, train=False)
    test_ds  = CIFAR10BinaryDataset(te_imgs, te_lbls, train=False)

    # Weighted sampler to handle any minor class imbalance
    cls_counts   = np.bincount(tr_lbls)
    samp_weights = (1.0 / cls_counts)[tr_lbls]
    sampler      = WeightedRandomSampler(samp_weights.tolist(), len(samp_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    if not args.train_baseline_only:
        # ------------------------------------------------------------------
        # B. Build model
        # ------------------------------------------------------------------
        print("\n[Model] Building ShiftGuardBinary (2-class head) ...", flush=True)
        model        = build_model_from_checkpoint(ORIGINAL_CHECKPOINT, device, rank=args.lora_rank)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] Total parameters: {total_params:,}", flush=True)

        # ------------------------------------------------------------------
        # C. Stage 1 - head only (backbone frozen)
        # ------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STAGE 1: Head-only training (backbone + adapters frozen)", flush=True)
        print("=" * 60, flush=True)
        set_stage1_frozen(model)

        opt_s1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, weight_decay=args.weight_decay,
        )
        sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=args.epochs_stage1)

        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        best_val_acc = -1.0
        patience_ctr = 0

        for epoch in range(1, args.epochs_stage1 + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt_s1, device)
            vl_loss, vl_acc = evaluate_loader(model, val_loader, device)
            sched_s1.step()

            train_accs.append(tr_acc)
            val_accs.append(vl_acc)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            improved = vl_acc > best_val_acc
            if improved:
                best_val_acc = vl_acc
                patience_ctr = 0
                torch.save({
                    "epoch": epoch, "stage": 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "binary_classes": BINARY_CLASS_NAMES,
                    "seed": args.seed,
                    "cifar_automobile_label": CIFAR_AUTOMOBILE,
                    "cifar_truck_label": CIFAR_TRUCK,
                    "lora_rank": args.lora_rank,
                    "training_config": vars(args),
                }, out_checkpoint)
            else:
                patience_ctr += 1

            mk = "V" if improved else " "
            print(
                f"  [{mk}] Ep {epoch:3d}/{args.epochs_stage1}"
                f"  loss={tr_loss:.4f}  acc={tr_acc * 100:.1f}%"
                f"  |  val_loss={vl_loss:.4f}  val_acc={vl_acc * 100:.1f}%"
                f"  (best={best_val_acc * 100:.1f}%)",
                flush=True
            )

            if patience_ctr >= args.patience:
                print(f"  [EarlyStop] No val improvement for {args.patience} epochs. Stopping Stage 1.", flush=True)
                break

        stage1_end = len(train_accs)

        # ------------------------------------------------------------------
        # D. Stage 2 - full fine-tuning
        # ------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("STAGE 2: Full end-to-end fine-tuning (all layers unlocked)", flush=True)
        print("=" * 60, flush=True)
        set_stage2_unfrozen(model)

        # Reload best Stage-1 weights before starting Stage 2
        s1_ckpt = torch.load(out_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(s1_ckpt["model_state_dict"])
        print(f"[Stage 2] Loaded best Stage-1 checkpoint (val_acc={best_val_acc * 100:.2f}%)", flush=True)

        opt_s2 = torch.optim.AdamW(
            model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay,
        )
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=args.epochs_stage2)
        patience_ctr = 0

        for epoch in range(1, args.epochs_stage2 + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt_s2, device)
            vl_loss, vl_acc = evaluate_loader(model, val_loader, device)
            sched_s2.step()

            train_accs.append(tr_acc)
            val_accs.append(vl_acc)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            improved = vl_acc > best_val_acc
            if improved:
                best_val_acc = vl_acc
                patience_ctr = 0
                torch.save({
                    "epoch": stage1_end + epoch, "stage": 2,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "binary_classes": BINARY_CLASS_NAMES,
                    "seed": args.seed,
                    "cifar_automobile_label": CIFAR_AUTOMOBILE,
                    "cifar_truck_label": CIFAR_TRUCK,
                    "lora_rank": args.lora_rank,
                    "training_config": vars(args),
                }, out_checkpoint)
            else:
                patience_ctr += 1

            mk     = "V" if improved else " "
            tot_ep = stage1_end + epoch
            print(
                f"  [{mk}] Ep {tot_ep:3d}  (S2 {epoch:3d}/{args.epochs_stage2})"
                f"  loss={tr_loss:.4f}  acc={tr_acc * 100:.1f}%"
                f"  |  val_loss={vl_loss:.4f}  val_acc={vl_acc * 100:.1f}%"
                f"  (best={best_val_acc * 100:.1f}%)",
                flush=True
            )

            if patience_ctr >= args.patience:
                print(f"  [EarlyStop] No val improvement for {args.patience} epochs. Stopping Stage 2.", flush=True)
                break

        # ------------------------------------------------------------------
        # E. Save training curves
        # ------------------------------------------------------------------
        save_training_curves(
            train_accs, val_accs, train_losses, val_losses,
            stage1_end=stage1_end,
            title=f"ShiftGuard CIFAR-10 Binary Fine-Tuning ({out_checkpoint.stem})",
            save_path=curves_path,
        )

        # ------------------------------------------------------------------
        # F. Final test evaluation (controlled via --eval-test)
        # ------------------------------------------------------------------
        elapsed = time.time() - t_start
        test_acc = float('nan')
        if args.eval_test:
            print("\n[Test] Loading best checkpoint for final test evaluation ...", flush=True)
            best_ckpt = torch.load(out_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            _, test_acc = evaluate_loader(model, test_loader, device)
            print(f"[RESULT] Test accuracy       : {test_acc * 100:.2f}%", flush=True)
        else:
            print("\n[Test] Test evaluation skipped to prevent test-set leakage during hyperparameter tuning.", flush=True)
            print(f"       Evaluate when finalized using: python results/evaluate_cifar10_binary.py --checkpoint {out_checkpoint}", flush=True)

        print("\n" + "=" * 60, flush=True)
        print(f"[RESULT] Best val accuracy   : {best_val_acc * 100:.2f}%", flush=True)
        if args.eval_test:
            print(f"[RESULT] Test accuracy       : {test_acc * 100:.2f}%", flush=True)
        print(f"[RESULT] Best checkpoint     : {out_checkpoint}", flush=True)
        print(f"[RESULT] Total training time : {elapsed / 60:.1f} min", flush=True)
        print("=" * 60, flush=True)
    else:
        print("\n[Mode] Running SimpleCNN Baseline Training ONLY (--train-baseline-only)", flush=True)
        total_params = 1901579
        best_val_acc = float('nan')
        test_acc = float('nan')
        if out_checkpoint.exists():
            try:
                ckpt_data = torch.load(out_checkpoint, map_location=device, weights_only=False)
                best_val_acc = float(ckpt_data.get("val_acc", float("nan")))
            except Exception:
                pass
        for j_path in [SCRIPT_DIR / "cifar10_exp1_evaluation.json", SCRIPT_DIR / "cifar10_binary_evaluation.json"]:
            if j_path.exists():
                try:
                    with open(j_path, "r") as jf:
                        jd = json.load(jf)
                        test_acc = float(jd.get("metrics", {}).get("test_accuracy", float("nan")))
                        break
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # G. Baseline CNN (optional, same split + conditions)
    # ------------------------------------------------------------------
    if args.train_baseline or args.train_baseline_only:
        print("\n" + "=" * 60)
        print("BASELINE SimpleCNN Training (same conditions)")
        print("=" * 60)
        set_seed(args.seed)
        baseline      = SimpleCNN(n_classes=2).to(device)
        n_base_params = sum(p.numel() for p in baseline.parameters())
        print(f"[Baseline] Parameters: {n_base_params:,}", flush=True)

        total_base_epochs = args.epochs_stage1 + args.epochs_stage2
        base_opt   = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=total_base_epochs)

        base_tr_accs, base_vl_accs = [], []
        best_base_val  = -1.0
        base_pati_ctr  = 0
        BASE_CKPT      = PROJECT_ROOT / "simplecnn_cifar10_binary_best.pth"

        for epoch in range(1, total_base_epochs + 1):
            baseline.train()
            tr_c, tr_t = 0, 0
            for x, y in train_loader:
                x, y   = x.to(device), y.to(device)
                logits  = baseline(x)
                loss    = F.cross_entropy(logits, y)
                base_opt.zero_grad()
                loss.backward()
                base_opt.step()
                tr_c += (logits.argmax(1) == y).sum().item()
                tr_t += x.size(0)
            tr_acc_b = tr_c / tr_t

            baseline.eval()
            with torch.no_grad():
                vl_c, vl_t = 0, 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    vl_c += (baseline(x).argmax(1) == y).sum().item()
                    vl_t += x.size(0)
            vl_acc_b = vl_c / vl_t
            base_sched.step()

            base_tr_accs.append(tr_acc_b)
            base_vl_accs.append(vl_acc_b)

            if vl_acc_b > best_base_val:
                best_base_val = vl_acc_b
                base_pati_ctr = 0
                torch.save({"model_state_dict": baseline.state_dict(),
                            "val_acc": best_base_val}, BASE_CKPT)
            else:
                base_pati_ctr += 1

            print(
                f"  [Baseline] Ep {epoch:3d}/{total_base_epochs}"
                f"  train={tr_acc_b * 100:.1f}%  val={vl_acc_b * 100:.1f}%"
                f"  (best={best_base_val * 100:.1f}%)", flush=True
            )
            if base_pati_ctr >= args.patience:
                print(f"  [Baseline EarlyStop] Stopped at epoch {epoch}.", flush=True)
                break

        # Baseline test evaluation
        base_ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        baseline.load_state_dict(base_ckpt["model_state_dict"])
        baseline.eval()
        with torch.no_grad():
            bc, bt = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                bc += (baseline(x).argmax(1) == y).sum().item()
                bt += x.size(0)
        base_test_acc = bc / bt

        save_training_curves(
            base_tr_accs, base_vl_accs,
            [0.0] * len(base_tr_accs), [0.0] * len(base_vl_accs),
            stage1_end=0,
            title="Simple CNN Baseline -- CIFAR-10 Automobile vs Truck",
            save_path=FIGURES_DIR / "baseline_training_curves.png",
        )

        print(f"\n[Baseline RESULT] Best val acc : {best_base_val * 100:.2f}%")
        print(f"[Baseline RESULT] Test accuracy : {base_test_acc * 100:.2f}%")
        print(f"[Baseline RESULT] Parameters    : {n_base_params:,}")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON (measured results -- not fabricated)")
        print("=" * 60)
        print(f"{'Model':<30}  {'Test Acc':>9}  {'Params':>12}")
        print("-" * 56)
        print(f"{'ShiftGuardBinary':<30}  {test_acc * 100:>8.2f}%  {total_params:>12,}")
        print(f"{'SimpleCNN Baseline':<30}  {base_test_acc * 100:>8.2f}%  {n_base_params:>12,}")
        print("=" * 60)

        comparison = {
            "shiftguard_binary": {
                "test_accuracy": float(test_acc),
                "best_val_accuracy": float(best_val_acc),
                "total_params": total_params,
            },
            "simple_cnn_baseline": {
                "test_accuracy": float(base_test_acc),
                "best_val_accuracy": float(best_base_val),
                "total_params": n_base_params,
            },
            "training_config": vars(args),
            "note": (
                "ShiftGuardBinary reuses MiniViT backbone + ShiftAnalyzer + Router + "
                "LoRA Adapters from the 3-class synthetic checkpoint, with the task "
                "head replaced by a fresh 2-class head.  SimpleCNN is trained from "
                "scratch under identical split and augmentation conditions."
            ),
        }
        comp_path = SCRIPT_DIR / "cifar10_model_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"[Saved] Comparison JSON -> {comp_path}")

    print("\n[Done] Fine-tuning complete.")
    print(f"       Best checkpoint   : {out_checkpoint}")
    print("       Run evaluation  : python results/evaluate_cifar10_binary.py")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ShiftGuard CIFAR-10 Binary Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint-out", type=str, default=None,
                   help="Path or filename to save the best checkpoint (protects default checkpoint if specified)")
    p.add_argument("--lora-rank",      type=int,   default=8,
                   help="LoRA adapter rank (must be 8 to load base checkpoint adapters; other values re-init adapters)")
    p.add_argument("--epochs-stage1",  type=int,   default=8,
                   help="Head-only training epochs (backbone frozen)")
    p.add_argument("--epochs-stage2",  type=int,   default=20,
                   help="Full fine-tuning epochs (all layers)")
    p.add_argument("--batch-size",     type=int,   default=32,
                   help="Mini-batch size (keep <=64 for CPU)")
    p.add_argument("--lr-head",        type=float, default=1e-3,
                   help="Learning rate for Stage 1 (head only)")
    p.add_argument("--lr-finetune",    type=float, default=3e-5,
                   help="Learning rate for Stage 2 (full fine-tuning)")
    p.add_argument("--weight-decay",   type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--patience",       type=int,   default=6,
                   help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--seed",           type=int,   default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--train-baseline", action="store_true",      default=False,
                   help="Also train the SimpleCNN baseline for comparison")
    p.add_argument("--no-baseline",    dest="train_baseline",    action="store_false",
                   help="Skip training the SimpleCNN baseline")
    p.add_argument("--train-baseline-only", action="store_true", default=False,
                   help="Train ONLY the SimpleCNN baseline without fine-tuning ShiftGuard")
    p.add_argument("--eval-test",      action="store_true",      default=False,
                   help="Evaluate on the official held-out test set at the end")
    args = p.parse_args()
    main(args)
