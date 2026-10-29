r"""
================================================================================
evaluate_cifar10_binary.py  --  ShiftGuard CIFAR-10 Binary Evaluation
================================================================================
Loads the best checkpoint produced by finetune_cifar10.py and evaluates it
on the OFFICIAL CIFAR-10 test set (automobile + truck classes only).

The test set is NEVER used during training or model selection.  This script
must be run AFTER finetune_cifar10.py has completed.

Evaluation scope
  CIFAR-10 test batch -> automobile (label 1) + truck (label 9) only
  Binary mapping: automobile -> 0, truck -> 1

Outputs
  - Console report (accuracy, per-class P/R/F1, macro F1, confusion matrix)
  - results/cifar10_binary_evaluation.json    (machine-readable metrics)
  - results/cifar10_binary_evaluation.txt     (human-readable text report)
  - results/figures/shiftguard_confusion_matrix.png

Usage
  cd D:\FYP\ShiftGuard\results
  python evaluate_cifar10_binary.py
================================================================================
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT     = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

BEST_CHECKPOINT  = PROJECT_ROOT / "shiftguard_cifar10_binary_best.pth"
CIFAR_TEST_PATH  = PROJECT_ROOT / "data" / "cifar-10-batches-py" / "test_batch"
FIGURES_DIR      = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON     = SCRIPT_DIR / "cifar10_binary_evaluation.json"
RESULTS_TXT      = SCRIPT_DIR / "cifar10_binary_evaluation.txt"
CONFUSION_PNG    = FIGURES_DIR / "shiftguard_confusion_matrix.png"

# ---------------------------------------------------------------------------
# CIFAR-10 label constants (must match finetune_cifar10.py)
# ---------------------------------------------------------------------------
CIFAR_AUTOMOBILE   = 1
CIFAR_TRUCK        = 9
BINARY_LABELS      = {CIFAR_AUTOMOBILE: 0, CIFAR_TRUCK: 1}
BINARY_CLASS_NAMES = ["automobile (car)", "truck"]


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_cifar10_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def filter_automobile_truck(images, labels):
    mask = np.isin(labels, [CIFAR_AUTOMOBILE, CIFAR_TRUCK])
    imgs = images[mask]
    lbls = np.array([BINARY_LABELS[l] for l in labels[mask]], dtype=np.int64)
    return imgs, lbls


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def load_finetuned_model(checkpoint_path, device):
    """
    Load ShiftGuardBinary from the fine-tuned checkpoint.
    Imports ShiftGuardBinary from finetune_cifar10 (same module as was used
    to train) to guarantee architectural consistency.
    """
    from finetune_cifar10 import ShiftGuardBinary
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ShiftGuardBinary(n_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Checkpoint] Loaded from {checkpoint_path}")
    print(f"  Saved at epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"  Stage          : {ckpt.get('stage', 'unknown')}")
    print(f"  Val accuracy   : {ckpt.get('val_acc', float('nan')) * 100:.2f}%  "
          "(NOT the test accuracy)")
    return model, ckpt


# ===========================================================================
# CONFUSION MATRIX PLOT
# ===========================================================================

def save_confusion_matrix(cm, class_names, title, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=title,
        )
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Confusion matrix saved -> {save_path}")
    except Exception as exc:
        print(f"[Plot] Could not save confusion matrix: {exc}")


# ===========================================================================
# SAMPLE IMAGES VISUALISATION
# ===========================================================================

def save_sample_predictions(images, true_lbls, pred_lbls, confidences,
                              n_show=8, save_path=None):
    """
    Save a grid of correctly and incorrectly classified test images with
    true label, predicted label, and confidence.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        correct_idx   = np.where(true_lbls == pred_lbls)[0]
        incorrect_idx = np.where(true_lbls != pred_lbls)[0]

        n_correct   = min(n_show // 2, len(correct_idx))
        n_incorrect = min(n_show - n_correct, len(incorrect_idx))

        rng   = np.random.default_rng(0)
        shown_c = rng.choice(correct_idx,   n_correct,   replace=False) if n_correct   else []
        shown_i = rng.choice(incorrect_idx, n_incorrect, replace=False) if n_incorrect else []
        shown   = list(shown_c) + list(shown_i)

        if not shown:
            return

        n_cols = min(4, len(shown))
        n_rows = (len(shown) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for ax_flat, idx in zip(axes.flat, shown):
            img  = images[idx].astype(np.uint8)
            true = BINARY_CLASS_NAMES[true_lbls[idx]]
            pred = BINARY_CLASS_NAMES[pred_lbls[idx]]
            conf = confidences[idx]
            ok   = true_lbls[idx] == pred_lbls[idx]
            color = "green" if ok else "red"
            ax_flat.imshow(img)
            ax_flat.set_title(
                f"True: {true}\nPred: {pred}  ({conf:.2f})",
                fontsize=8, color=color,
            )
            ax_flat.axis("off")

        # hide any unused axes
        for ax_flat in list(axes.flat)[len(shown):]:
            ax_flat.axis("off")

        plt.suptitle("Sample Test Predictions  (green=correct, red=wrong)", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Sample predictions saved -> {save_path}")
    except Exception as exc:
        print(f"[Plot] Could not save sample predictions: {exc}")


# ===========================================================================
# MAIN EVALUATION
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ShiftGuard CIFAR-10 Binary Evaluation")
    p.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT),
                   help="Path to model checkpoint to evaluate")
    p.add_argument("--output-json", type=str, default=str(RESULTS_JSON),
                   help="Path to output JSON metrics")
    p.add_argument("--output-txt", type=str, default=str(RESULTS_TXT),
                   help="Path to output text report")
    p.add_argument("--output-cm", type=str, default=str(CONFUSION_PNG),
                   help="Path to output confusion matrix PNG")
    p.add_argument("--output-samples", type=str, default=str(FIGURES_DIR / "shiftguard_sample_predictions.png"),
                   help="Path to sample predictions PNG")
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    results_json = Path(args.output_json)
    if not results_json.is_absolute():
        results_json = PROJECT_ROOT / results_json

    results_txt = Path(args.output_txt)
    if not results_txt.is_absolute():
        results_txt = PROJECT_ROOT / results_txt

    confusion_png = Path(args.output_cm)
    if not confusion_png.is_absolute():
        confusion_png = PROJECT_ROOT / confusion_png

    sample_png = Path(args.output_samples)
    if not sample_png.is_absolute():
        sample_png = PROJECT_ROOT / sample_png

    # ------------------------------------------------------------------ #
    # Checks                                                               #
    # ------------------------------------------------------------------ #
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_CHECKPOINT}\n"
            "Please run:  python results/finetune_cifar10.py  first."
        )
    if not CIFAR_TEST_PATH.exists():
        raise FileNotFoundError(
            f"CIFAR-10 test_batch not found: {CIFAR_TEST_PATH}\n"
            "Expected: data/cifar-10-batches-py/test_batch"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("ShiftGuard CIFAR-10 Binary Evaluation")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Test data  : {CIFAR_TEST_PATH}")

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    model, ckpt_meta = load_finetuned_model(checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # ------------------------------------------------------------------ #
    # Load and verify test data                                            #
    # ------------------------------------------------------------------ #
    print("\n[Data] Loading CIFAR-10 test batch ...")
    raw_imgs, raw_lbls = load_cifar10_batch(CIFAR_TEST_PATH)
    test_imgs, test_lbls = filter_automobile_truck(raw_imgs, raw_lbls)

    n_auto  = int((test_lbls == 0).sum())
    n_truck = int((test_lbls == 1).sum())
    print(f"[Data] Test images total     : {len(test_lbls):,}")
    print(f"[Data]   automobile (class 0): {n_auto:,}")
    print(f"[Data]   truck      (class 1): {n_truck:,}")

    assert n_auto > 0  and n_truck > 0, "Expected both classes in test set"
    assert n_auto == 1000 and n_truck == 1000, (
        f"Unexpected test counts: auto={n_auto}, truck={n_truck}. "
        "CIFAR-10 test set should have exactly 1000 per class."
    )

    # ------------------------------------------------------------------ #
    # Preprocessing  (must match training preprocessing)                  #
    # ------------------------------------------------------------------ #
    import torchvision.transforms as transforms
    from finetune_cifar10 import CIFAR10BinaryDataset, IMG_SIZE as FT_IMG_SIZE

    test_ds = CIFAR10BinaryDataset(test_imgs, test_lbls, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #
    print("\n[Eval] Running inference ...")
    all_preds   = []
    all_confs   = []
    inference_start = time.time()

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(out["pred"].cpu().numpy().tolist())
            all_confs.extend(out["confidence"].cpu().numpy().tolist())

    inference_time = time.time() - inference_start
    n_images = len(test_lbls)
    ms_per_image = 1000.0 * inference_time / n_images

    y_true = test_lbls
    y_pred = np.array(all_preds, dtype=np.int64)
    confs  = np.array(all_confs, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])

    p_per, r_per, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=BINARY_CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    # ------------------------------------------------------------------ #
    # Console report                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS  (official CIFAR-10 test set, held-out)")
    print("=" * 70)
    print(f"Images evaluated          : {n_images:,}")
    print(f"  automobile (true class 0): {n_auto:,}")
    print(f"  truck      (true class 1): {n_truck:,}")
    print()
    print(f"Test accuracy             : {acc * 100:.2f}%")
    print(f"Macro precision           : {p_macro * 100:.2f}%")
    print(f"Macro recall              : {r_macro * 100:.2f}%")
    print(f"Macro F1-score            : {f1_macro * 100:.2f}%")
    print(f"Mean max-softmax conf     : {confs.mean() * 100:.2f}%")
    print()
    print("Per-class metrics:")
    print(f"  {'Class':<25}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("  " + "-" * 65)
    for i, cls in enumerate(BINARY_CLASS_NAMES):
        print(f"  {cls:<25}  {p_per[i] * 100:>9.2f}%  "
              f"{r_per[i] * 100:>7.2f}%  "
              f"{f1_per[i] * 100:>7.2f}%  "
              f"{int(sup[i]):>8,}")
    print()
    print("Confusion matrix  (rows=true, cols=predicted):")
    print(f"  {'':30}  pred_{BINARY_CLASS_NAMES[0]}  pred_{BINARY_CLASS_NAMES[1]}")
    for i, cls in enumerate(BINARY_CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:8,}" for j in range(2))
        print(f"  true_{cls:<25}  {row}")
    print()
    print("Full sklearn classification report:")
    print(report_str)
    print(f"Inference time total      : {inference_time:.2f} s  "
          f"({ms_per_image:.2f} ms/image)")
    print(f"Model parameter count     : {total_params:,}")
    print("=" * 70)

    target_note = (
        "TARGET MET (>=75%)" if acc >= 0.75
        else f"Target 75% NOT met (actual={acc * 100:.2f}%); see limitations."
    )
    print(f"\n[Target assessment] {target_note}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    save_confusion_matrix(
        cm, BINARY_CLASS_NAMES,
        title=f"ShiftGuard CIFAR-10 Binary Confusion Matrix\n(Acc={acc*100:.2f}%)",
        save_path=confusion_png,
    )
    save_sample_predictions(
        test_imgs, y_true, y_pred, confs,
        n_show=8,
        save_path=sample_png,
    )

    # ------------------------------------------------------------------ #
    # Save JSON + TXT reports                                              #
    # ------------------------------------------------------------------ #
    results = {
        "dataset": "CIFAR-10 official test set (automobile + truck only)",
        "n_images": n_images,
        "n_automobile": n_auto,
        "n_truck": n_truck,
        "binary_class_names": BINARY_CLASS_NAMES,
        "cifar10_automobile_label": CIFAR_AUTOMOBILE,
        "cifar10_truck_label": CIFAR_TRUCK,
        "model": "ShiftGuardBinary (MiniViT + ShiftAnalyzer + Router + LoRA Adapters + 2-class head)",
        "checkpoint": str(checkpoint_path),
        "val_accuracy_at_selection": float(ckpt_meta.get("val_acc", float("nan"))),
        "metrics": {
            "test_accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "mean_max_softmax_confidence": float(confs.mean()),
            "per_class": {
                cls: {
                    "precision": float(p_per[i]),
                    "recall": float(r_per[i]),
                    "f1": float(f1_per[i]),
                    "support": int(sup[i]),
                }
                for i, cls in enumerate(BINARY_CLASS_NAMES)
            },
            "confusion_matrix": cm.tolist(),
        },
        "inference": {
            "total_seconds": float(inference_time),
            "ms_per_image": float(ms_per_image),
            "device": str(device),
        },
        "model_params": total_params,
        "target_75pct": acc >= 0.75,
        "target_note": target_note,
    }

    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] JSON report  -> {results_json}")

    # Human-readable text report
    txt_lines = [
        "=" * 70,
        "ShiftGuard CIFAR-10 Binary Evaluation Report",
        "=" * 70,
        f"Checkpoint   : {checkpoint_path}",
        f"Val acc (selection criterion, NOT test): {ckpt_meta.get('val_acc', float('nan'))*100:.2f}%",
        "",
        f"Test accuracy        : {acc * 100:.2f}%",
        f"Macro precision      : {p_macro * 100:.2f}%",
        f"Macro recall         : {r_macro * 100:.2f}%",
        f"Macro F1-score       : {f1_macro * 100:.2f}%",
        f"Mean max-conf        : {confs.mean() * 100:.2f}%",
        f"Inference ms/image   : {ms_per_image:.2f}",
        f"Model parameters     : {total_params:,}",
        "",
        "Per-class report:",
        report_str,
        f"Confusion matrix:",
        f"  {cm}",
        "",
        f"Target assessment: {target_note}",
        "=" * 70,
    ]
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"[Saved] Text report  -> {results_txt}")
    print(f"[Saved] Confusion PNG-> {confusion_png}")


if __name__ == "__main__":
    main()
