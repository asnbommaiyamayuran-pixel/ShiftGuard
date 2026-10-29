"""
verify_evaluation.py
Independent verification of both checkpoints on CIFAR-10 test set.
Does NOT modify any files or overwrite checkpoints.
"""
import sys, json, pickle, time, numpy as np
from pathlib import Path

sys.path.insert(0, r"D:\FYP\ShiftGuard\results")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from finetune_cifar10 import SimpleCNN, CIFAR10BinaryDataset, IMG_SIZE, confidence_and_entropy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Constants
PROJECT_ROOT = Path(r"D:\FYP\ShiftGuard")
CIFAR_TEST_PATH = PROJECT_ROOT / "data" / "cifar-10-batches-py" / "test_batch"
BEST_CKPT = PROJECT_ROOT / "shiftguard_cifar10_exp1_lr1e4.pth"
BASELINE_CKPT = PROJECT_ROOT / "simplecnn_cifar10_binary_best.pth"
BINARY_LABELS = {1: 0, 9: 1}  # CIFAR-10 label 1 -> class 0 (auto), 9 -> class 1 (truck)
BINARY_CLASS_NAMES = ["automobile (car)", "truck"]

def load_cifar10_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels

def filter_automobile_truck(images, labels):
    mask = np.isin(labels, [1, 9])
    imgs = images[mask]
    lbls = np.array([BINARY_LABELS[l] for l in labels[mask]], dtype=np.int64)
    return imgs, lbls

def load_shiftguard_binary(checkpoint_path, device):
    from finetune_cifar10 import ShiftGuardBinary
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ShiftGuardBinary(n_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def load_simplecnn(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SimpleCNN(n_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def evaluate_model(model, test_loader, device, name="Model"):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {name}")
    print(f"{'='*70}")
    
    all_preds, all_confs = [], []
    
    inference_start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, dict):
                all_preds.extend(out["pred"].cpu().numpy().tolist())
                all_confs.extend(out["confidence"].cpu().numpy().tolist())
            else:
                logits = out
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_confs.extend(conf.cpu().numpy().tolist())
    inference_time = time.time() - inference_start
    
    y_pred = np.array(all_preds, dtype=np.int64)
    confs = np.array(all_confs, dtype=np.float64)
    total_params = sum(p.numel() for p in model.parameters())
    
    return y_pred, confs, inference_time, total_params

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load CIFAR-10 test set
print("\n[Data] Loading CIFAR-10 official test set...")
raw_imgs, raw_lbls = load_cifar10_batch(CIFAR_TEST_PATH)
test_imgs, test_lbls = filter_automobile_truck(raw_imgs, raw_lbls)
print(f"[Data] Test images: {len(test_lbls):,} (auto={int((test_lbls==0).sum()):,}, truck={int((test_lbls==1).sum()):,})")

test_ds = CIFAR10BinaryDataset(test_imgs, test_lbls, train=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

# --- Evaluate ShiftGuard ---
print("\n[Model] Loading ShiftGuardBinary checkpoint...")
sg_model, sg_ckpt = load_shiftguard_binary(BEST_CKPT, device)
sg_val_acc = sg_ckpt.get("val_acc", float("nan"))
sg_preds, sg_confs, sg_time, sg_params = evaluate_model(sg_model, test_loader, device, "ShiftGuardBinary")

# --- Evaluate SimpleCNN ---
print("\n[Model] Loading SimpleCNN checkpoint...")
base_model, base_ckpt = load_simplecnn(BASELINE_CKPT, device)
base_val_acc = base_ckpt.get("val_acc", float("nan"))
base_preds, base_confs, base_time, base_params = evaluate_model(base_model, test_loader, device, "SimpleCNN")

# --- Compute metrics for both ---
results = {}
for name, y_pred, confs, inf_time, params, val_acc in [
    ("ShiftGuardBinary", sg_preds, sg_confs, sg_time, sg_params, sg_val_acc),
    ("SimpleCNN", base_preds, base_confs, base_time, base_params, base_val_acc),
]:
    n_images = len(test_lbls)
    ms_per_image = 1000.0 * inf_time / n_images
    
    acc = accuracy_score(test_lbls, y_pred)
    cm = confusion_matrix(test_lbls, y_pred, labels=[0, 1])
    p_per, r_per, f1_per, sup = precision_recall_fscore_support(
        test_lbls, y_pred, labels=[0, 1], zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        test_lbls, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"{'='*70}")
    print(f"  Checkpoint val acc (NOT test): {val_acc*100:.2f}%" if not np.isnan(val_acc) else "  No val_acc in checkpoint")
    print(f"  Test accuracy      : {acc*100:.2f}%")
    print(f"  Macro precision    : {p_macro*100:.2f}%")
    print(f"  Macro recall       : {r_macro*100:.2f}%")
    print(f"  Macro F1           : {f1_macro*100:.2f}%")
    print(f"  Mean max-confidence: {confs.mean()*100:.2f}%")
    print(f"  Inference time     : {inf_time:.2f}s ({ms_per_image:.2f} ms/image)")
    print(f"  Parameters         : {params:,}")
    print(f"\n  Confusion matrix:")
    print(f"    [[{cm[0,0]:4}  {cm[0,1]:4}]")
    print(f"     [{cm[1,0]:4}  {cm[1,1]:4}]]")
    print(f"\n  Per-class:")
    for i, cls in enumerate(BINARY_CLASS_NAMES):
        print(f"    {cls:<20}  prec={p_per[i]*100:.2f}%  rec={r_per[i]*100:.2f}%  f1={f1_per[i]*100:.4f}  supp={int(sup[i])}")
    
    results[name] = {
        "test_accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "ms_per_image": float(ms_per_image),
        "total_params": int(params),
        "confusion_matrix": cm.tolist(),
        "val_accuracy": float(val_acc) if not np.isnan(val_acc) else None,
    }

# Summary comparison
print(f"\n{'='*70}")
print(f"VERIFICATION SUMMARY")
print(f"{'='*70}")
print(f"{'Metric':<35}  {'ShiftGuard':<18}  {'SimpleCNN':<18}")
print(f"{'-'*70}")
for metric in ["test_accuracy", "macro_precision", "macro_recall", "macro_f1", "ms_per_image", "total_params"]:
    sg_val = results["ShiftGuardBinary"][metric]
    base_val = results["SimpleCNN"][metric]
    if metric == "total_params":
        print(f"{metric:<35}  {sg_val:<18,}  {base_val:<18,}")
    else:
        print(f"{metric:<35}  {sg_val:<18.4f}  {base_val:<18.4f}")

# Save verification results
verify_path = Path(r"D:\FYP\ShiftGuard\results") / "verification_results.json"
with open(verify_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] Verification results -> {verify_path}")
