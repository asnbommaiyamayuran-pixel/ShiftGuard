"""
Custom corruption benchmark for existing ShiftGuardBinary and SimpleCNN checkpoints.
Run from project root: python results\evaluate_corruption_benchmark.py
This is NOT the official CIFAR-10-C benchmark. No training/checkpoints are modified.
"""
import csv, io, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "cifar-10-batches-py" / "test_batch"
SG_CKPT = ROOT / "shiftguard_cifar10_exp1_lr1e4.pth"
CNN_CKPT = ROOT / "simplecnn_cifar10_binary_best.pth"
sys.path.insert(0, str(RESULTS))
from finetune_cifar10 import ShiftGuardBinary, SimpleCNN, CIFAR10BinaryDataset

SEED, SEVERITY, BATCH_SIZE = 42, 3, 64

def load_test(path):
    with open(path, "rb") as f:
        b = pickle.load(f, encoding="bytes")
    imgs = b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(b[b"labels"], dtype=np.int64)
    mask = np.isin(labels, [1, 9])
    return imgs[mask], np.where(labels[mask] == 1, 0, 1).astype(np.int64)

def corrupt(arr, kind, severity, seed):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if kind == "clean":
        out = im
    elif kind == "gaussian_noise":
        sigma = [8, 16, 24, 32, 40][severity-1]
        x = np.asarray(im).astype(np.float32)
        out = Image.fromarray(np.clip(x + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8))
    elif kind == "defocus_blur":
        out = im.filter(ImageFilter.GaussianBlur(radius=[0.5, 1, 1.5, 2, 2.5][severity-1]))
    elif kind == "brightness":
        out = ImageEnhance.Brightness(im).enhance([0.6, 0.75, 0.9, 1.15, 1.3][severity-1])
    elif kind == "jpeg_compression":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=[50, 35, 25, 15, 10][severity-1])
        buf.seek(0)
        out = Image.open(buf).convert("RGB").copy()
    else:
        raise ValueError(kind)
    return np.asarray(out, dtype=np.uint8)

def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins+1)
    val = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & ((conf <= edges[i+1]) if i == bins-1 else (conf < edges[i+1]))
        if mask.any():
            val += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(val)

def load_models(device):
    for p in (SG_CKPT, CNN_CKPT):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    sck = torch.load(SG_CKPT, map_location=device, weights_only=False)
    sg = ShiftGuardBinary(n_classes=2).to(device)
    sg.load_state_dict(sck["model_state_dict"]); sg.eval()
    cck = torch.load(CNN_CKPT, map_location=device, weights_only=False)
    cnn = SimpleCNN(n_classes=2).to(device)
    cnn.load_state_dict(cck["model_state_dict"]); cnn.eval()
    return {"ShiftGuard": sg, "SimpleCNN": cnn}

def evaluate(model, loader, device):
    ys, preds, confs = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            out = model(x.to(device))
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            ys.extend(y.numpy().tolist()); preds.extend(pred.cpu().numpy().tolist())
            confs.extend(conf.cpu().numpy().tolist())
    ys, preds, confs = np.asarray(ys), np.asarray(preds), np.asarray(confs)
    return {"accuracy": float(accuracy_score(ys, preds)),
            "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
            "ece": ece(confs, (ys == preds).astype(float)), "n_images": int(len(ys))}

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    images, labels = load_test(DATA)
    print(f"Official test subset: {len(labels)} images; automobile={(labels==0).sum()}, truck={(labels==1).sum()}")
    if len(labels) != 2000 or (labels == 0).sum() != 1000 or (labels == 1).sum() != 1000:
        raise RuntimeError("Expected exactly 2,000 test images with 1,000 per class.")
    models = load_models(device)
    kinds = ["clean", "gaussian_noise", "defocus_blur", "brightness", "jpeg_compression"]
    rows = []
    for kind in kinds:
        print(f"\n=== {kind} ===")
        sev = 3 if kind != "clean" else 1
        altered = np.stack([corrupt(img, kind, sev, SEED+i) for i, img in enumerate(images)])
        ds = CIFAR10BinaryDataset(altered, labels, train=False)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for name, model in models.items():
            metrics = evaluate(model, loader, device)
            row = {"corruption": kind, "severity": 0 if kind == "clean" else sev, "model": name, **metrics}
            rows.append(row)
            print(f"{name:<12} accuracy={metrics['accuracy']*100:.2f}%  macro-F1={metrics['macro_f1']*100:.2f}%  ECE={metrics['ece']:.4f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    jp, cp, pp = RESULTS/"corruption_benchmark_results.json", RESULTS/"corruption_benchmark_results.csv", RESULTS/"corruption_benchmark_accuracy.png"
    with jp.open("w", encoding="utf-8") as f:
        json.dump({"experiment":"Custom CIFAR-10 automobile/truck corruption benchmark",
                   "official_cifar10_c":False, "test_images":len(labels),
                   "class_counts":{"automobile":1000,"truck":1000}, "severity":SEVERITY,
                   "seed":SEED, "results":rows}, f, indent=2)
    with cp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(kinds)); width = 0.36
    for offset, name in [(-width/2,"ShiftGuard"),(width/2,"SimpleCNN")]:
        vals = [next(r["accuracy"]*100 for r in rows if r["corruption"]==k and r["model"]==name) for k in kinds]
        ax.bar(x+offset, vals, width, label=name)
    ax.set_xticks(x); ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,100)
    ax.set_title("Custom CIFAR-10 Corruption Benchmark"); ax.legend()
    fig.tight_layout(); fig.savefig(pp, dpi=200); plt.close(fig)
    print("\nSaved:", jp, "\nSaved:", cp, "\nSaved:", pp)
    print("Note: this is a custom corruption experiment, not official CIFAR-10-C.")

if __name__ == "__main__":
    main()
