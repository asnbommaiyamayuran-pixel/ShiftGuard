"""
================================================================================
ShiftGuard: Visualization & Publication Plot Generation Suite
================================================================================
Generates high-resolution PNG charts & reliability diagrams:
  1. Training Loss & Accuracy Curves (shiftguard_training_curves.png)
  2. ECE Reliability Diagrams comparing Baseline vs. ShiftGuard (shiftguard_reliability_diagram.png)
  3. Adapter Activation & Gating Distributions (shiftguard_adapter_gating.png)
  4. In-Distribution vs. OOD Predictive Entropy Histograms (shiftguard_ood_entropy.png)
================================================================================
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shiftguard import (
    set_seed, SyntheticShiftDataset, ShiftGuard, NoAdapterBaseline,
    ID_CLASSES, collate_fn, expected_calibration_error
)

# Set clean matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def train_and_record_metrics(model, train_loader, optimizer, device, epochs=8):
    model.train()
    history = {"epoch": [], "loss": [], "accuracy": []}
    for epoch in range(1, epochs + 1):
        loss_sum, correct, total = 0.0, 0, 0
        for x, y, domain_id, mod_flag, sem_flag in train_loader:
            x, y = x.to(device), y.to(device)
            domain_id, mod_flag, sem_flag = domain_id.to(device), mod_flag.to(device), sem_flag.to(device)
            id_mask = sem_flag < 0.5
            out = model(x)
            
            task_loss = torch.tensor(0.0, device=device)
            if id_mask.any():
                task_loss = F.cross_entropy(out["logits"][id_mask], y[id_mask])
            
            if "domain_logits" in out:
                domain_loss = F.cross_entropy(out["domain_logits"], domain_id)
                modality_loss = F.binary_cross_entropy(out["s_m"], mod_flag)
                semantic_loss = F.binary_cross_entropy(out["s_s"], sem_flag)
                loss = task_loss + 0.3 * domain_loss + 0.3 * modality_loss + 0.3 * semantic_loss
            else:
                loss = task_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item() * x.size(0)
            if id_mask.any():
                total += id_mask.sum().item()
                correct += (out["pred"][id_mask] == y[id_mask]).sum().item()
        
        avg_loss = loss_sum / len(train_loader.dataset)
        accuracy = correct / max(total, 1)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    return history


def plot_training_curves(history, output_path="shiftguard_training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history["epoch"], history["loss"], 'o-', color='#1f77b4', linewidth=2.5, label='Loss')
    ax1.set_title('ShiftGuard Joint Loss Trajectory', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(history["epoch"], [acc * 100 for acc in history["accuracy"]], 's-', color='#2ca02c', linewidth=2.5, label='Accuracy (%)')
    ax2.set_title('Training Accuracy Trajectory', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


@torch.no_grad()
def collect_eval_outputs(model, loader, device):
    model.eval()
    all_conf, all_correct, all_entropy, all_is_ood = [], [], [], []
    adapter_weights = []
    
    for x, y, domain_id, mod_flag, sem_flag in loader:
        x, y, sem_flag = x.to(device), y.to(device), sem_flag.to(device)
        out = model(x)
        
        id_mask = sem_flag < 0.5
        if id_mask.any():
            all_correct.append((out["pred"][id_mask] == y[id_mask]).float().cpu().numpy())
            all_conf.append(out["confidence"][id_mask].cpu().numpy())
        
        all_entropy.append(out["entropy"].cpu().numpy())
        all_is_ood.append(sem_flag.cpu().numpy())
        if "adapter_weights" in out:
            adapter_weights.append(out["adapter_weights"].cpu().numpy())

    conf = np.concatenate(all_conf) if all_conf else np.array([])
    corr = np.concatenate(all_correct) if all_correct else np.array([])
    ent = np.concatenate(all_entropy) if all_entropy else np.array([])
    ood = np.concatenate(all_is_ood)
    weights = np.concatenate(adapter_weights) if adapter_weights else None
    
    return conf, corr, ent, ood, weights


def plot_reliability_diagram(sg_conf, sg_corr, base_conf, base_corr, output_path="shiftguard_reliability_diagram.png", n_bins=10):
    fig, ax = plt.subplots(figsize=(7, 6))
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Calculate bin accuracies
    def compute_bin_data(confidences, correct):
        bin_accs, bin_confs = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(confidences[mask].mean())
            else:
                bin_accs.append(0)
                bin_confs.append((lo + hi) / 2)
        return np.array(bin_accs), np.array(bin_confs)

    sg_accs, sg_confs = compute_bin_data(sg_conf, sg_corr)
    base_accs, base_confs = compute_bin_data(base_conf, base_corr)
    
    sg_ece = expected_calibration_error(sg_conf, sg_corr, n_bins=n_bins)
    base_ece = expected_calibration_error(base_conf, base_corr, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (ECE=0.0)')
    ax.bar(bins[:-1] + 0.02, sg_accs, width=0.04, alpha=0.7, color='#2ca02c', label=f'ShiftGuard (ECE = {sg_ece:.4f})')
    ax.bar(bins[:-1] + 0.06, base_accs, width=0.04, alpha=0.7, color='#d62728', label=f'Baseline (ECE = {base_ece:.4f})')

    ax.set_title('Reliability Diagram: Calibration Benchmark', fontsize=13, fontweight='bold')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_adapter_gating(adapter_weights, output_path="shiftguard_adapter_gating.png"):
    if adapter_weights is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ['Domain Adapter', 'Modality Adapter', 'Semantic Adapter']
    means = adapter_weights.mean(axis=0)
    stds = adapter_weights.std(axis=0)
    colors = ['#1f77b4', '#ff7f0e', '#9467bd']

    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_title('Selective Adapter Activation Means & Variances', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Gating Weight')
    ax.set_ylim([0, 1.1])
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ood_entropy(entropy, is_ood, output_path="shiftguard_ood_entropy.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    id_entropy = entropy[is_ood == 0]
    ood_entropy = entropy[is_ood == 1]

    ax.hist(id_entropy, bins=20, alpha=0.6, color='#1f77b4', label=f'In-Distribution (ID, N={len(id_entropy)})', density=True)
    ax.hist(ood_entropy, bins=20, alpha=0.6, color='#d62728', label=f'Out-of-Distribution (OOD, N={len(ood_entropy)})', density=True)

    ax.set_title('Predictive Entropy Distribution (ID vs. OOD)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Normalized Predictive Entropy')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots():
    print("=" * 80)
    print(" GENERATING SHIFTGUARD PUBLICATION VISUALIZATIONS")
    print("=" * 80)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SyntheticShiftDataset(3000, split="train", include_ood=False, seed=42)
    val_compound_ds = SyntheticShiftDataset(800, split="val_compound", include_ood=True, seed=43)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_compound_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 1. Train ShiftGuard & record history
    print("\n---> Training ShiftGuard model...")
    sg_model = ShiftGuard(n_classes=len(ID_CLASSES)).to(device)
    sg_optimizer = torch.optim.AdamW(sg_model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = train_and_record_metrics(sg_model, train_loader, sg_optimizer, device, epochs=8)
    
    # 2. Train Baseline
    print("\n---> Training Baseline model...")
    base_model = NoAdapterBaseline(n_classes=len(ID_CLASSES)).to(device)
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=3e-4, weight_decay=1e-4)
    _ = train_and_record_metrics(base_model, train_loader, base_optimizer, device, epochs=8)

    # 3. Collect outputs
    print("\n---> Evaluating models for visualization...")
    sg_conf, sg_corr, sg_ent, sg_ood, sg_weights = collect_eval_outputs(sg_model, val_loader, device)
    base_conf, base_corr, _, _, _ = collect_eval_outputs(base_model, val_loader, device)

    # 4. Generate & Save Figures
    print("\n---> Rendering and saving figures...")
    plot_training_curves(history)
    plot_reliability_diagram(sg_conf, sg_corr, base_conf, base_corr)
    plot_adapter_gating(sg_weights)
    plot_ood_entropy(sg_ent, sg_ood)

    print("\n" + "=" * 80)
    print(" ALL VISUALIZATION PLOTS GENERATED SUCCESSFULLY!")
    print(" Output files created in c:\\Users\\DELL\\Downloads\\File:")
    print("   - shiftguard_training_curves.png")
    print("   - shiftguard_reliability_diagram.png")
    print("   - shiftguard_adapter_gating.png")
    print("   - shiftguard_ood_entropy.png")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_plots()
