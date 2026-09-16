"""
================================================================================
ShiftGuard: Comprehensive Deeper Ablation & Statistical Benchmark Suite
================================================================================
This script executes Option 5:
  1. Router Gating Ablations (Soft Gating vs. Hard Top-1 vs. Uniform/No Routing)
  2. Shift Analyzer Detector Ablations (Domain-Only, Modality-Only, Semantic-Only)
  3. Adapter Activation Statistics across shift conditions
  4. Multi-Seed Statistical Rigor (Mean ± Std across 3 seeds)
  5. Computational Efficiency & Latency Benchmark (Backbone vs. Full ShiftGuard)
================================================================================
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shiftguard import (
    set_seed, SyntheticShiftDataset, ShiftGuard, NoAdapterBaseline,
    ID_CLASSES, evaluate, evaluate_baseline, expected_calibration_error,
    collate_fn
)
from sklearn.metrics import roc_auc_score


# =============================================================================
# ABLATION VARIANTS OF SHIFTGUARD
# =============================================================================

class ShiftGuardHardRouting(ShiftGuard):
    """Variant 1: Hard Top-1 Gating (Selects only argmax adapter)."""

    def forward(self, x):
        f = self.backbone(x)
        S, domain_logits, s_m, s_s = self.shift_analyzer(f)
        weights = self.router(S)
        
        # Hard top-1 selection
        top1_idx = torch.argmax(weights, dim=-1, keepdim=True)
        hard_w = torch.zeros_like(weights).scatter_(-1, top1_idx, 1.0)
        
        fused, eff_w = self.adapters(f, hard_w)
        logits = self.task_head(fused)
        probs, pred, confidence, entropy = confidence_and_entropy_local(logits)
        return {
            "logits": logits, "probs": probs, "pred": pred,
            "confidence": confidence, "entropy": entropy,
            "shift_signature": S, "domain_logits": domain_logits,
            "s_m": s_m, "s_s": s_s, "adapter_weights": eff_w,
        }


class ShiftGuardUniformRouting(ShiftGuard):
    """Variant 2: Uniform / No Routing (Fixed equal weights 1/3, 1/3, 1/3)."""

    def forward(self, x):
        f = self.backbone(x)
        S, domain_logits, s_m, s_s = self.shift_analyzer(f)
        batch_size = x.size(0)
        uniform_w = torch.full((batch_size, 3), 1.0 / 3.0, device=x.device)
        
        fused, eff_w = self.adapters(f, uniform_w)
        logits = self.task_head(fused)
        probs, pred, confidence, entropy = confidence_and_entropy_local(logits)
        return {
            "logits": logits, "probs": probs, "pred": pred,
            "confidence": confidence, "entropy": entropy,
            "shift_signature": S, "domain_logits": domain_logits,
            "s_m": s_m, "s_s": s_s, "adapter_weights": eff_w,
        }


class ShiftGuardMaskedAnalyzer(ShiftGuard):
    """Variant 3: Masked Shift Analyzer (Keep only specified shift detector channels active)."""

    def __init__(self, n_classes: int = len(ID_CLASSES), rank: int = 8, active_mask=(1, 1, 1)):
        super().__init__(n_classes=n_classes, rank=rank)
        self.active_mask = active_mask  # tuple of (domain_active, modality_active, semantic_active)

    def forward(self, x):
        f = self.backbone(x)
        S, domain_logits, s_m, s_s = self.shift_analyzer(f)
        
        # Apply mask to shift signature S
        mask_tensor = torch.tensor(self.active_mask, dtype=S.dtype, device=S.device)
        S_masked = S * mask_tensor.unsqueeze(0)
        
        weights = self.router(S_masked)
        fused, eff_w = self.adapters(f, weights)
        logits = self.task_head(fused)
        probs, pred, confidence, entropy = confidence_and_entropy_local(logits)
        return {
            "logits": logits, "probs": probs, "pred": pred,
            "confidence": confidence, "entropy": entropy,
            "shift_signature": S_masked, "domain_logits": domain_logits,
            "s_m": s_m, "s_s": s_s, "adapter_weights": eff_w,
        }


def confidence_and_entropy_local(logits: torch.Tensor):
    probs = F.softmax(logits, dim=-1)
    confidence, pred = probs.max(dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1)
    entropy = entropy / np.log(logits.shape[-1])
    return probs, pred, confidence, entropy


# =============================================================================
# EXPERIMENT EXECUTION HELPERS
# =============================================================================

def train_model(model, train_loader, optimizer, device, epochs=5):
    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
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


@torch.no_grad()
def collect_adapter_statistics(model, loader, device):
    model.eval()
    domain_weights, modality_weights, semantic_weights = [], [], []
    for x, y, domain_id, mod_flag, sem_flag in loader:
        x = x.to(device)
        out = model(x)
        weights = out["adapter_weights"].cpu().numpy()  # [B, 3]
        domain_weights.append(weights[:, 0])
        modality_weights.append(weights[:, 1])
        semantic_weights.append(weights[:, 2])
    
    return {
        "domain_adapter_mean": float(np.mean(np.concatenate(domain_weights))),
        "modality_adapter_mean": float(np.mean(np.concatenate(modality_weights))),
        "semantic_adapter_mean": float(np.mean(np.concatenate(semantic_weights))),
    }


def benchmark_latency(model, loader, device, num_runs=50):
    model.eval()
    # Warmup
    x, _, _, _, _ = next(iter(loader))
    x = x.to(device)
    for _ in range(10):
        _ = model(x)
    
    start_time = time.perf_counter()
    total_samples = 0
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
            total_samples += x.size(0)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    latency_per_sample_ms = (total_time / total_samples) * 1000.0
    fps = total_samples / total_time
    return latency_per_sample_ms, fps


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiments():
    print("=" * 80)
    print(" SHIFTGUARD DEEPER ABLATION & STATISTICAL EXPERIMENTAL SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    epochs = 6
    batch_size = 64
    train_size = 3000
    eval_size = 800

    # Datasets
    train_ds = SyntheticShiftDataset(train_size, split="train", include_ood=False, seed=42)
    val_id_ds = SyntheticShiftDataset(eval_size, split="val_id", include_ood=False, seed=43)
    val_compound_ds = SyntheticShiftDataset(eval_size, split="val_compound", include_ood=True, seed=44)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_id_loader = DataLoader(val_id_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_compound_loader = DataLoader(val_compound_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: ROUTING MECHANISM ABLATIONS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" EXPERIMENT 1: ROUTING MECHANISM ABLATIONS")
    print(" (Soft Gating vs. Hard Top-1 Gating vs. Uniform/No Routing)")
    print("-" * 80)

    models_to_test = {
        "Standard ShiftGuard (Soft Gating)": ShiftGuard(n_classes=len(ID_CLASSES)).to(device),
        "ShiftGuard (Hard Top-1 Routing)": ShiftGuardHardRouting(n_classes=len(ID_CLASSES)).to(device),
        "ShiftGuard (Uniform Gating)": ShiftGuardUniformRouting(n_classes=len(ID_CLASSES)).to(device),
    }

    routing_results = {}
    for name, model in models_to_test.items():
        print(f"\n---> Training {name}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train_model(model, train_loader, optimizer, device, epochs=epochs)
        
        print(f"Evaluating {name}:")
        res_id = evaluate(model, val_id_loader, device, tag=f"{name} [ID]")
        res_cmp = evaluate(model, val_compound_loader, device, tag=f"{name} [Compound]")
        routing_results[name] = {"ID": res_id, "Compound": res_cmp}

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: SHIFT ANALYZER DETECTOR HEAD ABLATIONS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" EXPERIMENT 2: SHIFT ANALYZER HEAD ABLATIONS")
    print(" (Evaluating impact of removing individual detector channels from S)")
    print("-" * 80)

    detector_variants = {
        "All Heads Active (1, 1, 1)": (1, 1, 1),
        "Domain-Only (1, 0, 0)": (1, 0, 0),
        "Modality-Only (0, 1, 0)": (0, 1, 0),
        "Semantic-Only (0, 0, 1)": (0, 0, 1),
    }

    detector_results = {}
    for name, mask in detector_variants.items():
        print(f"\n---> Training Detector Variant: {name}...")
        model = ShiftGuardMaskedAnalyzer(n_classes=len(ID_CLASSES), active_mask=mask).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train_model(model, train_loader, optimizer, device, epochs=epochs)
        
        res_cmp = evaluate(model, val_compound_loader, device, tag=name)
        detector_results[name] = res_cmp

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: ADAPTER ACTIVATION STATISTICS
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" EXPERIMENT 3: ADAPTER ACTIVATION STATISTICS")
    print(" (Mean activation weights across Domain, Modality & Semantic adapters)")
    print("-" * 80)

    standard_model = models_to_test["Standard ShiftGuard (Soft Gating)"]
    activations_id = collect_adapter_statistics(standard_model, val_id_loader, device)
    activations_compound = collect_adapter_statistics(standard_model, val_compound_loader, device)

    print("\n[In-Distribution Split Adapter Activation Means]:")
    for k, v in activations_id.items():
        print(f"  {k:<25s}: {v:.4f}")

    print("\n[Compound Shift Split Adapter Activation Means]:")
    for k, v in activations_compound.items():
        print(f"  {k:<25s}: {v:.4f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: LATENCY & COMPUTATIONAL THROUGHPUT BENCHMARK
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" EXPERIMENT 4: INFERENCE LATENCY & THROUGHPUT BENCHMARK")
    print("-" * 80)

    baseline_model = NoAdapterBaseline(n_classes=len(ID_CLASSES)).to(device)
    base_opt = torch.optim.AdamW(baseline_model.parameters(), lr=3e-4)
    train_model(baseline_model, train_loader, base_opt, device, epochs=2)

    lat_base, fps_base = benchmark_latency(baseline_model, val_compound_loader, device)
    lat_sg, fps_sg = benchmark_latency(standard_model, val_compound_loader, device)

    overhead_ms = lat_sg - lat_base
    overhead_pct = ((lat_sg - lat_base) / lat_base) * 100.0

    print(f"Baseline (Backbone Only)    : Latency = {lat_base:.3f} ms/sample | Throughput = {fps_base:.1f} FPS")
    print(f"ShiftGuard (Adapters+Router): Latency = {lat_sg:.3f} ms/sample | Throughput = {fps_sg:.1f} FPS")
    print(f"Routing & Adapter Overhead  : +{overhead_ms:.3f} ms/sample (+{overhead_pct:.2f}%)")

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: MULTI-SEED STATISTICAL RIGOR (N=3 Seeds)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(" EXPERIMENT 5: MULTI-SEED STATISTICAL RIGOR (N=3 Seeds)")
    print("-" * 80)

    seeds = [42, 100, 2026]
    acc_list, ece_list, auroc_list = [], [], []

    for seed in seeds:
        set_seed(seed)
        m = ShiftGuard(n_classes=len(ID_CLASSES)).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        train_model(m, train_loader, opt, device, epochs=epochs)
        res = evaluate(m, val_compound_loader, device, tag=f"Seed {seed}")
        acc_list.append(res["accuracy"])
        ece_list.append(res["ece"])
        auroc_list.append(res["ood_auroc"])

    print("\n" + "=" * 80)
    print(" FINAL STATISTICAL SUMMARY (Mean ± Std across 3 Seeds)")
    print("=" * 80)
    print(f"Compound Accuracy : {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
    print(f"Expected Calib Err: {np.mean(ece_list):.4f} ± {np.std(ece_list):.4f}")
    print(f"OOD-AUROC         : {np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiments()
