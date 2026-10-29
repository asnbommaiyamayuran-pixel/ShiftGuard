---
name: ShiftGuard
description: >
  ShiftGuard is a shift-aware adaptive vision system for robust perception under
  distribution shifts. It dynamically routes features through specialized LoRA
  adapters based on a multi-head shift signature (domain, modality, semantic).
---

# ShiftGuard Agent Instructions

## Project Overview

ShiftGuard is a **parameter-efficient adaptive vision architecture** that detects and adapts to three types of distribution shift in real time:

1. **Domain Shifts** — environmental lighting/weather (day, night, rain, different camera ISP)
2. **Modality Shifts** — sensor noise, dropped color channels, dead pixels, JPEG compression, defocus blur
3. **Semantic Shifts** — unseen/novel out-of-distribution (OOD) objects

The system uses a **Mixture-of-Adapters** approach with **soft gating** to selectively activate LoRA adapters based on a multi-head **shift signature** `S = [s_d, s_m, s_s]`.

---

## Project Structure

```
ShiftGuard/
├── shiftguard.py                    # Core model: all architecture, training, evaluation
├── shiftguard_pipeline.py           # Checkpoint management, inference engine, real dataset loader
├── run_custom_inference.py          # Demo: load checkpoint + run inference on custom images
├── export_onnx.py                   # Production export: TorchScript + ONNX
├── cifar10_c_benchmark.py           # Benchmark: local image corruption evaluation
├── ablation_suite.py                # Ablation experiments (routing, detectors, latency, multi-seed)
├── generate_visualizations.py       # Publication-grade charts (training curves, reliability, gating, entropy)
├── generate_pdf_report.py           # IEEE-format PDF report generator
├── ShiftGuard_Research_Report.md    # Full research paper (Markdown)
├── ShiftGuard_Research_Report.pdf   # Generated IEEE-format PDF
├── agent.md                         # This file — agent instructions
├── README.md                        # Project README with abstract, tables, architecture diagram
├── .gitignore
│
├── shiftguard_checkpoint.pth        # Trained checkpoint (3-class ID)
├── shiftguard_cifar10_binary_best.pth  # CIFAR-10 binary fine-tuned checkpoint
├── shiftguard_cifar10_exp1_lr1e4.pth  # CIFAR-10 Experiment 1 checkpoint
├── simplecnn_cifar10_binary_best.pth  # SimpleCNN baseline checkpoint
├── shiftguard_traced.pt             # TorchScript traced model for deployment
│
├── shiftguard_training_curves.png   # Training loss/accuracy curves
├── shiftguard_reliability_diagram.png # ECE reliability diagram
├── shiftguard_adapter_gating.png    # Adapter gating weight distributions
├── shiftguard_ood_entropy.png       # OOD vs ID entropy histograms
├── pdf_page_1.png through pdf_page_5.png  # PDF report page renders
│
├── data/
│   └── cifar-10-batches-py/         # CIFAR-10 dataset (downloaded/extracted)
│
├── docs/
│   └── shiftguard_architecture_diagram.png  # System architecture diagram
│
├── sample_data/                     # Procedurally generated benchmark images
│   ├── car/
│   ├── truck/
│   ├── bike/
│   └── unknown_object/
│
├── test_images/                     # Custom inference test images
│   ├── daytime_car.png
│   ├── nighttime_truck.png
│   ├── noisy_bike.png
│   └── novel_ood_object.png
│
└── results/                         # Evaluation outputs, experiments, and CIFAR-10 scripts
    ├── shiftguard.py                # (Copy) Core model
    ├── shiftguard_pipeline.py       # (Copy) Pipeline
    ├── shiftguard_checkpoint.pth    # (Copy) Checkpoint
    ├── shiftguard_traced.pt         # (Copy) Traced model
    ├── shiftguard_*.png             # (Copies) Visualizations
    ├── cifar10_binary_evaluation.json  # Binary evaluation metrics
    ├── cifar10_binary_evaluation.txt   # Human-readable binary evaluation
    ├── cifar10_exp1_evaluation.json    # Experiment 1 metrics
    ├── cifar10_exp1_evaluation.txt     # Experiment 1 evaluation report
    ├── cifar10_model_comparison.json   # ShiftGuard vs SimpleCNN comparison
    ├── finetune_cifar10.py             # CIFAR-10 binary fine-tuning pipeline
    ├── evaluate_cifar10_binary.py      # CIFAR-10 binary evaluation script
    ├── evaluate_cifar10_existing_checkpoint.py  # Evaluate on CIFAR-10 test set
    ├── export_onnx.py, generate_visualizations.py, run_custom_inference.py  # (Copies)
    ├── generate_pdf_report.py          # PDF report generator (copy)
    ├── ablation_suite.py               # Ablation suite (copy)
    ├── cifar10_c_benchmark.py          # Corruption benchmark (copy)
    ├── ShiftGuard_Research_Report.md   # Research report (copy)
    ├── ShiftGuard_Research_Report.pdf  # Generated PDF (copy)
    └── figures/                        # Generated evaluation figures
        ├── shiftguard_confusion_matrix.png
        ├── shiftguard_sample_predictions.png
        ├── baseline_training_curves.png
        ├── exp1_confusion_matrix.png
        ├── exp1_sample_predictions.png
        └── shiftguard_cifar10_exp1_lr1e4_training_curves.png
```

---

## Core Architecture (`shiftguard.py`)

### Data Flow (End-to-End Pipeline)

```
Input Image → MiniViT Backbone → ShiftAnalyzer → Shift Signature S = [s_d, s_m, s_s]
                                                       ↓
                                                Router (Gating MLP)
                                                       ↓
                                            Soft Gating Weights w = [w_d, w_m, w_s]
                                                       ↓
Backbone Feature f → DomainAdapter(f) ─┐
                   → ModalityAdapter(f) ┤─── Weighted Fusion → TaskPredictor → Output
                   → SemanticAdapter(f) ┘
```

### Key Components

| Module              | Class              | Purpose                                                                                  |
| ------------------- | ------------------ | ---------------------------------------------------------------------------------------- |
| **Vision Backbone** | `MiniViT`          | Compact Vision Transformer (patch_size=8, dim=192, depth=4, heads=4). 64×64 RGB input.   |
| **Shift Analyzer**  | `ShiftAnalyzer`    | 3 parallel heads producing shift signature `S = [s_d, s_m, s_s]` from backbone features. |
| **Router**          | `Router`           | Small MLP (3→16→16→3) mapping shift signature to softmax gating weights.                 |
| **Adapters**        | `AdaptiveAdapters` | 3 LoRA adapters (domain, modality, semantic) with rank r (default 8).                    |
| **Task Predictor**  | `TaskPredictor`    | Classification head: Linear(192→128→n_classes) with ReLU + Dropout.                      |
| **Decision Engine** | `decide_output()`  | Dual-threshold: TRUSTED if confidence ≥ 0.6 AND entropy ≤ 0.5, else UNCERTAIN_OOD.       |

### Shift Signature

- **s_d** (Domain): `1.0 - softmax(domain_logits)[:, 0]` — how far from "day" (nominal) domain
- **s_m** (Modality): `sigmoid(modality_head(f))` — probability of sensor corruption
- **s_s** (Semantic): `sigmoid(semantic_head(f))` — probability of OOD/novel object

### Adaptive Feature Fusion

```
f_fused = Σ ŵ_i · (f + (α/r) · B_i · A_i · f) + (1 - min(1, Σ ŵ_i)) · f
```

Where `ŵ_i = w_i · I(w_i > τ)` with default threshold τ=0.1 (soft gating with thresholding).

### Training

Joint multi-task loss:
```
L = L_task + 0.3·L_domain + 0.3·L_modality + 0.3·L_semantic
```
- `L_task`: Cross-entropy on in-distribution (ID) samples only (OOD samples excluded)
- `L_domain`: Cross-entropy over 4 domain types (day, night, rain, diffcam)
- `L_modality`: Binary cross-entropy for modality corruption flag
- `L_semantic`: Binary cross-entropy for semantic novelty flag

Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)

### Synthetic Data (`SyntheticShiftDataset`)

100% self-contained, procedurally generated road scenes (no downloads):
- 64×64 RGB images with sky + road + geometric object proxies
- ID classes: `["car", "truck", "bike"]`
- OOD classes: `["pedestrian", "unknown_object"]`
- Domain types: `["day", "night", "rain", "diffcam"]`
- 25% probability of modality corruption (dropped channel, sensor noise, dead pixels)
- Optional semantic novelty (OOD classes)

---

## Pipeline & Inference (`shiftguard_pipeline.py`)

### Checkpoint Management
- `save_checkpoint()` — saves model state_dict + optimizer + epoch + accuracy metadata
- `load_checkpoint()` — loads and restores model, returns (model, epoch, loss, acc)

### Inference Engine
- `predict_image()` — end-to-end inference on file path, PIL Image, or Tensor
- Returns: predicted class, confidence, entropy, decision (TRUSTED/UNCERTAIN_OOD), shift signature, adapter weights

### Real Dataset Loader
- `RealImageFolderDataset` — loads images from `root/class_name/` structure
- Applies standard transforms (resize 64×64, normalize [-1, 1])
- Optional modality corruption injection

---

## Key Scripts & How to Run

### 1. Train & Evaluate Full Model
```bash
python shiftguard.py --epochs 6 --batch-size 64 --rank 8
```
Trains ShiftGuard + NoAdapterBaseline on synthetic data. Outputs accuracy, ECE, OOD-AUROC.

### 2. Pipeline Demo (Checkpoint + Inference)
```bash
python shiftguard_pipeline.py
```
Trains a quick model, saves checkpoint, loads it back, runs single-sample inference.

### 3. Custom Image Inference
```bash
python run_custom_inference.py
```
Generates test images (daytime car, nighttime truck, noisy bike, novel OOD object), loads checkpoint, runs inference on each.

### 4. CIFAR-10 Binary Fine-Tuning
```bash
cd results
python finetune_cifar10.py --epochs-stage1 8 --epochs-stage2 20
```
Fine-tunes ShiftGuard for automobile vs. truck binary classification on CIFAR-10.

### 5. CIFAR-10 Binary Evaluation
```bash
cd results
python evaluate_cifar10_binary.py
```
Evaluates the best binary checkpoint on official CIFAR-10 test set.

### 6. CIFAR-10 Existing Checkpoint Evaluation
```bash
cd results
python evaluate_cifar10_existing_checkpoint.py
```
Evaluates the synthetic-trained 3-class ShiftGuard on CIFAR-10 automobile/truck test data.

### 7. Local Corruption Benchmark
```bash
python cifar10_c_benchmark.py
```
Benchmarks against Gaussian noise, defocus blur, brightness shift, JPEG compression.

### 8. Ablation Suite
```bash
python ablation_suite.py
```
Runs 5 experiments: routing mechanisms, shift detector heads, adapter statistics, latency benchmark, multi-seed statistics.

### 9. Generate Visualizations
```bash
python generate_visualizations.py
```
Produces: training curves, reliability diagram, adapter gating bar chart, OOD entropy histogram.

### 10. Export Production Models
```bash
python export_onnx.py
```
Exports TorchScript (`shiftguard_traced.pt`) and ONNX (`shiftguard.onnx`) for deployment.

### 11. Generate PDF Report
```bash
python generate_pdf_report.py
```
Generates IEEE double-column academic PDF report.

---

## Ablation Variants (`ablation_suite.py`)

| Variant                    | Description                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `ShiftGuardHardRouting`    | Hard top-1 argmax gating instead of soft weighted gating                                |
| `ShiftGuardUniformRouting` | Fixed equal weights (1/3 each), no learned routing                                      |
| `ShiftGuardMaskedAnalyzer` | Mask out individual shift detector channels (domain-only, modality-only, semantic-only) |

### Experiments:
1. **Routing Mechanism**: Soft Gating vs Hard Top-1 vs Uniform
2. **Shift Detector Heads**: All vs Domain-only vs Modality-only vs Semantic-only
3. **Adapter Activation Statistics**: Mean weights per adapter across ID and compound splits
4. **Latency Benchmark**: Backbone-only vs full ShiftGuard (ms/sample, FPS, overhead %)
5. **Multi-Seed Statistical Rigor**: Mean ± Std across 3 seeds (42, 100, 2026)

---

## CIFAR-10 Experiments

### Binary Classification (automobile vs truck)
- **Model**: `ShiftGuardBinary` — reuses MiniViT backbone + ShiftAnalyzer + Router + Adapters from 3-class checkpoint, replaces task head with 2-class head
- **Training**: Staged (Stage 1: head-only frozen, Stage 2: full fine-tuning)
- **Checkpoints**: `shiftguard_cifar10_binary_best.pth`, `shiftguard_cifar10_exp1_lr1e4.pth`
- **Baseline**: `SimpleCNN` trained from scratch
- **Evaluation**: accuracy, precision/recall/F1, confusion matrix

---

## Visualization Outputs (`generate_visualizations.py`)

| Output File                          | Description                                                     |
| ------------------------------------ | --------------------------------------------------------------- |
| `shiftguard_training_curves.png`     | Side-by-side loss trajectory and accuracy curve                 |
| `shiftguard_reliability_diagram.png` | ECE reliability diagram comparing ShiftGuard vs Baseline        |
| `shiftguard_adapter_gating.png`      | Bar chart of mean ± std adapter gating weights                  |
| `shiftguard_ood_entropy.png`         | Overlaid histograms of predictive entropy for ID vs OOD samples |

---

## Production Deployment (`export_onnx.py`)

- **TorchScript**: `shiftguard_traced.pt` — for C++ runtime, mobile (iOS/Android), PyTorch server
- **ONNX**: `shiftguard.onnx` — for cross-platform inference (requires onnxscript)
- **Wrapper**: `ShiftGuardWrapper` returns flat tensors (logits, probs, pred, confidence, entropy, shift_signature, adapter_weights)
- **Input**: `[Batch_Size, 3, 64, 64]` normalized to [-1, 1]

---

## Performance Benchmarks (from README)

| Metric                   | Value                       |
| ------------------------ | --------------------------- |
| In-Distribution Accuracy | 95.13%                      |
| Compound Shift Accuracy  | 93.50%                      |
| Parameter Overhead       | 0.96% (LoRA rank r=16)      |
| Latency Overhead         | +0.022 ms / +0.72%          |
| ECE (Calibration Error)  | 0.0262 (vs baseline 0.0880) |
| OOD-AUROC                | 0.6868                      |

---

## Reviewer Expectations (from `REVIEWER_CHECKLIST`)

For Q1-journal publication, reviewers expect:
1. **Positioning** against DG/DA/TTA baselines (DANN, CORAL, TENT, EATA, SHOT, T3A, CoTTA, ARM)
2. **Real benchmarks**: CIFAR-10-C/100-C, ImageNet-C, DomainNet, BDD100K, nuScenes, WILDS
3. **Full ablations**: No-adapter baseline, single generic adapter, hard vs soft routing, each shift head removed
4. **All metrics**: accuracy per shift condition, ECE + reliability diagrams, OOD-AUROC/AUPR/FPR@95TPR, adapter activation statistics, compute overhead
5. **Statistical rigor**: 3-5 seeds, mean ± std, significance tests
6. **Failure analysis**: behavior under unseen compounding shifts
7. **Reproducibility**: released code, fixed seeds, synthetic data generator
