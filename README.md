# Rethinking Efficiency: A Comparative Study of Lightweight CNN Architectures for Image Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.64878%2Fjistics.v2i1.167-blue)](https://doi.org/10.64878/jistics.v2i1.167)

> **Published in:** Journal of Intelligent Systems, Technology, and Informatics (JISTICS), Vol. 2, No. 1, pp. 13–21, March 2026

Official implementation of our benchmark paper comparing five lightweight CNN architectures on CIFAR-100 under unified training conditions.

---

## Overview

Lightweight CNNs are essential for deploying image classification in resource-constrained environments, yet fair comparisons across architectures remain scarce. This repository provides a **single, reproducible training pipeline** that benchmarks:

| Model | Params | Top-1 Acc (%) | Top-5 Acc (%) | Latency (ms) |
|:---|---:|---:|---:|---:|
| **EfficientNet-B0** | 4.14 M | **82.75** | **96.46** | 5.43 |
| MobileNetV2 | 2.35 M | 80.89 | 96.00 | 3.55 |
| ShuffleNetV2 | 1.36 M | 77.96 | 95.20 | 4.32 |
| ResNet-18 | 11.23 M | 76.90 | 94.13 | 1.70 |
| **SqueezeNet** | **0.77 M** | 65.82 | 90.00 | **1.52** |

> Results shown for **basic augmentation** at 128×128 input resolution (best config per paper).

### Key Findings

- **EfficientNet-B0** achieves the highest accuracy (82.75% Top-1) with only 4.14M parameters.
- **SqueezeNet** offers the fastest inference (1.52 ms) and smallest footprint, ideal for extreme edge deployment.
- **Advanced augmentation is not universally beneficial** — on average it decreased Top-1 accuracy by 0.66 pp; only ResNet-18 showed modest improvement.
- **Architecture design matters more than parameter count** for balancing accuracy and efficiency.

---

## Repository Structure

```
├── train_cnn_benchmark_grid.py   # Main training & evaluation script
├── cekspek.py                    # System/hardware spec checker
├── requirements.txt              # Python dependencies
├── data/                         # CIFAR-100 (auto-downloaded)
└── runs/                         # Experiment outputs
    └── YYYYMMDD_HHMMSS/
        ├── config.json           # Hyperparameters used
        ├── results_full.csv      # Per-setting results
        ├── results_agg_by_*.csv  # Aggregated tables
        ├── metrics_full.json     # Full metrics + training histories
        └── *_best.pt             # Best model checkpoints
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rizalfanex/lightweight-cnn-benchmark.git
cd lightweight-cnn-benchmark
pip install -r requirements.txt
```

### 2. Check System Requirements

```bash
python cekspek.py
```

### 3. Run the Full Benchmark

```bash
python train_cnn_benchmark_grid.py \
    --epochs 40 \
    --batch 128 \
    --lr 0.001 \
    --img-sizes 128 \
    --aug-modes basic advanced \
    --models mobilenet_v2 efficientnet_b0 shufflenet_v2_x1_0 squeezenet1_1 resnet18
```

CIFAR-100 will be downloaded automatically on the first run.

### 4. Train a Single Model (Quick Test)

```bash
python train_cnn_benchmark_grid.py \
    --epochs 5 \
    --models efficientnet_b0 \
    --img-sizes 128 \
    --aug-modes basic
```

---

## Training Configuration

| Hyperparameter | Value |
|:---|:---|
| Dataset | CIFAR-100 (100 classes) |
| Input Resolution | 128 × 128 |
| Epochs | 40 |
| Batch Size | 128 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | Cosine Annealing |
| Mixed Precision | Enabled (AMP) |
| Gradient Clipping | 1.0 |
| Pretrained Weights | ImageNet |
| Seed | 42 |

### Augmentation Strategies

| Strategy | Transforms |
|:---|:---|
| **Basic** | Resize → RandomResizedCrop (0.80–1.0) → HorizontalFlip → Normalize |
| **Advanced** | Resize → RandomResizedCrop (0.75–1.0) → HorizontalFlip → ColorJitter → RandomRotation(±10°) → Normalize |

---

## Full Results

### By Model (averaged across augmentation modes)

| Model | Top-1 (%) | Top-5 (%) | Latency (ms) | Train Time (s) |
|:---|---:|---:|---:|---:|
| EfficientNet-B0 | 82.75 | 96.46 | 5.46 | 1047 |
| MobileNetV2 | 80.42 | 95.82 | 3.55 | 811 |
| ShuffleNetV2 | 77.71 | 95.14 | 4.35 | 629 |
| ResNet-18 | 77.18 | 94.03 | 1.68 | 722 |
| SqueezeNet | 64.83 | 89.36 | 1.52 | 571 |

### Augmentation Effect

| Augmentation | Avg Top-1 (%) | Avg Top-5 (%) |
|:---|---:|---:|
| Basic | 76.91 | 94.35 |
| Advanced | 76.24 | 93.97 |
| **Δ (Adv − Basic)** | **−0.66** | **−0.38** |

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{fauzan2026rethinking,
  title   = {Rethinking Efficiency: A Comparative Study of Lightweight CNN Architectures for Image Classification},
  author  = {Fauzan, Mochamad Rizal and Iskandar, Naufal Nadhif Rabbani and Fauzi, Rafi Zahran},
  journal = {Journal of Intelligent Systems, Technology, and Informatics (JISTICS)},
  volume  = {2},
  number  = {1},
  pages   = {13--21},
  year    = {2026},
  doi     = {10.64878/jistics.v2i1.167}
}
```

---

## Authors

- **Mochamad Rizal Fauzan** — National Taipei University of Technology
- **Naufal Nadhif Rabbani Iskandar** — Universitas Pendidikan Indonesia
- **Rafi Zahran Fauzi** — Universitas Pendidikan Indonesia

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
