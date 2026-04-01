# Uncertainty-Aware Skin Lesion Classification with Evidential Deep Learning

An 8-class dermoscopic image classification pipeline built on the ISIC 2019 dataset. ConvNeXt-Small and DenseNet-121 are trained with **Evidential Deep Learning (EDL)** to produce Dirichlet-based uncertainty estimates alongside predictions. The pipeline includes a full evaluation suite — selective prediction, OOD detection, fairness analysis, calibration, Grad-CAM++/LIME explainability, and external validation on HAM10000 — with rigorous comparison against cross-entropy and MC Dropout baselines.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## Overview

Skin lesion classification from dermoscopic images requires not just accuracy but **reliable uncertainty quantification** — a model should know what it doesn't know. This project compares three uncertainty paradigms:

| Method | Uncertainty Signal | Forward Passes |
|--------|-------------------|----------------|
| **Evidential Deep Learning (EDL)** | Epistemic uncertainty from Dirichlet concentration | 1 |
| Cross-Entropy + Temperature Scaling | Max softmax probability (MSP) | 1 |
| MC Dropout (30 passes) | Predictive variance / mutual information | 30 |

EDL places a Dirichlet prior over class probabilities, parameterised by evidence from the network. This yields epistemic uncertainty in a **single forward pass** — no ensembling or dropout sampling required — making it practical for clinical deployment.

### Key Contributions

- **Lesion-level data splitting** to prevent information leakage from duplicate/augmented images of the same lesion
- **Shades-of-Gray color constancy** preprocessing to reduce scanner bias
- **Comprehensive uncertainty evaluation**: selective prediction with AURC, OOD detection (CIFAR-10, noise, texture, corrupted images), calibration (ECE, Brier, reliability diagrams)
- **Fairness analysis** across sex, age, and anatomical site subgroups with Cohen's h effect sizes
- **External validation** on HAM10000 to test generalization
- **Statistical rigour**: bootstrap 95% CIs and McNemar tests for all comparisons

---

## Architecture

```
                        ┌─────────────────────┐
                        │  Dermoscopic Image   │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Shades-of-Gray     │
                        │  Color Constancy    │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Albumentations     │
                        │  + Mixup (α=0.4)    │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
   ┌──────────▼──────────┐               ┌──────────────▼──────────┐
   │   ConvNeXt-Small    │               │     DenseNet-121        │
   │   (timm, IN-22k)    │               │     (timm, ImageNet)    │
   └──────────┬──────────┘               └──────────────┬──────────┘
              │                                         │
   ┌──────────▼──────────┐               ┌──────────────▼──────────┐
   │  Evidential Head    │               │    Evidential Head      │
   │  softplus → evidence│               │    softplus → evidence  │
   └──────────┬──────────┘               └──────────────┬──────────┘
              │                                         │
              │              ┌───────────┐              │
              └──────────────► Ensemble  ◄──────────────┘
                             │ (avg evidence)
                             └─────┬─────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Dirichlet Parameters       │
                    │  α = evidence + 1           │
                    │  S = Σα                     │
                    │  probs = α / S              │
                    │  epistemic = K / S          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Prediction + Uncertainty   │
                    └─────────────────────────────┘
```

---

## Datasets

### Primary: ISIC 2019

| Class | Full Name | Description |
|-------|-----------|-------------|
| MEL   | Melanoma | Malignant skin cancer |
| NV    | Melanocytic Nevus | Common mole |
| BCC   | Basal Cell Carcinoma | Most common skin cancer |
| AK    | Actinic Keratosis | Pre-cancerous lesion |
| BKL   | Benign Keratosis | Seborrheic keratosis, solar lentigo |
| DF    | Dermatofibroma | Benign fibrous nodule |
| VASC  | Vascular Lesion | Angiomas, angiokeratomas |
| SCC   | Squamous Cell Carcinoma | Second most common skin cancer |

Source: [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)

### External Validation: HAM10000

7 overlapping classes from the HAM10000 dataset, mapped to the ISIC label space for cross-dataset generalization testing.

Source: [HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

### OOD Benchmarks

CIFAR-10, random Gaussian noise, texture patches (DTD), and corrupted ISIC images are used as out-of-distribution sources to evaluate uncertainty calibration.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (training requires ~16 GB VRAM; analysis steps are CPU-only)

### Setup

```bash
git clone https://github.com/<your-username>/isic-edl-uncertainty.git
cd isic-edl-uncertainty

pip install -r requirements.txt
```

### Dependencies

```
torch>=1.13
timm>=0.9
albumentations>=1.3
numpy
pandas
scikit-learn
scikit-image
opencv-python
matplotlib
seaborn
scipy
grad-cam
lime
shap
optuna
tqdm
```

---

## Usage

### Google Colab (recommended)

1. Upload the ISIC 2019 dataset ZIP to Google Drive.
2. Open `ISIC.ipynb` in Colab with a GPU runtime.
3. Run the blocks sequentially — each block is self-contained and saves outputs to Drive for crash recovery.

### Block Execution Order

The notebook is organized into numbered blocks that must be run in order:

| Block | Steps | GPU | Description |
|-------|-------|-----|-------------|
| 1 | Shared components | No | Defines model classes, dataset loaders, utilities |
| 2 | Selective prediction + TTA | No* | Risk-coverage curves, 5-view TTA |
| 3 | Ensemble + OOD detection | Yes | 2-model ensemble, CIFAR-10/noise/texture OOD |
| 4 | Fairness + Calibration | No | Subgroup analysis, ECE, Brier, reliability diagrams |
| 5 | Bootstrap CIs + McNemar | No | Statistical significance tests |
| 6 | Threshold tuning + Melanoma utility | Yes | Deployable thresholds from validation set |
| 7 | HAM10000 external validation | Yes | Cross-dataset generalization |
| 8 | CE baseline training | Yes | Cross-entropy comparison model (~3-4 hrs) |
| 9 | MC Dropout evaluation | Yes | 30 stochastic forward passes (~1 hr) |
| 10 | Final ablation + Grad-CAM++ | Yes | Ablation table, visual explainability |
| Mega | All paper figures | No | Regenerates every figure and table (~3 min) |

*TTA requires GPU for inference; selective prediction analysis is CPU-only.

### Crash Recovery

Every block saves results to Google Drive under `ISIC_outputs/`. If Colab disconnects, run the recovery cell to restore all models, metrics, and splits from Drive before continuing.

---

## Pipeline Steps

### Data Preparation

- **Lesion-level splitting** — Images are grouped by `lesion_id` (or MD5 hash for deduplication). The split ensures all images of the same lesion stay in the same fold, preventing leakage. Final split: 70% train / 15% val / 15% test.
- **Shades-of-Gray color constancy** (power=6) applied to every image before augmentation to reduce scanner-dependent color shifts.
- **Augmentation** — horizontal/vertical flip, random rotate 90°, color jitter, coarse dropout (cutout), plus Mixup (α=0.4) during training.

### Model Training

- **Backbone**: ConvNeXt-Small (`convnext_small.fb_in22k` from timm) and DenseNet-121, pretrained on ImageNet/IN-22k.
- **Evidential head**: `nn.Linear(in_features, 8)` → `softplus` → evidence vector. Replaces the standard classification head.
- **Loss**: Evidential loss = Bayes risk (expected cross-entropy under Dirichlet) + KL divergence regularizer (annealed over first 10 epochs).
- **Training**: Adam, lr=1e-4, 30 epochs, early stopping (patience=5), image size 224–300 depending on backbone.

### Uncertainty Quantification

| Signal | Source | Formula |
|--------|--------|---------|
| Epistemic uncertainty | EDL | `K / Σ(evidence + 1)` |
| Max softmax probability | EDL / CE | `max(α / S)` or `max(softmax(logits))` |
| Predictive variance | MC Dropout | Variance across 30 forward passes |
| Mutual information | MC Dropout | `H[E[p]] - E[H[p]]` |
| Ensemble variance | 2-model ensemble | Variance of per-model predictions |

### Evaluation Suite

- **Selective prediction**: Risk-coverage curves at 10 coverage levels + AURC
- **OOD detection**: AUROC for separating in-distribution vs. CIFAR-10, Gaussian noise, texture, and corrupted images using epistemic uncertainty
- **Calibration**: Expected Calibration Error (uniform + adaptive binning), Brier score, reliability diagrams
- **Fairness**: Per-subgroup accuracy and macro-F1 across sex, age brackets, and anatomical site; Cohen's h effect sizes
- **Statistical tests**: 2000-iteration bootstrap 95% CIs for accuracy and macro-F1; McNemar's test for pairwise method comparisons
- **Clinical utility**: Melanoma-specific sensitivity/specificity at deployable thresholds tuned on validation set
- **Explainability**: Grad-CAM++ heatmaps and LIME superpixel explanations

---

## Outputs

All outputs are saved to `outputs/` with the following structure:

```
outputs/
├── models/
│   ├── convnext_small_evidential.pt     # EDL-trained ConvNeXt-Small
│   ├── densenet121_evidential.pt        # EDL-trained DenseNet-121
│   └── convnext_small_ce.pt            # CE baseline (for comparison)
│
├── metrics/
│   ├── test_uncertainty.npz            # EDL predictions + epistemic uncertainty
│   ├── test_ce_baseline.npz           # CE baseline predictions + MSP
│   ├── test_mc_dropout.npz            # MC Dropout predictions + variance
│   ├── test_tta.npz                   # TTA predictions
│   ├── test_ensemble.npz             # Ensemble predictions
│   ├── ham10000_results.npz          # External validation results
│   ├── bootstrap_cis.json            # Bootstrap confidence intervals
│   ├── mcnemar_tests.json            # McNemar p-values
│   ├── ece_brier.json                # Calibration metrics
│   ├── fairness_summary.json         # Per-subgroup metrics
│   ├── ood_aurocs.json               # OOD detection AUROC scores
│   ├── selective_prediction.json     # Risk-coverage + AURC
│   ├── ablation_table.csv            # Full ablation results
│   └── melanoma_clinical_utility.json # Sensitivity/specificity at thresholds
│
├── figures/
│   └── paper/
│       ├── confusion_matrix_edl.png
│       ├── confusion_matrix_normalized.png
│       ├── roc_curves_ood.png
│       ├── risk_coverage_curves.png
│       ├── reliability_diagrams.png
│       ├── fairness_bars.png
│       ├── gradcam_grid.png
│       ├── lime_grid.png
│       ├── ablation_bar.png
│       └── ...
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── logs/
    └── training_log.json              # Per-epoch loss and accuracy
```

---

## Project Structure

```
isic-edl-uncertainty/
│
├── ISIC.ipynb                # Complete pipeline notebook (Colab-ready)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT license
│
└── outputs/                  # Generated at runtime (not committed)
    ├── models/
    ├── metrics/
    ├── figures/
    ├── splits/
    └── logs/
```

---

## Configuration

All hyperparameters are defined at the top of each block:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEED` | 42 | Global random seed |
| `IMG_SIZE` | 300 | Input resolution (ConvNeXt); 224 for EfficientNet-B0 |
| `BATCH` | 64 | Batch size |
| `LR` | 1e-4 | Learning rate (Adam) |
| `EPOCHS` | 30 | Maximum training epochs |
| `PATIENCE` | 5 | Early stopping patience |
| `NUM_CLASSES` | 8 | Number of skin lesion classes |
| `KL_RAMP` | 10 | Epochs over which KL annealing reaches full weight |
| `MIXUP_ALPHA` | 0.4 | Beta distribution parameter for Mixup |
| `N_TTA` | 5 | Number of augmented views for TTA |
| `N_MC` | 30 | MC Dropout forward passes |

### Using a Different Dataset

The pipeline expects a metadata CSV with columns `image`, `class_name`, `class_idx`, and `image_path`. To adapt:

1. Prepare your CSV with the same schema.
2. Place image files in a flat directory.
3. Update `IMG_DIR`, `ROOT`, and `CLASSES` in the notebook.

### Adding a New Backbone

In Block 1, the `EvidentialModel` class accepts any `timm` model name:

```python
model = EvidentialModel(
    timm_name="efficientnet_b3.ra2_in1k",  # any timm model
    num_classes=8,
    probe_size=300,
)
```

Add the model to the `MODELS` dict for automatic inclusion in ensemble and evaluation steps.

---

## Reproducibility

Seeds are set for PyTorch, NumPy, and scikit-learn. For full GPU determinism:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
```

Note: deterministic mode disables cuDNN autotuner and may reduce training speed by 10–20%.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{isic_edl_2025,
  title   = {Uncertainty-Aware Skin Lesion Classification with Evidential Deep Learning},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/<your-username>/isic-edl-uncertainty}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) organizers and data providers
- [HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161) — Tschandl et al., 2018
- [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
- [Sensoy et al., 2018](https://arxiv.org/abs/1806.01768) — Evidential Deep Learning to Quantify Classification Uncertainty
- [Albumentations](https://albumentations.ai/), [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam), [LIME](https://github.com/marcotcr/lime)
