# Ablation Study — Microservices Root Cause Analysis

The notebook systematically evaluates six independent design dimensions of the proposed architecture, generates publication-quality figures, LaTeX tables, and statistical significance reports.

---

## Overview

Modern microservice systems can span dozens of interconnected services. When anomalies occur, pinpointing the root cause requires both accurate anomaly detection and principled causal graph discovery. This notebook ablates every major architectural decision of the proposed pipeline so that each component's contribution can be independently quantified.

**Proposed configuration** (baseline everything is compared against):

| Dimension | Proposed value |
|-----------|---------------|
| Architecture | Full BiLSTM + attention + reconstruction + classification |
| Loss weight λ | 0.3 |
| Sequence window | 60 time steps (stride 15) |
| Hidden dimension | 128 |
| Causal pipeline | All 5 stages including mutual information |
| CPD sensitivity | Adaptive threshold (TM=0.8, MA=8) |

---

## Ablation Dimensions

### A — Model Architecture Components
Removes one component at a time from the full BiLSTM model:

| Variant | Description |
|---------|-------------|
| A1 (Proposed) | Full model: attention + bidirectional + 4 layers + reconstruction + classification |
| A2 | Without temporal attention |
| A3 | Unidirectional LSTM only |
| A4 | Without reconstruction loss (λ_cls = 0) |
| A5 | Without classification head |
| A6 | Single LSTM layer |
| A7 | Without dropout regularisation |

### B — Loss Function Weighting (λ)
Sweeps the classification loss weight from 0.1 to 1.0 while keeping architecture fixed.

### C — Sequence Window Size
Tests five window/stride pairs (w=30 to w=120) to study temporal context vs. dataset size trade-offs.

### D — Hidden Dimension
Tests five hidden sizes (32, 64, 128, 192, 256) to evaluate model capacity vs. overfitting.

### E — Causal Discovery Pipeline Stages
Incrementally adds pipeline stages to the causal graph discovery module:

1. Anomaly propagation only  
2. + Service tier filtering  
3. + Propagation asymmetry test  
4. + Granger causality  
5. + Mutual information (proposed)

### F — Change Point Detector (CPD) Sensitivity
Compares fixed mean+std thresholding variants against the proposed **adaptive percentile-based** thresholding with tier-coherence weighting.

### S — Sensitivity Heatmap (λ × Hidden Dimension)
A 5×5 interaction grid to reveal joint sensitivity and confirm the proposed (λ=0.3, h=128) combination.

---

## Architecture

```
MicroservicesDataGenerator
    │  50 services × 6 metrics × 5000 timestamps
    │  5-tier DAG: frontend → api → business → data → database
    │  75 injected anomaly incidents + 8 change points
    ▼
prepare_sequences  (sliding window)
    ▼
BiLSTMAnomalyDetector
    ├── BiLSTM encoder  (configurable layers, hidden dim, directionality)
    ├── Temporal attention
    ├── LSTM decoder  → reconstruction loss
    └── MLP classifier → classification loss  (weighted by λ)
    ▼
score = 0.7 × classifier_score + 0.3 × reconstruction_error
    ▼
Evaluation: Precision / Recall / F1 / ROC-AUC / PR-AUC
    ▼
MultiServiceChangePointDetector  (CUSUM, adaptive thresholding)
    ▼
AnomalyPropagationCausalDiscovery  (incremental pipeline)
```

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
scipy
scikit-learn
matplotlib
seaborn
networkx
```

Install dependencies:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn networkx
```

### Optional 

The notebook attempts to import from `microservices_rca`:

```python
from microservices_rca import (
    Config, MicroservicesDataGenerator, BiLSTMAnomalyDetector,
    prepare_sequences, train_model,
    MultiServiceChangePointDetector, AnomalyPropagationCausalDiscovery,
)
```

If the module is not present, **fully self-contained fallbacks** are used automatically — the notebook runs end-to-end without any external module improt.

---

## Usage

### Run all dimensions (default)

```python
# Inside the notebook — runs all 6 ablation dimensions + sensitivity heatmap
main()
```

### Quick smoke-test (5 epochs, reduced data)

Pass `--quick` to the argument parser inside `main()`, or set directly:

```python
import argparse, copy
args = argparse.Namespace(quick=True, runs=1, dims=list("ABCDEF") + ["S"])
```

### Run a subset of dimensions

```python
args = argparse.Namespace(quick=False, runs=3, dims=["A", "B", "S"])
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quick` | bool | False | 5 epochs, 2000 timestamps |
| `--runs` | int | 1 | Repeated runs per variant (for std estimates) |
| `--dims` | list | A B C D E F S | Dimensions to evaluate |

---

## Outputs

All artefacts are saved to `OUTPUT_DIR` (default: `/content/user-data/outputs`):

| File | Description |
|------|-------------|
| `ablation_study.png` | Multi-panel bar chart, one subplot per dimension |
| `sensitivity_heatmap.png` | 2-D F1 heatmap (λ × hidden dim) |
| `ablation_architecture.tex` | LaTeX table for Dimension A |
| `ablation_lambda.tex` | LaTeX table for Dimension B |
| `ablation_window.tex` | LaTeX table for Dimension C |
| `ablation_hidden.tex` | LaTeX table for Dimension D |
| `ablation_causal.tex` | LaTeX table for Dimension E |
| `ablation_cpd.tex` | LaTeX table for Dimension F |

Figures are rendered at 300 DPI and suitable for direct inclusion in IEEE-style papers.

---

## Evaluation Metrics

**Anomaly detection (Dimensions A–D)**

- Precision, Recall, F1 — threshold selected by maximising F1 over `[0.05, 0.95]`
- ROC-AUC and PR-AUC — threshold-free ranking metrics

**Causal discovery (Dimension E)**

- CD-Precision, CD-Recall, CD-F1 over the discovered vs. ground-truth edge set

**Change point detection (Dimension F)**

- Precision, Recall, F1 — a detection is correct if within ±150 timesteps of a true change point

**Statistical significance**

Wilcoxon signed-rank test (one-sided, α = 0.05) comparing each variant against the proposed configuration. Results are marked with `*` and a direction arrow (↑ / ↓ / =) in console tables and LaTeX output.

---

## Key Design Decisions

**Adaptive CPD thresholding (F4 — proposed)**  
Instead of a fixed `mean + k·std` threshold, F4 blends the 75th and 90th percentiles of non-trivial scores and applies tier-coherence weighting: change points affecting only 1–2 service tiers receive a bonus, while diffuse changes spanning many tiers are downweighted as likely noise. This consistently outperforms the fixed-threshold baseline.

**Composite anomaly score**  
The final score combines the BiLSTM classifier output and the reconstruction error:
```
score = 0.7 × normalize(classifier_score) + 0.3 × normalize(recon_error)
```
This makes the detector robust when one signal is uninformative.

**Class-imbalance handling**  
The binary classification loss is weighted by the inverse positive-class frequency (clamped at 10×) to compensate for the rarity of anomaly windows.

---

## Configuration Reference

```python
class Config:
    N_SERVICES    = 50      # Number of microservices
    N_METRICS     = 6       # Metrics per service
    N_TIMESTAMPS  = 5000    # Simulation length
    RANDOM_SEED   = 42
    HIDDEN_DIM    = 128     # BiLSTM hidden units
    BATCH_SIZE    = 64
    LEARNING_RATE = 1e-3
    MAX_EPOCHS    = 30
    EARLY_STOP_PATIENCE = 7
    WINDOW_SIZE   = 60      # Sequence window
    STRIDE        = 15
    FIGURE_DPI    = 300
    OUTPUT_DIR    = Path("/content/user-data/outputs")
```

---

## Reproducibility

Seeds are set deterministically per run:

```python
np.random.seed(seed)
torch.manual_seed(seed)
```

With `N_RUNS = 1` (default), results are fully deterministic. Use `--runs 3` or higher for variance estimates and meaningful Wilcoxon tests.

