# Baseline Comparison — Microservices Anomaly Detection

A rigorous IEEE-standard benchmark that compares **8 anomaly detection baselines** against the proposed **BiLSTM + Attention** method on a synthetic but realistic microservices observability dataset. All methods share identical train/val/test splits and are evaluated with the same metrics, threshold selection strategy, and statistical significance test.

---

## Baselines Implemented

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 1 | **Z-Score (3σ)** | Statistical | Shewhart (1931) |
| 2 | **CUSUM** | Statistical | Page (1954) |
| 3 | **Isolation Forest** | ML (tree ensemble) | Liu et al., ICDM 2008 |
| 4 | **One-Class SVM** | ML (kernel) | Schölkopf et al., 1999 |
| 5 | **Local Outlier Factor (LOF)** | ML (density) | Breunig et al., SIGMOD 2000 |
| 6 | **MLP AutoEncoder** | Deep learning | Sakurada & Yairi, MLSDA 2014 |
| 7 | **LSTM AutoEncoder** | Deep learning (seq2seq) | Malhotra et al., ICML-W 2016 |
| 8 | **Transformer AutoEncoder** | Deep learning (attention) | Xu et al., ICLR 2022 |
| 9 | **Proposed: BiLSTM + Attention** | Deep learning (proposed) | This paper |

---

## Evaluation Protocol

- Identical 60/20/20 train/val/test splits for every method
- Metrics: **Precision, Recall, F1, ROC-AUC, PR-AUC**
- Threshold selection: F1-optimal over `[0.05, 0.95]` on test scores
- Statistical significance: **Wilcoxon signed-rank test** (one-sided, α = 0.05), proposed vs. each baseline
- Results reported as **Mean ± Std** over `N_RUNS` independent trials
- LaTeX-ready result table printed to stdout
- Publication-quality 5-panel comparison figure saved to `OUTPUT_DIR`

---

## Architecture Overview

```
MicroservicesDataGenerator
    │  50 services × 6 metrics × 5000 timestamps
    │  5-tier DAG causal graph (frontend → api → business → data → database)
    │  75 injected anomaly incidents + 8 distributional change points
    ▼
prepare_sequences  (window=60, stride=15)
    │
    ├── Unsupervised baselines (train on normal windows only)
    │       Z-Score · CUSUM · IsolationForest · OneClassSVM · LOF
    │       MLP-AE · LSTM-AE · Transformer-AE
    │
    └── Proposed (supervised)
            BiLSTM encoder (6-layer, bidirectional, hidden=128)
            → Temporal attention
            → LSTM decoder (reconstruction loss)
            → MLP classifier (classification loss, λ=0.3)
            score = 0.7 × classifier_score + 0.3 × reconstruction_error
    ▼
compute_metrics → significance_test → plot_comparison / print_latex_table
```

---

## Method Details

### Statistical Baselines
**Z-Score** computes per-feature z-scores over training windows and flags a window anomalous when any feature's worst-case z-score exceeds a fixed threshold (default 3σ). Scores are min-max normalised for fair threshold-agnostic evaluation.

**CUSUM** runs a per-feature cumulative-sum control chart across each window's time steps, accumulating deviation from the training mean. The maximum CUSUM statistic over features and time is used as the anomaly score.

### Classical ML Baselines
**Isolation Forest**, **One-Class SVM**, and **LOF** each receive flattened, StandardScaler-normalised windows. OC-SVM and LOF additionally apply PCA (20 components) to manage the high dimensionality of `window × metrics` feature vectors. All three are fitted only on windows with label 0 (normal), matching their one-class assumption.

### Deep Learning Baselines
All three autoencoder baselines (MLP-AE, LSTM-AE, Transformer-AE) are trained unsupervised on normal windows only and score test windows by their reconstruction MSE.

**MLP AutoEncoder** — symmetric encoder/decoder with layers `[256, 128, 64]`, latent dim 64, trained with denoising (Gaussian noise, σ=0.01), cosine LR schedule, and early stopping (patience 10).

**LSTM AutoEncoder** — seq2seq architecture: 3-layer LSTM encoder compresses the sequence to a context vector, 3-layer LSTM decoder reconstructs the original sequence from it.

**Transformer AutoEncoder** — encoder-decoder transformer with learnable positional encodings, pre-norm (`norm_first=True`), trained with AdamW and cosine scheduling. Simplified variant of Anomaly Transformer without the association discrepancy loss (reconstruction-only baseline for fair comparison).

### Proposed Method
**BiLSTM + Attention** — combines supervised and unsupervised signals:
- 6-layer bidirectional LSTM encoder with temporal attention pooling
- LSTM decoder for sequence reconstruction
- MLP classifier head with GELU activations
- Joint loss: `L = L_reconstruction + 0.3 × L_classification`
- Class imbalance handled with inverse-frequency weighting (capped at 10×)
- Final score: `0.7 × norm(classifier) + 0.3 × norm(reconstruction_error)`

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

Install:

```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn networkx
```

### Optional 

The notebook first attempts:

```python
from microservices_rca import (
    Config, MicroservicesDataGenerator, BiLSTMAnomalyDetector,
    prepare_sequences, train_model
)
```

If the module is absent, **complete self-contained fallbacks** are used automatically and the notebook runs end-to-end without any external dependency.

---

## Usage

### Run the full comparison (default)

```python
# Inside the notebook
main()
```

### Quick smoke-test (5 epochs, 1 run)

```python
# Modify args inside main() or set directly:
args = argparse.Namespace(quick=True, runs=1)
```

### Adjust number of runs

```python
args = argparse.Namespace(quick=False, runs=5)
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quick` | bool | False | 5 epochs, forces 1 run |
| `--runs` | int | 1 | Independent trials per method |

---

## Outputs

All artefacts are saved to `OUTPUT_DIR` (default `/content/user-data/outputs`):

| File | Description |
|------|-------------|
| `baseline_comparison.png` | 5-panel IEEE-style figure (300 DPI) |
| LaTeX table | Printed to stdout — paste directly into paper |

### Figure panels

| Panel | Content |
|-------|---------|
| (a) | Grouped bar chart — F1 score per method, with significance markers (`*`) |
| (b) | Horizontal bar chart — ROC-AUC per method |
| (c) | Radar / spider chart — all 5 metrics for a representative subset of methods |
| (d) | F1 gain of proposed over each baseline (green = positive, red = negative) |
| (e) | Wall-clock training + inference time (log scale) |

---

## Configuration Reference

```python
class Config:
    N_SERVICES    = 50      # Simulated microservices
    N_METRICS     = 6       # cpu, memory, latency, throughput, errors, connections
    N_TIMESTAMPS  = 5000    # Total time steps
    RANDOM_SEED   = 42
    HIDDEN_DIM    = 128     # BiLSTM hidden units
    BATCH_SIZE    = 64
    LEARNING_RATE = 1e-3
    MAX_EPOCHS    = 30
    EARLY_STOP_PATIENCE = 7
    WINDOW_SIZE   = 60
    STRIDE        = 15
    FIGURE_DPI    = 300
    OUTPUT_DIR    = Path("/content/user-data/outputs")
```

---

## Data Generation

`MicroservicesDataGenerator` produces a realistic synthetic dataset:

- **Baseline signal** — multi-frequency sinusoidal pattern (daily, weekly, 1-hour periods) with Gaussian noise
- **Anomaly injection** — 75 failure incidents, each starting at a random root service and propagating downstream through the causal DAG with configurable delay and magnitude decay
- **Change points** — 8 distributional shifts (mean shift, variance change, or both) affecting 15–30 services simultaneously
- **Normalisation** — global z-score normalisation applied after injection

The `causal_graph` (a `networkx.DiGraph`) encodes ground-truth service dependencies and is also used for evaluating causal discovery in the companion ablation study notebook.

---

## Reproducibility

Seeds are set per run:

```python
np.random.seed(seed)
torch.manual_seed(seed)
```

With the default `N_RUNS = 1`, results are fully deterministic. Set `--runs 5` or higher to obtain variance estimates and meaningful Wilcoxon test results.

---

## Relationship to Other Notebooks

| Notebook | Purpose |
|----------|---------|
| `Baseline_Comparison_RCA.ipynb` | **This file** — compares proposed method against 8 baselines |
| `Ablation_Study_Microservices_RCA.ipynb` | Ablates 6 design dimensions of the proposed BiLSTM model |
| `microservices_rca.py` (optional) | Shared module imported by both notebooks (present in Complete-model/microservices_rca.py) |

