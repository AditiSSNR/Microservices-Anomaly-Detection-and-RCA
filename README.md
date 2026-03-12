# Microservices Anomaly Detection & Root Cause Analysis

A four-stage deep learning framework that automatically detects anomalies, discovers service dependencies, identifies coordinated regime changes, and pinpoints root causes — all from raw time-series metrics, no manual dependency maps required.

---

## Results at a Glance

| Stage | Metric | Score |
|---|---|---|
| Anomaly Detection | F1 / ROC-AUC | **0.82 / 0.90** |
| Change Point Detection | F1 / Recall | **0.82 / 0.87** |
| Root Cause Analysis | Hit@1 / Hit@3 / Hit@5 | **1.00 / 1.00 / 1.00** |

Evaluated on a 50-service synthetic benchmark with 75 cascade failure scenarios and 8 coordinated change points. Outperforms 8 baselines (Z-Score, CUSUM, Isolation Forest, One-Class SVM, LOF, MLP-AE, LSTM-AE, Transformer-AE); Wilcoxon signed-rank tests confirm statistical significance.

---

## How It Works

```
Raw Metrics (50 services × 6 metrics × 5,000 timestamps)
         │
         ▼
┌─────────────────────────┐
│  1. Anomaly Detection   │  4-layer BiLSTM + 4-head attention
│     model.py / train.py │  Joint reconstruction + focal-loss classification
└────────────┬────────────┘
             │  score matrix  (timestamps × services)
     ┌───────┴────────────────────────┐
     │                                │
     ▼                                ▼
┌──────────────────────┐   ┌───────────────────────────┐
│  2. Causal Discovery │   │  3. Change Point Detection│
│  causal_discovery.py │   │  change_point.py          │
│  5-stage pipeline    │   │  3-detector fusion        │
└──────────┬───────────┘   └───────────────────────────┘
           │ dependency graph + change-point timeline
           ▼
┌──────────────────────┐
│  4. Root Cause RCA   │  PageRank + ancestor scoring
│  root_cause.py       │  + tier prior + anomaly magnitude
└──────────────────────┘
           │
    Ranked candidate services  →  Hit@1 / Hit@3 / Hit@5
```

---

## Repository Structure

```
├──complete-model
│  ├── microservices_rca.py  # Full self-contained pipeline (Colab / local CPU)
|  ├── requirements.txt
|  ├── README.md
│  └── src/
|      ├── README.md
│      ├── config.py            # All hyperparameters — single source of truth
│      ├── data_generator.py    # Synthetic 50-service dataset with injected failures
│      ├── model.py             # BiLSTMAnomalyDetector + focal loss
│      ├── train.py             # Sequence prep, training loop, score matrix
│      ├── causal_discovery.py  # AnomalyPropagationCausalDiscovery
│      ├── change_point.py      # MultiServiceChangePointDetector
│      ├── root_cause.py        # GraphBasedRootCauseAnalyzer
│      ├── evaluate.py          # End-to-end evaluation across all 4 stages
│      └── visualizer.py        # Three publication-quality figures
│
├── Ablation-Studies
|   ├── README.md
|   └── Ablation_Study_Microservices_RCA.ipynb    # Ablation experiments
├── Baseline-Comparison
|   ├── README.md
|   └── Baseline_Comparison_RCA.ipynb             # Baseline benchmarks
└── README.md                            
```

---

## Installation

```bash
git clone https://github.com/AditiSSNR/Microservices-Anomaly-Detection-and-RCA
cd Microservices-Anomaly-Detection-and-RCA
pip install torch numpy scipy scikit-learn networkx matplotlib

# Optional — enables Granger causality stage in causal discovery
pip install statsmodels
```

Tested on Python 3.10, PyTorch 2.0, scikit-learn 1.3, NetworkX 3.1, scipy 1.11. Runs on CPU and GPU.

---

## Quick Start

```python
from src.config         import Config
from src.data_generator import MicroservicesDataGenerator
from src.model          import BiLSTMAnomalyDetector
from src.train          import prepare_sequences, train_model, build_score_matrix
from src.evaluate       import evaluate_system
from src.visualizer     import Visualizer

Config.setup()   # creates ./outputs/, sets random seed, prints device

# 1. Generate data
dataset = MicroservicesDataGenerator(seed=42).generate()

# 2. Prepare sequences and split (60 / 20 / 20, chronological)
X, y, sids, _ = prepare_sequences(dataset["data"], dataset["anomaly_labels"])
tr, vl = int(0.6 * len(X)), int(0.2 * len(X))
X_tr, y_tr     = X[:tr],        y[:tr]
X_val, y_val   = X[tr:tr+vl],   y[tr:tr+vl]
X_te, y_te     = X[tr+vl:],     y[tr+vl:]
sids_te        = sids[tr+vl:]

# 3. Train
model = train_model(BiLSTMAnomalyDetector(), X_tr, y_tr, X_val, y_val)

# 4. Evaluate all four stages
results = evaluate_system(model, X_te, y_te, sids_te, dataset, X_val, y_val)

# 5. Save figures to ./outputs/
Visualizer().plot_all(dataset, results)
```

Expected output on the default configuration:

```
[Anomaly Detection]   F1=0.82  ROC-AUC=0.90  PR-AUC=0.81
[Change Points]       F1=0.82  Recall=0.87   Detected 7/9
[Root Cause]          Hit@1=1.00  Hit@3=1.00  Hit@5=1.00
```

---

## Model Architecture

The core model (`model.py`) jointly optimises reconstruction fidelity and anomaly classification:

```
Input  x  (batch × 60 timesteps × 6 metrics)
       │
       ▼  4-layer Bidirectional LSTM  →  256-dim per timestep
       │  LayerNorm
       ▼  4-head Multi-Head Self-Attention  →  context vector c
      / \
     /   \
    ▼     ▼
LSTM      MLP Classifier
Decoder   (256 → 64 → 1, sigmoid)
L_MSE     Focal Loss  (γ=2, α=0.75)
```

**Loss:** `L = L_MSE(x̂, x) + 0.5 · FocalLoss(ŝ, y)`

**Final anomaly score (three-component ensemble):**

```
ŝ = 0.55 · norm(clf)  +  0.30 · norm(recon_err)  +  0.15 · norm(|Δclf|)
```

Where `|Δclf|` is the temporal gradient of consecutive classifier outputs — it rewards sustained multi-window anomaly bursts and penalises isolated spikes.

---

## Causal Discovery Pipeline

Five sequential stages build a directed service dependency graph with no labelled dependency data needed:

| Stage | Filter | Threshold |
|---|---|---|
| 1 | Propagation strength `P(i→j)` | > 0.35 |
| 2 | Tier-coherence (no backward / skip-2+ edges) | architectural constraint |
| 3 | Asymmetry test `P(i,j) − P(j,i)` | > 0.30 |
| 4 | Granger causality p-value *(requires statsmodels)* | < 0.03 |
| 5 | Mutual information (lagged, lags 1–5) | > 0.20 |

---

## Configuration

All hyperparameters live in `src/config.py`. Override any value before calling `Config.setup()`:

```python
from src.config import Config
Config.MAX_EPOCHS = 10          # quick experiment
Config.HIDDEN_DIM = 64          # smaller model
Config.OUTPUT_DIR = Path("./my_run")
Config.setup()
```

Key defaults:

| Group | Parameter | Default |
|---|---|---|
| Model | `HIDDEN_DIM` / `LSTM_LAYERS` / `ATTN_HEADS` | 128 / 4 / 4 |
| Training | `LEARNING_RATE` / `MAX_EPOCHS` / `EARLY_STOP` | 1e-4 / 50 / 15 |
| Sequences | `WINDOW_SIZE` / `STRIDE` | 60 / 10 |
| Loss | `FOCAL_GAMMA` / `FOCAL_ALPHA` / `LAMBDA_CLS` | 2.0 / 0.75 / 0.5 |
| Score fusion | `W_CLF` / `W_RECON` / `W_GRAD` | 0.55 / 0.30 / 0.15 |
| RCA weights | `RCA_W_PR` / `RCA_W_ANC` / `RCA_W_TIER` / `RCA_W_ANOM` | 0.30 / 0.40 / 0.20 / 0.10 |

---

## Notebooks

| Notebook | What it does |
|---|---|
| `Ablation_Study_Microservices_RCA.ipynb` | Systematically ablates 6 design dimensions (architecture, loss weight λ, window size, hidden dim, causal discovery stages, CPD sensitivity). Produces `ablation_study.png`, `sensitivity_heatmap.png`, and 6 LaTeX tables. |
| `Baseline_Comparison_RCA.ipynb` | Benchmarks 8 baselines against the proposed method on identical splits. Produces a 5-panel comparison figure and a LaTeX results table. |

---
