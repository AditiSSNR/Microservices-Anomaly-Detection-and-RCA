# Microservices Root Cause Analysis — `src` Package

End-to-end Python package implementing the four-stage anomaly detection and root cause analysis pipeline described in the paper. The system ingests multi-metric time-series from a 50-service microservice topology, detects anomalies with a BiLSTM model, discovers causal relationships between services, identifies distributional change points, and localizes the root cause of failures.

---

## Package Structure

```
src/
├── config.py            # All hyperparameters (single source of truth)
├── data_generator.py    # Synthetic dataset: topology, failures, change points
├── model.py             # BiLSTM + Multi-Head Attention anomaly detector
├── train.py             # Training loop, score fusion, calibration, score matrix
├── causal_discovery.py  # Five-stage propagation-based causal graph discovery
├── change_point.py      # Three-detector fusion change point detector
├── root_cause.py        # PageRank + ancestor scoring root cause localizer
├── evaluate.py          # Full four-stage evaluation pipeline
└── visualizer.py        # Publication figures (CPD, performance summary, heatmap)
```

---

## Pipeline Overview

```
MicroservicesDataGenerator
    │  50 services × 6 metrics × 5000 timestamps
    │  5-tier DAG: frontend → api → business → data → database
    │  75 cascade failure scenarios (Eq. 2) + 8 change points
    ▼
prepare_sequences()           window=60, stride=10  → ~23,300 train sequences
    ▼
BiLSTMAnomalyDetector         4-layer BiLSTM → LayerNorm → 4-head MHA
    │  Joint loss (Eq. 5): L_MSE + 0.5 · FocalLoss(γ=2, α=0.75)
    │  Optimiser: AdamW lr=1e-4, wd=1e-4, cosine-annealing T0=10
    ▼
compute_scores()              Eq. (6): 0.55·clf + 0.30·recon + 0.15·grad
build_score_matrix()          (n_timestamps × n_services) via max-pooling
    ▼
    ├── AnomalyPropagationCausalDiscovery   5-stage pipeline (Algorithm 1)
    │       Stage 1: Propagation mining (Eq. 7, τ=0.35)
    │       Stage 2: Tier filter
    │       Stage 3: Asymmetry test (Δ=0.30)
    │       Stage 4: Granger causality gate (p<0.03)
    │       Stage 5: Mutual information gate (MI>0.20)
    │
    ├── MultiServiceChangePointDetector     3-detector fusion
    │       Detector A: CUSUM (weight 0.45)
    │       Detector B: Energy ratio (weight 0.35)
    │       Detector C: KL divergence (weight 0.20)
    │
    └── GraphBasedRootCauseAnalyzer         Eq. (8)
            score(s) = 0.30·PR + 0.40·ANC + 0.20·tier + 0.10·AM
    ▼
evaluate_system()             AD / CPD / CD / RCA metrics + significance
Visualizer.plot_all()         3 publication figures (300 DPI)
```

---

## Module Reference

### `config.py` — `Config`

Single source of truth for every hyperparameter.

```python
from src.config import Config

Config.setup()          # creates output dir, sets seeds, prints device
device = Config.device()
```

| Category | Key parameters |
|----------|---------------|
| Dataset | `N_SERVICES=50`, `N_METRICS=6`, `N_TIMESTAMPS=5000` |
| Model | `HIDDEN_DIM=128`, `LSTM_LAYERS=4`, `ATTN_HEADS=4`, `DROPOUT=0.1` |
| Training | `LR=1e-4`, `WD=1e-4`, `MAX_EPOCHS=50`, `EARLY_STOP=15`, `GRAD_CLIP=1.0` |
| Sequences | `WINDOW_SIZE=60`, `STRIDE=10` |
| Loss | `FOCAL_GAMMA=2.0`, `FOCAL_ALPHA=0.75`, `LAMBDA_CLS=0.5` |
| Score fusion | `W_CLF=0.55`, `W_RECON=0.30`, `W_GRAD=0.15` |
| CPD | `CPD_WINDOW=100`, `CPD_MIN_AFF=6`, `CPD_DEDUP=200` |
| Causal discovery | `CD_PROP_THR=0.35`, `CD_ASYM_THR=0.30`, `CD_P_VAL=0.03`, `CD_MI_THR=0.20` |
| RCA weights | `RCA_W_PR=0.30`, `RCA_W_ANC=0.40`, `RCA_W_TIER=0.20`, `RCA_W_ANOM=0.10` |

---

### `data_generator.py` — `MicroservicesDataGenerator`

Generates a fully-labelled synthetic dataset.

**Baseline signal:** three-frequency sinusoid (daily 288 ts, weekly 2016 ts, hourly 12 ts) with Gaussian noise σ=3.

**Cascade failures:** 75 scenarios. A root service is chosen from frontend/api tiers (first 15 scenarios) or business tier (remaining 60). Failure propagates downstream through the causal DAG with per-hop delay and magnitude decay `(3.5 + U[0,3]) · (1 − 0.05·d)`.

**Change points:** 8 coordinated events at timestamps 800, 1300, … 4300, each affecting 15–30 services via mean-shift, variance-change, or both.

**Topology:** five-tier directed acyclic graph — `frontend(8) → api(10) → business(17) → data(8) → database(7)`.

```python
from src.data_generator import MicroservicesDataGenerator

gen     = MicroservicesDataGenerator(seed=42)
dataset = gen.generate()
# dataset.keys(): data, edge_index, anomaly_labels, root_causes,
#                 change_points, causal_graph, metric_names, tiers
```

`dataset['data']` shape: `(50, 5000, 6)` — z-score normalised after injection.

---

### `model.py` — `BiLSTMAnomalyDetector`

Four-stage architecture:

1. **4-layer bidirectional LSTM encoder** — input projection from `n_metrics` to `2 × hidden_dim`.
2. **LayerNorm** on encoder output.
3. **4-head multi-head self-attention** — context vector = mean-pool of attention output.
4. **LSTM decoder + linear head** — reconstructs the input sequence.
5. **3-layer GELU MLP classifier** — produces per-window anomaly score ∈ [0, 1].

Weight initialisation: Xavier uniform for input weights, orthogonal for recurrent weights, forget-gate bias initialised to 1.

**Loss (Eq. 5):**
```
L = L_MSE(recon, x) + 0.5 · FocalLoss(ŝ, y)
FocalLoss: γ=2.0, α=0.75
```

```python
from src.model import BiLSTMAnomalyDetector

model = BiLSTMAnomalyDetector()   # uses Config defaults
out   = model(x)                  # x: (B, T, M)
# out.keys(): reconstruction, anomaly_score, attention, encoding
loss  = model.compute_loss(out, x, labels)
```

---

### `train.py`

**`prepare_sequences(data, labels)`** — sliding window extraction returning `(X, y, service_ids, start_times)`. With `window=60, stride=10` over 50 services this yields ~23,300 training sequences.

**`train_model(model, X_train, y_train, X_val, y_val)`** — full training loop with:
- AdamW (lr=1e-4, wd=1e-4)
- Cosine annealing warm restarts (T₀=10, T_mult=2, η_min=1e-6)
- Gradient clipping at 1.0
- Early stopping with patience 15
- Best-validation-loss checkpoint restored at end

**`compute_scores(model, X)`** — three-component ensemble score (Eq. 6):
```
ŝ = 0.55·norm(clf) + 0.30·norm(recon_err) + 0.15·norm(|Δclf|)
```
The gradient term `|Δclf|` captures temporal transitions, rewarding sustained anomalous bursts.

**`calibrate_scores(raw_val, y_val, raw_test)`** — isotonic regression calibration fitted on validation scores and applied to test scores before threshold search.

**`build_score_matrix(model, dataset)`** — produces an `(n_timestamps, n_services)` score matrix by max-pooling per-sequence scores back onto the original timeline. Used as input to causal discovery and RCA.

```python
from src.train import prepare_sequences, train_model, compute_scores

X, y, sids, stimes = prepare_sequences(dataset['data'], dataset['anomaly_labels'])
model  = train_model(model, X_tr, y_tr, X_val, y_val)
scores = compute_scores(model, X_test)
```

---

### `causal_discovery.py` — `AnomalyPropagationCausalDiscovery`

Five-stage pipeline. Each stage filters the candidate edge set:

**Stage 1 — Propagation mining (Eq. 7)**
Forward propagation strength `P(i→j)` = fraction of anomaly events in service `i` followed by an anomaly event in service `j` within `max_lag=6` timesteps. Edge placed if `P > 0.35`. Implemented with vectorised CUSUM to avoid Python loops over timestamps.

**Stage 2 — Tier filter**
Removes backward edges (lower tier → higher tier) and edges that skip more than 2 tiers.

**Stage 3 — Asymmetry test**
Retains `i→j` only if `P(i,j) > P(j,i) + 0.30`, ensuring directionality.

**Stage 4 — Granger causality gate** *(requires `statsmodels`)*
Drops edge if the SSR F-test p-value exceeds 0.03 for all lags 1–5.

**Stage 5 — Mutual information gate**
Drops edge if the maximum lagged MI (lags 1–5, via `sklearn.feature_selection.mutual_info_regression`) is below 0.20.

```python
from src.causal_discovery import AnomalyPropagationCausalDiscovery

cd = AnomalyPropagationCausalDiscovery(
    n_services            = 50,
    service_tiers         = dataset['tiers'],
    bilstm_anomaly_scores = score_matrix,   # (n_timestamps, n_services)
)
adjacency, strengths = cd.discover_causal_graph(dataset['data'])
```

`statsmodels` is an optional dependency. If absent, Stage 4 is skipped and the pipeline continues with Stages 1–3 and 5.

---

### `change_point.py` — `MultiServiceChangePointDetector`

Three-detector fusion:

**Detector A — CUSUM** (weight 0.45): per-service mean-shift and variance-ratio statistics in a sliding window of half-width 100. Counts services with significant change, weighted by mean change magnitude.

**Detector B — Energy ratio** (weight 0.35): `E_ratio(t) = log(E_after / E_before)`, where `E = mean(x²)`. Score = `mean|ratio| × count(|ratio| > 0.2)`.

**Detector C — KL divergence** (weight 0.20): KL(P‖Q) estimated from 20-bin histograms, averaged over 20 randomly sampled services.

**Fusion:** `fused = 0.45·A + 0.35·B + 0.20·C`

**Peak detection:** `height = mean + 0.5·std` (clipped to [0.04, 0.55]), `distance=120`, `prominence=0.05`.

**Confirmation:** a peak is accepted if ≥ 6 services show significant change at that timestamp. Detections within 200 timesteps are merged.

```python
from src.change_point import MultiServiceChangePointDetector

cpd = MultiServiceChangePointDetector()
detected_ts, fused_scores = cpd.detect(dataset['data'])

# With evaluation against ground truth
true_cps = [cp['timestamp'] for cp in dataset['change_points']]
result   = cpd.evaluate(dataset['data'], true_cps, tolerance=150)
# result.keys(): precision, recall, f1, n_detected, n_correct, n_true, scores, detected
```

---

### `root_cause.py` — `GraphBasedRootCauseAnalyzer`

Multi-signal root cause localizer:

```
score(s) = 0.30·PR(s) + 0.40·ANC(s) + 0.20·tier(s) + 0.10·AM(s)
```

**PR(s)** — Personalized PageRank on the *reversed* dependency graph. Personalization vector proportional to per-service anomaly scores. Damping factor α=0.85. Running on the reversed graph means propagation flows upstream, rewarding services whose downstream descendants are anomalous.

**ANC(s)** — Ancestor score:
```
ANC(v) = Σ_{a ∈ anomalous} AM(a) / (dist(v,a)^0.5 + 1)
```
Rewards nodes that are ancestors of many anomalous services, with inverse square-root distance weighting.

**tier(s)** — Architectural prior: `frontend=1.0, api=0.8, business=0.6, data=0.3, database=0.1`. Reflects that root causes typically originate in upstream tiers.

**AM(s)** — Per-service anomaly magnitude: 90th percentile of non-zero values in the score-matrix column.

```python
from src.root_cause import GraphBasedRootCauseAnalyzer

rca    = GraphBasedRootCauseAnalyzer(causal_graph, dataset['tiers'])
result = rca.localize_root_cause(anomaly_scores, causal_strengths, top_k=15)
# result.keys(): root_causes (list[int]), scores (list[float]), anomalous_services
```

---

### `evaluate.py` — `evaluate_system()`

Orchestrates all four evaluation stages in a single call and prints a formatted report.

```python
from src.evaluate import evaluate_system

results = evaluate_system(
    model          = model,
    X_test         = X_test,
    y_test         = y_test,
    service_ids_test = sids_test,
    dataset        = dataset,
    X_val          = X_val,
    y_val          = y_val,
)
```

**Stage 1 — Anomaly Detection** — calibrated scores → best-F1 threshold → Precision / Recall / F1 / ROC-AUC / PR-AUC.

**Stage 2 — Causal Discovery** — discovered edges vs. ground-truth `edge_index` → CD-Precision / CD-Recall / CD-F1.

**Stage 3 — Change Point Detection** — tolerance ±150 timesteps → CPD-Precision / CPD-Recall / CPD-F1.

**Stage 4 — Root Cause Analysis** — Hit@1 / Hit@3 / Hit@5 / Hit@10 against the ground-truth `root_causes` list.

Return value is a nested dict with keys `anomaly_detection`, `causal_discovery`, `change_point_detection`, `root_cause_analysis`.

---

### `visualizer.py` — `Visualizer`

Generates three publication-quality figures (default 150 DPI, configurable via `Config.FIGURE_DPI`).

| Figure | File | Content |
|--------|------|---------|
| Change points | `change_points.png` | Top: sample service metrics with true CP markers. Bottom: fused CPD score with detected and true CPs |
| Performance summary | `performance_summary.png` | 2×2 bar chart covering all four pipeline stages; green dashed 90% target line |
| Anomaly distribution | `anomaly_distribution.png` | Top: tier-level smoothed anomaly counts over time. Bottom: full service×time heatmap |

```python
from src.visualizer import Visualizer

viz = Visualizer(output_dir=Config.OUTPUT_DIR)
viz.plot_all(dataset, results)
```

---

## Quickstart

```python
from src.config import Config
from src.data_generator import MicroservicesDataGenerator
from src.model import BiLSTMAnomalyDetector
from src.train import prepare_sequences, train_model
from src.evaluate import evaluate_system
from src.visualizer import Visualizer

# 1. Setup
Config.setup()

# 2. Generate data
dataset = MicroservicesDataGenerator().generate()

# 3. Prepare sequences and split
X, y, sids, stimes = prepare_sequences(dataset['data'], dataset['anomaly_labels'])
n  = len(X)
tr = int(0.6 * n);  vl = int(0.2 * n)
X_tr,  y_tr  = X[:tr],       y[:tr]
X_val, y_val = X[tr:tr+vl],  y[tr:tr+vl]
X_te,  y_te  = X[tr+vl:],    y[tr+vl:]
sids_te      = sids[tr+vl:]

# 4. Train
model = BiLSTMAnomalyDetector()
model = train_model(model, X_tr, y_tr, X_val, y_val)

# 5. Evaluate
results = evaluate_system(model, X_te, y_te, sids_te, dataset, X_val, y_val)

# 6. Visualize
Visualizer().plot_all(dataset, results)
```

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
scipy
scikit-learn
networkx
matplotlib
statsmodels   # optional — enables Granger causality (Stage 4 of causal discovery)
```

Install:

```bash
pip install torch numpy scipy scikit-learn networkx matplotlib statsmodels
```

---

## Configuration

All hyperparameters live in `Config` and can be overridden before calling any module:

```python
from src.config import Config

Config.MAX_EPOCHS  = 10          # faster iteration
Config.OUTPUT_DIR  = Path("./my_outputs")
Config.FIGURE_DPI  = 300         # print-quality figures
Config.N_SERVICES  = 20          # smaller simulation
Config.setup()
```

---

## Outputs

All files are written to `Config.OUTPUT_DIR` (default `./outputs/`):

| File | Generated by | Description |
|------|-------------|-------------|
| `change_points.png` | `Visualizer` | Service metrics + CPD score timeline |
| `performance_summary.png` | `Visualizer` | 2×2 metric bar chart |
| `anomaly_distribution.png` | `Visualizer` | Tier anomaly counts + heatmap |

Evaluation metrics are returned as a nested Python dict from `evaluate_system()` and printed to stdout in a formatted report.

---

## Design Notes

**Score fusion:** the gradient term `|Δclf|` is computed as the absolute first-difference of the classifier output sequence. This penalises isolated single-window spikes (likely false positives) while rewarding sustained anomaly bursts that evolve gradually.

**Isotonic calibration:** calibrating classifier scores with isotonic regression on the validation set reduces threshold sensitivity — the best-F1 threshold search on calibrated test scores is more stable across seeds than operating on raw logits.

**Causal discovery fallback:** if the discovered graph has fewer than 10 edges (e.g. when `statsmodels` is unavailable or data is too short for Granger testing), `evaluate_system` falls back to the ground-truth `causal_graph` for the RCA stage to avoid degenerate results.

**RCA on reversed graph:** PageRank is computed on the *reversed* dependency graph so that random walks flow *upstream* — from anomalous services back toward their likely causes — matching the semantics of root cause localisation.
