# Microservices Anomaly Detection and Root Cause Analysis

---

## Overview

A four-stage unified framework for microservices observability:

1. **BiLSTM + Multi-Head Attention** — Joint reconstruction + focal-loss anomaly classification  
2. **Propagation-Based Causal Discovery** — Five-stage pipeline (propagation mining → tier filter → asymmetry test → Granger causality → mutual information)  
3. **Multi-Detector Change Point Fusion** — CUSUM (A) + Energy-Ratio (B) + KL-Divergence (C) fused at 0.45 / 0.35 / 0.20  
4. **Graph-Based Root Cause Localizer** — Personalized PageRank + ancestor scoring + tier priors + anomaly magnitude  

## Results (50-service synthetic benchmark)

| Component | Precision | Recall | F1 |
|---|---|---|---|
| Anomaly Detection | 0.92 | 0.83 | 0.82 |
| Change Point Detection | 0.78 | 0.87 | 0.82 |
| Root Cause (Hit@3) | — | — | 1.00 |

## Repository Structure

```
microservices_rca/
├── microservices_rca.py        # Full self-contained pipeline (Colab / local CPU)
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── model.py
│   ├── train.py
│   ├── causal_discovery.py
│   ├── change_point.py
│   ├── root_cause.py
│   └── visualizer.py
├── requirements.txt
└── README.md
```

## Quick Start

### Google Colab (CPU)

1. Upload `microservices_rca.py` to your Colab session  
2. Install dependencies:
   ```
   !pip install torch numpy scipy scikit-learn networkx matplotlib seaborn statsmodels
   ```
3. Run:
   ```
   !python microservices_rca.py
   ```
   Or open as a notebook via `File → Upload notebook` and paste the `.py` contents into cells.

### Local

```bash
pip install -r requirements.txt
python microservices_rca.py
```

Outputs (PNG figures) are saved to `./outputs/`.

## Hardware

Tested on Google Colab CPU (12.7 GB RAM). No GPU required.

## Software Stack

- Python 3.10, PyTorch 2.0, scikit-learn 1.3, NetworkX 3.1
- statsmodels 0.14, scipy 1.11, matplotlib 3.7

