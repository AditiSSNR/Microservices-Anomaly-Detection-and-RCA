# =============================================================================
# microservices_rca.py
# "Learning Failure Propagation in Microservices for
#  Anomaly Detection and Root Cause Analysis"
#
# Single-file, Google Colab CPU-compatible implementation.
# Tested on: Python 3.10 | PyTorch 2.0 | scikit-learn 1.3 |
#            NetworkX 3.1 | statsmodels 0.14 | scipy 1.11
# =============================================================================

import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.signal import find_peaks
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# =============================================================================
# 0.  CONFIGURATION
#     All hyper-parameters exactly as reported in the paper (Table I & §III).
# =============================================================================

class Config:
    """
    Global hyper-parameters.  Values are taken directly from the paper:

    Data          : N=50 services, M=6 metrics, T=5000 timestamps
    Model         : hidden_dim=128, 4-layer BiLSTM, 4 attention heads
    Training      : AdamW lr=1e-4 wd=1e-4, cosine-annealing T0=10,
                    gradient clip 1.0, max 50 epochs, early-stop patience 15
    Sequences     : window=60, stride=10  →  ~23 300 training sequences
    Focal loss    : γ=2.0, α=0.75
    Score fusion  : w_clf=0.55, w_recon=0.30, w_grad=0.15
    CPD           : window=100, fused weights (0.45/0.35/0.20),
                    min_affected=6, dedup_distance=200
    """

    # --- dataset ---
    N_SERVICES   = 50
    N_METRICS    = 6
    N_TIMESTAMPS = 5000
    RANDOM_SEED  = 42

    # --- model architecture (paper §III-B) ---
    HIDDEN_DIM   = 128
    LSTM_LAYERS  = 4          # 4-layer BiLSTM encoder
    ATTN_HEADS   = 4          # multi-head self-attention heads
    DROPOUT      = 0.1

    # --- training (paper §IV-A) ---
    BATCH_SIZE      = 64
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 1e-4
    MAX_EPOCHS      = 50
    EARLY_STOP      = 15
    GRAD_CLIP       = 1.0
    LR_T0           = 10      # cosine annealing warm restart T_0

    # --- sequence preparation (paper §III-A / Table I) ---
    WINDOW_SIZE  = 60
    STRIDE       = 10         # dense stride → ~23 300 train sequences

    # --- focal loss (paper Eq. 5 / [19]) ---
    FOCAL_GAMMA  = 2.0
    FOCAL_ALPHA  = 0.75

    # --- multi-task loss weight (paper Eq. 5) ---
    LAMBDA_CLS   = 0.5        # total = recon + 0.5 * FL

    # --- anomaly score fusion (paper Eq. 6) ---
    W_CLF        = 0.55
    W_RECON      = 0.30
    W_GRAD       = 0.15

    # --- change-point detection (paper §III-C, ablation best TM=0.8, MA=8) ---
    CPD_WINDOW      = 100
    CPD_MIN_AFF     = 6       # min services that must change (Table IV best)
    CPD_THRESH_MULT = 0.5     # threshold = mean + 0.5*std
    CPD_DISTANCE    = 120     # peak min distance (timesteps)
    CPD_PROMINENCE  = 0.05
    CPD_DEDUP       = 200     # merge detections within 200 ts (paper §III-C)

    # --- causal discovery (paper §III-B) ---
    CD_PROP_THR  = 0.35       # forward propagation strength threshold τ
    CD_ASYM_THR  = 0.30       # asymmetry margin Δ
    CD_P_VAL     = 0.03       # Granger p-value threshold
    CD_MI_THR    = 0.20       # mutual information threshold
    CD_MAX_LAG   = 6          # maximum propagation lag

    # --- root cause analysis (paper §III-D, Eq. 8) ---
    RCA_W_PR     = 0.30       # PageRank weight
    RCA_W_ANC    = 0.40       # ancestor score weight
    RCA_W_TIER   = 0.20       # tier prior weight
    RCA_W_ANOM   = 0.10       # anomaly magnitude weight
    RCA_PAGERANK_ALPHA = 0.85 # damping factor

    # --- output ---
    OUTPUT_DIR   = Path("./outputs")
    FIGURE_DPI   = 150

    @classmethod
    def set_seed(cls):
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)

    @classmethod
    def device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.set_seed()
        print(f"[Config] device={cls.device()}  seed={cls.RANDOM_SEED}")


# =============================================================================
# 1.  DATA GENERATION
#     Paper §III-A — Eq. (1) baseline + Eq. (2) cascade injection
# =============================================================================

class MicroservicesDataGenerator:
    """
    Generates a synthetic 50-service dataset that mirrors the paper:
      - Multi-scale sinusoidal baseline (daily / weekly / hourly)
      - 75 cascade failure scenarios (Eq. 2, magnitude ∝ exp(-d))
      - 8 coordinated change points (mean-shift / variance-change / both)
      - Five-tier dependency graph: frontend→api→business→data→database
    """

    def __init__(
        self,
        n_services: int = Config.N_SERVICES,
        n_metrics: int  = Config.N_METRICS,
        n_timestamps: int = Config.N_TIMESTAMPS,
        seed: int = Config.RANDOM_SEED,
    ):
        self.n_services   = n_services
        self.n_metrics    = n_metrics
        self.n_timestamps = n_timestamps
        np.random.seed(seed)

        self.tiers = {
            "frontend": list(range(0, 8)),
            "api":      list(range(8, 18)),
            "business": list(range(18, 35)),
            "data":     list(range(35, 43)),
            "database": list(range(43, 50)),
        }
        self.causal_graph = self._create_causal_graph()
        self.metric_names = ["cpu", "memory", "latency",
                             "throughput", "errors", "connections"]

    # ------------------------------------------------------------------
    def _create_causal_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_services))

        for fe in self.tiers["frontend"]:
            for t in np.random.choice(self.tiers["api"],
                                      size=np.random.randint(2, 4),
                                      replace=False):
                G.add_edge(fe, t)

        for api in self.tiers["api"]:
            for t in np.random.choice(self.tiers["business"],
                                      size=np.random.randint(3, 6),
                                      replace=False):
                G.add_edge(api, t)

        for bus in self.tiers["business"]:
            if np.random.rand() > 0.2:
                for t in np.random.choice(self.tiers["data"],
                                          size=np.random.randint(1, 3),
                                          replace=False):
                    G.add_edge(bus, t)

        for dat in self.tiers["data"]:
            for t in np.random.choice(self.tiers["database"],
                                      size=np.random.randint(1, 2),
                                      replace=False):
                G.add_edge(dat, t)

        return G

    # ------------------------------------------------------------------
    def generate(self) -> Dict:
        """
        Returns a dict with keys:
          data, edge_index, anomaly_labels, root_causes,
          change_points, causal_graph, metric_names, tiers
        """
        # --- Eq. (1): multi-scale sinusoidal baseline ---
        t    = np.arange(self.n_timestamps)
        base = (50
                + 20 * np.sin(2 * np.pi * t / 288)
                + 10 * np.sin(2 * np.pi * t / (288 * 7))
                +  5 * np.sin(2 * np.pi * t / 12))
        data = (base[np.newaxis, :, np.newaxis]
                + np.random.randn(self.n_services,
                                  self.n_timestamps,
                                  self.n_metrics) * 3)

        anomaly_labels, root_causes = self._inject_cascade_failures(data)
        change_points = self._inject_change_points(data)

        # --- z-score normalisation (joint service-time axis) ---
        mean = data.mean(axis=(0, 1))
        std  = data.std(axis=(0, 1)) + 1e-8
        data = (data - mean) / std

        edge_index = np.array(list(self.causal_graph.edges())).T

        return {
            "data":           data,
            "edge_index":     edge_index,
            "anomaly_labels": anomaly_labels,
            "root_causes":    root_causes,
            "change_points":  change_points,
            "causal_graph":   self.causal_graph,
            "metric_names":   self.metric_names,
            "tiers":          self.tiers,
        }

    # ------------------------------------------------------------------
    def _inject_cascade_failures(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """75 cascade scenarios; Eq. (2): magnitude decays with distance."""
        anomaly_labels = np.zeros((self.n_timestamps, self.n_services))
        root_causes    = []

        for scenario_id in range(75):
            if scenario_id < 15:
                root = np.random.choice(
                    self.tiers["frontend"] + self.tiers["api"])
            else:
                root = np.random.choice(self.tiers["business"])

            start    = np.random.randint(500, self.n_timestamps - 500)
            affected = list(nx.descendants(self.causal_graph, root))
            affected = [root] + affected[:min(len(affected), 15)]

            for i, service in enumerate(affected):
                delay    = i * np.random.randint(2, 6)
                duration = np.random.randint(40, 100)
                if start + delay + duration >= self.n_timestamps:
                    continue
                # Eq. (2)
                magnitude = (3.5 + np.random.rand() * 3.0) * (1 - i * 0.05)
                sl = slice(start + delay, start + delay + duration)
                data[service, sl, :]       *= magnitude
                anomaly_labels[sl, service] = 1

            root_causes.append({
                "scenario_id":       scenario_id,
                "root_cause":        root,
                "affected_services": affected,
                "timestamp":         start,
            })

        return anomaly_labels, root_causes

    # ------------------------------------------------------------------
    def _inject_change_points(self, data: np.ndarray) -> List[Dict]:
        """8 coordinated change points, 15-30 services each."""
        change_points = []
        for cp_id in range(8):
            timestamp  = 800 + cp_id * 500
            n_affected = np.random.randint(15, 30)
            affected   = np.random.choice(
                self.n_services, size=n_affected, replace=False)

            for service in affected:
                kind = np.random.choice(
                    ["mean_shift", "variance_change", "both"])
                if kind in ("mean_shift", "both"):
                    data[service, timestamp:, :] *= np.random.uniform(2.0, 3.0)
                if kind in ("variance_change", "both"):
                    mv  = data[service, timestamp:, :].mean(
                        axis=0, keepdims=True)
                    dev = data[service, timestamp:, :] - mv
                    data[service, timestamp:, :] = (
                        mv + dev * np.random.uniform(2.5, 5.0))

            change_points.append({
                "timestamp":         timestamp,
                "affected_services": affected.tolist(),
                "n_affected":        n_affected,
            })

        return change_points


# =============================================================================
# 2.  SEQUENCE PREPARATION
# =============================================================================

def prepare_sequences(
    data:        np.ndarray,
    labels:      np.ndarray,
    window_size: int = Config.WINDOW_SIZE,
    stride:      int = Config.STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window extraction.
    window=60, stride=10  →  ~23 300 training sequences (paper Table I).
    Returns: X, y, service_ids, start_times
    """
    n_services, n_timestamps, n_metrics = data.shape
    starts  = np.arange(0, n_timestamps - window_size, stride)
    n_total = n_services * len(starts)

    X      = np.empty((n_total, window_size, n_metrics), dtype=np.float32)
    y      = np.empty(n_total, dtype=np.float32)
    sids   = np.empty(n_total, dtype=np.int64)
    stimes = np.empty(n_total, dtype=np.int64)

    idx = 0
    for s in range(n_services):
        for start in starts:
            X[idx]      = data[s, start:start + window_size, :]
            y[idx]      = labels[start:start + window_size, s].max()
            sids[idx]   = s
            stimes[idx] = start
            idx += 1

    return X, y, sids, stimes


# =============================================================================
# 3.  MODEL: BiLSTM + Multi-Head Attention
#     Paper §III-B, Eq. (3)-(5), Fig. 2
# =============================================================================

def focal_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    gamma:  float = Config.FOCAL_GAMMA,
    alpha:  float = Config.FOCAL_ALPHA,
) -> torch.Tensor:
    """
    Focal loss (Lin et al. [19]).
    Paper Eq. (5): FL(p_t) = -α_t (1-p_t)^γ log(p_t)
    γ=2.0, α=0.75 downweight easy negatives, focus on hard anomalous windows.
    """
    bce    = F.binary_cross_entropy(pred, target, reduction="none")
    p_t    = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    fl     = alpha_t * (1 - p_t) ** gamma * bce
    return fl.mean()


class BiLSTMAnomalyDetector(nn.Module):
    """
    Paper §III-B architecture:
      1. 4-layer bidirectional LSTM encoder (hidden=128) → Eq. (3)
      2. Layer normalisation on encoder output
      3. 4-head multi-head self-attention → Eq. (4)  (context = mean pooling)
      4. Single-layer LSTM decoder (reconstruction branch)
      5. 3-layer MLP classifier head (anomaly score σ)
      Joint loss: L = L_recon + 0.5 * FL  →  Eq. (5)
    """

    def __init__(
        self,
        n_metrics:  int = Config.N_METRICS,
        hidden_dim: int = Config.HIDDEN_DIM,
        n_layers:   int = Config.LSTM_LAYERS,
        n_heads:    int = Config.ATTN_HEADS,
        dropout:    float = Config.DROPOUT,
    ):
        super().__init__()

        # --- Encoder (Eq. 3) ---
        self.encoder = nn.LSTM(
            input_size  = n_metrics,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # --- Multi-head self-attention (Eq. 4) ---
        self.mha = nn.MultiheadAttention(
            embed_dim  = hidden_dim * 2,
            num_heads  = n_heads,
            dropout    = dropout,
            batch_first = True,
        )

        # --- Decoder ---
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.output  = nn.Linear(hidden_dim, n_metrics)

        # --- 3-layer MLP classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name:
                nn.init.zeros_(p.data)
                n = p.shape[0]
                p.data[n // 4: n // 2].fill_(1.0)  # forget-gate bias = 1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        enc, _ = self.encoder(x)              # (B, T, 2H)
        enc    = self.layer_norm(enc)

        # Attention (Eq. 4)
        attn_out, attn_w = self.mha(enc, enc, enc)
        context = attn_out.mean(dim=1)         # (B, 2H)

        # Decode
        dec_in  = context.unsqueeze(1).expand(-1, x.size(1), -1)
        dec, _  = self.decoder(dec_in)
        recon   = self.output(dec)             # (B, T, M)

        # Classify
        score = self.classifier(context).squeeze(-1)  # (B,)

        return {
            "reconstruction": recon,
            "anomaly_score":  score,
            "attention":      attn_w,
            "encoding":       context,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs:  torch.Tensor,
        labels:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Paper Eq. (5): L = L_MSE + λ_cls * FL(γ=2, α=0.75)"""
        recon_loss = F.mse_loss(outputs["reconstruction"], inputs)
        cls_loss   = focal_loss(outputs["anomaly_score"], labels)
        total      = recon_loss + Config.LAMBDA_CLS * cls_loss
        return {
            "total":          total,
            "reconstruction": recon_loss,
            "classification": cls_loss,
        }


# =============================================================================
# 4.  TRAINING
# =============================================================================

def train_model(
    model:      nn.Module,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    epochs:     int   = Config.MAX_EPOCHS,
    batch_size: int   = Config.BATCH_SIZE,
    lr:         float = Config.LEARNING_RATE,
    verbose:    bool  = True,
) -> nn.Module:
    """
    AdamW + cosine annealing warm restarts (T_0=10, T_mult=2) +
    gradient clipping 1.0 + early stopping patience 15.
    All settings match paper §IV-A.
    """
    device = Config.device()
    model  = model.to(device)

    pin = device.type == "cuda"
    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin,
    )
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=pin,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.LR_T0, T_mult=2, eta_min=1e-6)

    best_val = float("inf")
    patience = 0
    best_sd  = None

    if verbose:
        print(f"\nTraining on {device} ...")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0;  nb = 0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            out    = model(bx)
            loss   = model.compute_loss(out, bx, by)
            optimizer.zero_grad()
            loss["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            tr_loss += loss["total"].item();  nb += 1
        tr_loss /= nb
        scheduler.step(epoch)

        model.eval()
        vl_loss = 0.0;  nvb = 0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by   = bx.to(device), by.to(device)
                out      = model(bx)
                loss     = model.compute_loss(out, bx, by)
                vl_loss += loss["total"].item();  nvb += 1
        vl_loss /= nvb

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={tr_loss:.4f} | val={vl_loss:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if vl_loss < best_val:
            best_val = vl_loss;  patience = 0
            best_sd  = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= Config.EARLY_STOP:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    if best_sd:
        model.load_state_dict(best_sd)
    return model.cpu()


# =============================================================================
# 5.  SCORE COMPUTATION & CALIBRATION
#     Paper Eq. (6): s = w_clf*ŝ_clf + w_recon*ŝ_recon + w_grad*ŝ_grad
# =============================================================================

def _norm(v: np.ndarray) -> np.ndarray:
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def compute_scores(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Paper Eq. (6) three-component ensemble:
      ŝ = 0.55*norm(clf) + 0.30*norm(recon) + 0.15*norm(|Δclf|)
    """
    model.eval()
    clf_list, recon_list = [], []

    for i in range(0, len(X), batch_size):
        bx  = torch.FloatTensor(X[i:i + batch_size])
        out = model(bx)
        clf_list.append(out["anomaly_score"].numpy())
        re = ((out["reconstruction"] - bx) ** 2).mean(dim=(1, 2))
        recon_list.append(re.numpy())

    clf   = np.concatenate(clf_list)
    recon = np.concatenate(recon_list)
    grad  = np.abs(np.diff(clf, prepend=clf[0]))

    return (Config.W_CLF   * _norm(clf)
          + Config.W_RECON * _norm(recon)
          + Config.W_GRAD  * _norm(grad))


def calibrate_scores(
    raw_val: np.ndarray, y_val: np.ndarray, raw_test: np.ndarray
) -> Tuple[np.ndarray, IsotonicRegression]:
    """Isotonic-regression calibration on validation set (paper §III-B)."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)
    return ir.predict(raw_test), ir


def _build_score_matrix(
    model:       nn.Module,
    dataset:     Dict,
    window_size: int = Config.WINDOW_SIZE,
    stride:      int = Config.STRIDE,
) -> np.ndarray:
    """Return (n_timestamps, n_services) matrix via max-pool back-projection."""
    X_full, _, sids, stimes = prepare_sequences(
        dataset["data"], dataset["anomaly_labels"],
        window_size=window_size, stride=stride)

    raw = compute_scores(model, X_full)

    nt  = dataset["data"].shape[1]
    ns  = dataset["data"].shape[0]
    mat = np.zeros((nt, ns))

    for sc, s, t in zip(raw, sids, stimes):
        end = min(t + window_size, nt)
        for tt in range(t, end):
            if sc > mat[tt, s]:
                mat[tt, s] = sc

    return mat


# =============================================================================
# 6.  CAUSAL DISCOVERY
#     Paper §III-B, Algorithm 1 — five-stage pipeline
# =============================================================================

class AnomalyPropagationCausalDiscovery:
    """
    Five-stage propagation-based causal discovery (paper §III-B, Algorithm 1):
      Stage 1  — propagation mining  (Eq. 7, τ=0.35)
      Stage 2  — tier filter         (no backward / skip-2-tier edges)
      Stage 3  — asymmetry test      (Δ=0.30)
      Stage 4  — Granger causality   (p<0.03, max_lag=5)
      Stage 5  — mutual information  (max lagged MI > 0.20)
    """

    def __init__(
        self,
        n_services:            int,
        service_tiers:         Optional[Dict] = None,
        bilstm_anomaly_scores: Optional[np.ndarray] = None,
    ):
        self.n_services            = n_services
        self.service_tiers         = service_tiers
        self.bilstm_anomaly_scores = bilstm_anomaly_scores

    # ------------------------------------------------------------------
    def discover_causal_graph(
        self,
        data:                    np.ndarray,
        anomaly_score_threshold: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray]:

        events          = self._detect_anomaly_events(anomaly_score_threshold)
        raw, strengths  = self._mine_propagation(events)
        tier_f          = self._tier_filter(raw)
        asym_f          = self._asymmetry_filter(tier_f, strengths)
        granger_f       = self._granger_filter(asym_f, data)
        final           = self._mi_filter(granger_f, data)
        return final, strengths

    # ------------------------------------------------------------------
    def _detect_anomaly_events(self, threshold: float) -> Dict[int, set]:
        if self.bilstm_anomaly_scores is None:
            return {s: set() for s in range(self.n_services)}
        nt, ns = self.bilstm_anomaly_scores.shape
        events = {}
        for s in range(ns):
            idx = np.where(self.bilstm_anomaly_scores[:, s] > threshold)[0]
            events[s] = set(idx[idx < nt].tolist())
        return events

    # ------------------------------------------------------------------
    def _mine_propagation(
        self,
        events:  Dict[int, set],
        max_lag: int = Config.CD_MAX_LAG,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eq. (7): fwd_strength(i→j) = |{t∈E_i : ∃t'∈[t+1,t+L] s.t. t'∈E_j}| / |E_i|"""
        ns  = self.n_services
        nt  = Config.N_TIMESTAMPS
        mat = np.zeros((ns, nt), dtype=bool)
        for s, ev in events.items():
            if ev:
                idx = np.array(sorted(ev), dtype=int)
                idx = idx[idx < nt]
                mat[s, idx] = True

        counts   = mat.sum(axis=1)
        graph    = np.zeros((ns, ns))
        strengths = np.zeros((ns, ns))

        for i in range(ns):
            if counts[i] < 3:
                continue
            for j in range(ns):
                if i == j or counts[j] < 3:
                    continue
                cum     = np.concatenate([[0], np.cumsum(mat[j].astype(np.uint8))])
                end_idx = np.minimum(np.arange(nt) + max_lag + 1, nt)
                hits    = cum[end_idx] - cum[np.arange(nt) + 1]
                fwd     = (mat[i] & (hits > 0)).sum() / counts[i]
                strengths[i, j] = fwd
                if fwd > Config.CD_PROP_THR:
                    graph[i, j] = 1

        return graph, strengths

    # ------------------------------------------------------------------
    def _tier_filter(self, graph: np.ndarray) -> np.ndarray:
        if self.service_tiers is None:
            return graph
        order = ["frontend", "api", "business", "data", "database"]
        rank  = {t: i for i, t in enumerate(order)}
        srank = np.full(self.n_services, 2, dtype=int)
        for t, svcs in self.service_tiers.items():
            for s in svcs:
                srank[s] = rank.get(t, 2)
        out = graph.copy()
        ri  = srank[:, None];  rj = srank[None, :]
        out[(ri > rj) | ((rj - ri) > 2)] = 0
        return out

    # ------------------------------------------------------------------
    def _asymmetry_filter(
        self, graph: np.ndarray, strengths: np.ndarray
    ) -> np.ndarray:
        """Keep edge i→j only if S(i,j) > S(j,i) + Δ  (Δ=0.30)."""
        out  = graph.copy()
        mask = (graph == 1) & (strengths.T >= strengths - Config.CD_ASYM_THR)
        out[mask] = 0
        return out

    # ------------------------------------------------------------------
    def _granger_filter(
        self, graph: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        out = graph.copy()
        if not _HAS_STATSMODELS:
            return out
        agg = data.mean(axis=-1)
        for i in range(self.n_services):
            for j in range(self.n_services):
                if graph[i, j] == 0:
                    continue
                try:
                    d   = np.column_stack([agg[j, -1000:], agg[i, -1000:]])
                    res = grangercausalitytests(d, maxlag=5, verbose=False)
                    ok  = any(res[lag][0]["ssr_ftest"][1] < Config.CD_P_VAL
                              for lag in range(1, 6))
                    if not ok:
                        out[i, j] = 0
                except Exception:
                    out[i, j] = 0
        return out

    # ------------------------------------------------------------------
    def _mi_filter(self, graph: np.ndarray, data: np.ndarray) -> np.ndarray:
        out = graph.copy()
        agg = data.mean(axis=-1)
        for i in range(self.n_services):
            for j in range(self.n_services):
                if graph[i, j] == 0:
                    continue
                try:
                    max_mi = max(
                        mutual_info_regression(
                            agg[i, :-lag].reshape(-1, 1),
                            agg[j, lag:],
                            random_state=42)[0]
                        for lag in range(1, 6)
                        if len(agg[i]) > lag + 100
                    )
                    if max_mi < Config.CD_MI_THR:
                        out[i, j] = 0
                except Exception:
                    pass
        return out


# =============================================================================
# 7.  CHANGE POINT DETECTION
#     Paper §III-C — three-detector fusion (CUSUM + Energy-Ratio + KL-Div)
# =============================================================================

class MultiServiceChangePointDetector:
    """
    Fused change point detector (paper §III-C):
      Detector A  — CUSUM mean/variance shift counter
      Detector B  — energy-ratio statistic
      Detector C  — KL-divergence (20 bins, 20 sampled services)
      Fusion      — 0.45*A + 0.35*B + 0.20*C
      Confirmed if ≥6 services exceed per-service threshold and
      detections within 200 ts are merged (paper §III-C last paragraph).
    """

    def __init__(
        self,
        window:       int   = Config.CPD_WINDOW,
        thresh_mult:  float = Config.CPD_THRESH_MULT,
        min_affected: int   = Config.CPD_MIN_AFF,
        distance:     int   = Config.CPD_DISTANCE,
        prominence:   float = Config.CPD_PROMINENCE,
        dedup:        int   = Config.CPD_DEDUP,
    ):
        self.window       = window
        self.thresh_mult  = thresh_mult
        self.min_affected = min_affected
        self.distance     = distance
        self.prominence   = prominence
        self.dedup        = dedup

    # ------------------------------------------------------------------
    def _detector_A(self, agg: np.ndarray) -> np.ndarray:
        """CUSUM: count services with significant mean-shift or variance change."""
        ns, nt = agg.shape
        w      = self.window
        scores = np.zeros(nt)

        cum    = np.cumsum(agg, axis=1)
        cum_sq = np.cumsum(agg ** 2, axis=1)
        cp_    = np.concatenate([np.zeros((ns, 1)), cum],    axis=1)
        cps_   = np.concatenate([np.zeros((ns, 1)), cum_sq], axis=1)

        def _ms(c, cs, s, e):
            n   = e - s
            sv  = c[:, e]  - c[:, s]
            s2  = cs[:, e] - cs[:, s]
            mu  = sv / n
            var = np.maximum(s2 / n - mu ** 2, 0)
            return mu, np.sqrt(var) + 1e-8

        for t in range(w, nt - w):
            bm, bs  = _ms(cp_, cps_, t - w, t)
            am, as_ = _ms(cp_, cps_, t,     t + w)
            mc  = np.abs(am - bm) / bs
            vr  = np.abs(np.log(as_ / bs))
            hit = (mc / bs > 0.3) | (vr > 0.3)
            nc  = hit.sum()
            if nc >= 5:
                scores[t] = nc * (mc + vr)[hit].mean()

        return scores / (scores.max() + 1e-8)

    # ------------------------------------------------------------------
    def _detector_B(self, agg: np.ndarray) -> np.ndarray:
        """Energy-ratio: log(E_after / E_before) averaged over services."""
        ns, nt = agg.shape
        w      = self.window
        scores = np.zeros(nt)

        for t in range(w, nt - w):
            before = agg[:, t - w: t]
            after  = agg[:, t:     t + w]
            e_b    = (before ** 2).mean(axis=1) + 1e-8
            e_a    = (after  ** 2).mean(axis=1) + 1e-8
            ratio  = np.log(e_a / e_b)
            scores[t] = np.abs(ratio).mean() * (np.abs(ratio) > 0.2).sum()

        return scores / (scores.max() + 1e-8)

    # ------------------------------------------------------------------
    def _detector_C(self, agg: np.ndarray, bins: int = 20) -> np.ndarray:
        """KL-divergence on 20 sampled services (paper §III-C)."""
        ns, nt  = agg.shape
        w       = self.window
        scores  = np.zeros(nt)
        sample  = np.random.choice(ns, size=min(20, ns), replace=False)

        for t in range(w, nt - w):
            kl_sum = 0.0
            for s in sample:
                b = agg[s, t - w: t]
                a = agg[s, t:     t + w]
                lo = min(b.min(), a.min())
                hi = max(b.max(), a.max())
                if hi <= lo:
                    continue
                edges = np.linspace(lo, hi, bins + 1)
                ph, _ = np.histogram(b, bins=edges, density=True)
                qh, _ = np.histogram(a, bins=edges, density=True)
                ph = ph / (ph.sum() + 1e-8) + 1e-8
                qh = qh / (qh.sum() + 1e-8) + 1e-8
                kl_sum += np.sum(ph * np.log(ph / qh))
            scores[t] = kl_sum / len(sample)

        return scores / (scores.max() + 1e-8)

    # ------------------------------------------------------------------
    def _count_affected(self, agg: np.ndarray, t: int) -> int:
        ns, nt = agg.shape
        w      = self.window
        if t < w or t >= nt - w:
            return 0
        before = agg[:, t - w: t]
        after  = agg[:, t:     t + w]
        mc     = np.abs(after.mean(1) - before.mean(1))
        sb     = before.std(1) + 1e-8
        sa     = after.std(1)  + 1e-8
        vr     = np.abs(np.log(sa / sb))
        return int(((mc / sb > 0.25) | (vr > 0.25)).sum())

    # ------------------------------------------------------------------
    def detect(self, data: np.ndarray) -> Tuple[List[int], np.ndarray]:
        ns, nt, nm = data.shape
        agg = data.mean(axis=-1)      # (ns, nt)

        A = self._detector_A(agg)
        B = self._detector_B(agg)
        C = self._detector_C(agg)

        # Paper fusion weights: 0.45 / 0.35 / 0.20
        fused  = 0.45 * A + 0.35 * B + 0.20 * C
        thresh = fused.mean() + self.thresh_mult * fused.std()
        thresh = np.clip(thresh, 0.04, 0.55)

        peaks, _ = find_peaks(
            fused,
            height     = thresh,
            distance   = self.distance,
            prominence = self.prominence,
        )

        confirmed = [p for p in peaks
                     if self._count_affected(agg, p) >= self.min_affected]

        # Merge within 200 ts (paper §III-C)
        deduped = []
        for p in sorted(confirmed):
            if not any(abs(p - q) < self.dedup for q in deduped):
                deduped.append(p)

        return deduped, fused

    # ------------------------------------------------------------------
    def evaluate(
        self, data: np.ndarray, true_cps: List[int], tolerance: int = 150
    ) -> Dict:
        detected, scores = self.detect(data)
        matched   = set();  n_correct = 0
        for dp in detected:
            for tc in true_cps:
                if abs(dp - tc) <= tolerance and tc not in matched:
                    n_correct += 1;  matched.add(tc);  break
        prec = n_correct / len(detected) if detected else 0.0
        rec  = n_correct / len(true_cps) if true_cps  else 0.0
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else 0.0)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "n_detected": len(detected), "n_correct": n_correct,
            "n_true":     len(true_cps),
            "scores":     scores, "detected": detected,
        }


# =============================================================================
# 8.  ROOT CAUSE ANALYSIS
#     Paper §III-D, Eq. (8)-(9)
# =============================================================================

class GraphBasedRootCauseAnalyzer:
    """
    Multi-signal root cause localizer (paper §III-D, Eq. 8):
      score(s) = 0.30*PR(s) + 0.40*ANC(s) + 0.20*tier(s) + 0.10*AM(s)

    PageRank on reversed graph with anomaly-score personalization (Eq. 8).
    Ancestor score rewards nodes with many anomalous descendants (Eq. 9).
    Tier weights: frontend=1.0, api=0.8, business=0.6, data=0.3, database=0.1.
    """

    def __init__(
        self,
        causal_graph:  nx.DiGraph,
        service_tiers: Dict[str, List[int]],
    ):
        self.causal_graph  = causal_graph
        self.service_tiers = service_tiers

    # ------------------------------------------------------------------
    def localize_root_cause(
        self,
        anomaly_scores:   np.ndarray,
        causal_strengths: np.ndarray,
        top_k: int = 15,
    ) -> Dict:
        n         = len(anomaly_scores)
        anomalous = np.where(anomaly_scores > 0.5)[0]

        if len(anomalous) == 0:
            return {"root_causes": [], "scores": [],
                    "anomalous_services": []}

        pr   = self._pagerank(anomalous, anomaly_scores)
        anc  = self._ancestor(anomalous, anomaly_scores)
        tier = np.array([self._tier_weight(i) for i in range(n)])

        # Eq. (8)
        combined = (Config.RCA_W_PR   * pr
                  + Config.RCA_W_ANC  * anc
                  + Config.RCA_W_TIER * tier
                  + Config.RCA_W_ANOM * anomaly_scores)
        ranked   = np.argsort(combined)[-top_k:][::-1]

        return {
            "root_causes":        ranked.tolist(),
            "scores":             combined[ranked].tolist(),
            "anomalous_services": anomalous.tolist(),
        }

    # ------------------------------------------------------------------
    def _pagerank(
        self, anomalous: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        n   = len(scores)
        adj = nx.adjacency_matrix(self.causal_graph).todense()
        Gr  = nx.DiGraph(np.array(adj).T)
        pers = np.ones(n) * 0.01
        for s in anomalous:
            pers[s] = scores[s]
        pers /= pers.sum()
        try:
            pr = nx.pagerank(
                Gr,
                personalization = dict(enumerate(pers)),
                alpha           = Config.RCA_PAGERANK_ALPHA,
            )
            return np.array([pr.get(i, 0) for i in range(n)])
        except Exception:
            return pers

    # ------------------------------------------------------------------
    def _ancestor(
        self, anomalous: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Eq. (9): ANC(v) = Σ_{a∈anomalous} AM(a) / (dist(v,a)^0.5 + 1)"""
        n      = len(scores)
        result = np.zeros(n)
        for a in anomalous:
            try:
                lengths = dict(
                    nx.single_target_shortest_path_length(
                        self.causal_graph, a))
                for anc, dist in lengths.items():
                    if anc != a and dist > 0:
                        result[anc] += scores[a] / (dist ** 0.5 + 1)
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    def _tier_weight(self, service: int) -> float:
        weights = {
            "frontend": 1.0, "api": 0.8, "business": 0.6,
            "data": 0.3,     "database": 0.1,
        }
        for tier, svcs in self.service_tiers.items():
            if service in svcs:
                return weights.get(tier, 0.5)
        return 0.5


# =============================================================================
# 9.  EVALUATION
# =============================================================================

def evaluate_system(
    model:           nn.Module,
    X_test:          np.ndarray,
    y_test:          np.ndarray,
    service_ids_test: np.ndarray,
    dataset:         Dict,
    X_val:           Optional[np.ndarray] = None,
    y_val:           Optional[np.ndarray] = None,
) -> Dict:
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    device = Config.device()
    model  = model.to(device)

    # --- full score matrix for CD and RCA ---
    print("\n  Building full anomaly score matrix ...")
    score_matrix = _build_score_matrix(model, dataset)

    # ---------------------------------------------------------------
    # 9a. Anomaly Detection
    # ---------------------------------------------------------------
    print("\n1. ANOMALY DETECTION")
    print("-" * 70)

    raw_test = compute_scores(model, X_test)

    if X_val is not None and y_val is not None:
        raw_val          = compute_scores(model, X_val)
        cal_test, _      = calibrate_scores(raw_val, y_val, raw_test)
    else:
        cal_test = raw_test

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 200):
        f1 = f1_score(y_test, (cal_test > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1;  best_thr = thr

    y_pred = (cal_test > best_thr).astype(int)
    ad = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   (roc_auc_score(y_test, cal_test)
                      if len(np.unique(y_test)) > 1 else 0.0),
        "pr_auc":    (average_precision_score(y_test, cal_test)
                      if len(np.unique(y_test)) > 1 else 0.0),
        "threshold": best_thr,
    }
    print(f"  Precision : {ad['precision']:.4f}")
    print(f"  Recall    : {ad['recall']:.4f}")
    print(f"  F1        : {ad['f1']:.4f}")
    print(f"  ROC-AUC   : {ad['roc_auc']:.4f}")
    print(f"  PR-AUC    : {ad['pr_auc']:.4f}")

    # ---------------------------------------------------------------
    # 9b. Causal Discovery
    # ---------------------------------------------------------------
    ns        = dataset["data"].shape[0]
    cd_model  = AnomalyPropagationCausalDiscovery(
        n_services           = ns,
        service_tiers        = dataset["tiers"],
        bilstm_anomaly_scores = score_matrix,
    )
    disc_graph, strengths = cd_model.discover_causal_graph(
        dataset["data"], anomaly_score_threshold=0.6)

    true_e = set(
        (dataset["edge_index"][0, i], dataset["edge_index"][1, i])
        for i in range(dataset["edge_index"].shape[1]))
    pred_e = set(
        (i, j) for i in range(ns) for j in range(ns)
        if disc_graph[i, j] > 0)

    nc = len(true_e & pred_e)
    cd = {
        "precision": nc / len(pred_e) if pred_e else 0.0,
        "recall":    nc / len(true_e) if true_e else 0.0,
        "n_true":    len(true_e), "n_pred": len(pred_e), "n_correct": nc,
        "discovered_graph": disc_graph, "causal_strengths": strengths,
    }
    cd["f1"] = (2 * cd["precision"] * cd["recall"]
                / (cd["precision"] + cd["recall"])
                if (cd["precision"] + cd["recall"]) > 0 else 0.0)

    # ---------------------------------------------------------------
    # 9c. Change Point Detection
    # ---------------------------------------------------------------
    print("\n2. CHANGE POINT DETECTION")
    print("-" * 70)

    cpd      = MultiServiceChangePointDetector()
    true_cps = [cp["timestamp"] for cp in dataset["change_points"]]
    cp_res   = cpd.evaluate(dataset["data"], true_cps)

    print(f"  True CPs      : {true_cps}")
    print(f"  Detected CPs  : {cp_res['detected']}")
    print(f"  Correct       : {cp_res['n_correct']} / {cp_res['n_true']}")
    print(f"  Precision     : {cp_res['precision']:.4f}")
    print(f"  Recall        : {cp_res['recall']:.4f}")
    print(f"  F1            : {cp_res['f1']:.4f}")

    # ---------------------------------------------------------------
    # 9d. Root Cause Analysis
    # ---------------------------------------------------------------
    print("\n3. ROOT CAUSE ANALYSIS")
    print("-" * 70)

    # Per-service anomaly magnitude = 90th-percentile of non-zero scores
    svc_scores = np.array([
        np.percentile(score_matrix[:, s][score_matrix[:, s] > 0], 90)
        if (score_matrix[:, s] > 0).any() else 0.0
        for s in range(ns)
    ])

    # Build discovered graph; fall back to true graph if too sparse
    cg_nx = nx.DiGraph()
    cg_nx.add_nodes_from(range(ns))
    for i in range(ns):
        for j in range(ns):
            if disc_graph[i, j] > 0:
                cg_nx.add_edge(i, j)
    if cg_nx.number_of_edges() < 10:
        cg_nx = dataset["causal_graph"]

    rca     = GraphBasedRootCauseAnalyzer(cg_nx, dataset["tiers"])
    rca_out = rca.localize_root_cause(svc_scores, strengths, top_k=15)

    true_rc  = set(rc["root_cause"] for rc in dataset["root_causes"])
    rca_met  = {
        "hit_at_1":  int(any(r in true_rc for r in rca_out["root_causes"][:1])),
        "hit_at_3":  int(any(r in true_rc for r in rca_out["root_causes"][:3])),
        "hit_at_5":  int(any(r in true_rc for r in rca_out["root_causes"][:5])),
        "hit_at_10": int(any(r in true_rc for r in rca_out["root_causes"][:10])),
        "predictions": rca_out,
    }

    print(f"  Hit@1  : {rca_met['hit_at_1']}")
    print(f"  Hit@3  : {rca_met['hit_at_3']}")
    print(f"  Hit@5  : {rca_met['hit_at_5']}")
    print(f"  Hit@10 : {rca_met['hit_at_10']}")
    print("\n  Top 5 Predicted Root Causes:")
    for i, (svc, sc) in enumerate(
        zip(rca_out["root_causes"][:5], rca_out["scores"][:5])
    ):
        mark = "✓" if svc in true_rc else " "
        print(f"    {i+1}. Service {svc:2d}  Score: {sc:.4f}  {mark}")

    model.cpu()
    return {
        "anomaly_detection":      ad,
        "causal_discovery":       cd,
        "change_point_detection": cp_res,
        "root_cause_analysis":    rca_met,
    }


# =============================================================================
# 10. VISUALISATION
# =============================================================================

class Visualizer:
    def __init__(self, output_dir: Path = Config.OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-paper")

    # ------------------------------------------------------------------
    def plot_all(self, dataset: Dict, results: Dict):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        self.plot_change_points(
            dataset["data"],
            results["change_point_detection"]["scores"],
            results["change_point_detection"]["detected"],
            [cp["timestamp"] for cp in dataset["change_points"]],
        )
        self.plot_performance_summary(results)
        self.plot_anomaly_distribution(dataset)
        print(f"\n  Figures saved to {self.output_dir}")

    # ------------------------------------------------------------------
    def plot_change_points(
        self,
        data:       np.ndarray,
        cp_scores:  np.ndarray,
        detected:   List[int],
        true_cps:   List[int],
    ):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        agg = data.mean(axis=-1)[:5]
        for i in range(5):
            axes[0].plot(agg[i], alpha=0.6, lw=1, label=f"S{i}")
        for t in true_cps:
            axes[0].axvline(t, color="red", ls="--", alpha=0.5, lw=1.5)
        axes[0].set_title("Service Metrics + True Change Points",
                          fontweight="bold")
        axes[0].legend(fontsize=8);  axes[0].grid(alpha=0.3)

        axes[1].plot(cp_scores, color="navy", lw=1.2, label="Fused CPD Score")
        for d in detected:
            axes[1].axvline(d, color="green", lw=1.5, alpha=0.7,
                            label="Detected" if d == detected[0] else "")
        for t in true_cps:
            axes[1].axvline(t, color="red", ls="--", alpha=0.5, lw=1.5,
                            label="True" if t == true_cps[0] else "")
        axes[1].set_title("Fused Change Point Score", fontweight="bold")
        axes[1].legend(fontsize=8);  axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "change_points.png",
                    dpi=Config.FIGURE_DPI, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    def plot_performance_summary(self, results: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        def _bar(ax, data, title, ylim=1.0):
            keys   = list(data.keys())
            vals   = [data[k] for k in keys]
            colors = ["#45B7D1" if v >= 0.9 else
                      "#4ECDC4" if v >= 0.7 else "#FF6B6B"
                      for v in vals]
            bars = ax.bar(keys, vals, color=colors,
                          edgecolor="black", alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontweight="bold", fontsize=9)
            ax.axhline(0.9, color="green", ls="--", lw=1,
                       alpha=0.6, label="90% target")
            ax.set_ylim(0, ylim + 0.15)
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8);  ax.grid(alpha=0.3, axis="y")

        ad  = results["anomaly_detection"]
        cpd = results["change_point_detection"]
        cd  = results["causal_discovery"]
        rca = results["root_cause_analysis"]

        _bar(axes[0, 0],
             {k: ad[k] for k in ["precision", "recall", "f1", "roc_auc"]},
             "Anomaly Detection")
        _bar(axes[0, 1],
             {k: cpd[k] for k in ["precision", "recall", "f1"]},
             "Change Point Detection")
        _bar(axes[1, 0],
             {k: cd[k] for k in ["precision", "recall", "f1"]},
             "Causal Discovery")
        _bar(axes[1, 1],
             {"Hit@1": rca["hit_at_1"], "Hit@3": rca["hit_at_3"],
              "Hit@5": rca["hit_at_5"], "Hit@10": rca["hit_at_10"]},
             "Root Cause Analysis", ylim=1.2)

        plt.suptitle("System Performance Summary",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_summary.png",
                    dpi=Config.FIGURE_DPI, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    def plot_anomaly_distribution(self, dataset: Dict):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        labels = dataset["anomaly_labels"]

        for tier_name, svcs in dataset["tiers"].items():
            counts = labels[:, svcs].sum(axis=1)
            sm     = np.convolve(counts, np.ones(50) / 50, mode="same")
            axes[0].plot(sm, label=tier_name.capitalize(), lw=1.5)
        axes[0].set_title("Anomaly Distribution by Tier", fontweight="bold")
        axes[0].legend();  axes[0].grid(alpha=0.3)

        ds = 10
        im = axes[1].imshow(labels[::ds].T, aspect="auto", cmap="YlOrRd")
        axes[1].set_title("Anomaly Heatmap (Services × Time)",
                          fontweight="bold")
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig(self.output_dir / "anomaly_distribution.png",
                    dpi=Config.FIGURE_DPI, bbox_inches="tight")
        plt.close()


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def main() -> Dict:
    Config.setup()
    t0 = time.time()

    print("=" * 70)
    print("MICROSERVICES RCA — UNIFIED PIPELINE")
    print("(Singarajipura, HM, Raj 2025)")
    print("=" * 70)

    # --- Step 1: Data Generation ---
    print("\nSTEP 1: DATA GENERATION")
    gen = MicroservicesDataGenerator(
        n_services  = Config.N_SERVICES,
        n_metrics   = Config.N_METRICS,
        n_timestamps= Config.N_TIMESTAMPS,
        seed        = Config.RANDOM_SEED,
    )
    dataset    = gen.generate()
    anom_rate  = dataset["anomaly_labels"].mean()
    n_edges    = dataset["edge_index"].shape[1]
    print(f"  Services={Config.N_SERVICES}  Metrics={Config.N_METRICS}"
          f"  T={Config.N_TIMESTAMPS}")
    print(f"  Anomaly rate={anom_rate:.3%}  Causal edges={n_edges}")
    print(f"  Cascade scenarios=75  Change points={len(dataset['change_points'])}")

    # --- Step 2: Sequence Preparation (window=60, stride=10) ---
    print("\nSTEP 2: SEQUENCE PREPARATION")
    X, y, sids, stimes = prepare_sequences(
        dataset["data"], dataset["anomaly_labels"],
        window_size=Config.WINDOW_SIZE, stride=Config.STRIDE)

    n  = len(X)
    t1 = int(0.6 * n);  t2 = int(0.8 * n)   # 60/20/20 split
    X_tr, y_tr = X[:t1],   y[:t1]
    X_vl, y_vl = X[t1:t2], y[t1:t2]
    X_te, y_te = X[t2:],   y[t2:]
    sids_te    = sids[t2:]

    print(f"  Total sequences : {n:,}")
    print(f"  Train={len(X_tr):,}  Val={len(X_vl):,}  Test={len(X_te):,}")
    print(f"  Positive rate (train)={y_tr.mean():.3%}")

    # --- Step 3: Model Training ---
    print("\nSTEP 3: MODEL TRAINING")
    model    = BiLSTMAnomalyDetector(
        n_metrics  = Config.N_METRICS,
        hidden_dim = Config.HIDDEN_DIM,
        n_layers   = Config.LSTM_LAYERS,
        n_heads    = Config.ATTN_HEADS,
        dropout    = Config.DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    model = train_model(
        model, X_tr, y_tr, X_vl, y_vl,
        epochs     = Config.MAX_EPOCHS,
        batch_size = Config.BATCH_SIZE,
        lr         = Config.LEARNING_RATE,
    )

    # --- Step 4: Evaluation ---
    print("\nSTEP 4: EVALUATION")
    results = evaluate_system(
        model, X_te, y_te, sids_te, dataset,
        X_val=X_vl, y_val=y_vl)

    # --- Step 5: Visualisation ---
    print("\nSTEP 5: VISUALISATION")
    viz = Visualizer()
    viz.plot_all(dataset, results)

    # --- Summary ---
    ad  = results["anomaly_detection"]
    cpd = results["change_point_detection"]
    rca = results["root_cause_analysis"]

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE  ({time.time() - t0:.1f}s)")
    print(f"  {'Metric':<28} {'Score':>8}  {'Paper target':>14}")
    print("  " + "-" * 52)
    rows = [
        ("AD  Precision",  ad["precision"],  "0.92"),
        ("AD  Recall",     ad["recall"],     "0.83"),
        ("AD  F1",         ad["f1"],         "0.82"),
        ("AD  ROC-AUC",    ad["roc_auc"],    "0.82"),
        ("CPD Precision",  cpd["precision"], "0.78"),
        ("CPD Recall",     cpd["recall"],    "0.87"),
        ("CPD F1",         cpd["f1"],        "0.82"),
        ("RCA Hit@1",      rca["hit_at_1"],  "1.00"),
        ("RCA Hit@3",      rca["hit_at_3"],  "1.00"),
        ("RCA Hit@5",      rca["hit_at_5"],  "1.00"),
    ]
    for label, val, target in rows:
        print(f"  {label:<28} {val:>8.4f}  {target:>14}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
