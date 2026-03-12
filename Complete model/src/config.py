# Global hyper-parameters exactly as reported in the paper.

from pathlib import Path
import numpy as np
import torch


class Config:
    """
    All values taken directly from the paper (Table I, §III, §IV-A).

    Data          : N=50 services, M=6 metrics, T=5000 timestamps
    Model         : hidden_dim=128, 4-layer BiLSTM, 4 attention heads
    Training      : AdamW lr=1e-4 wd=1e-4, cosine-annealing T0=10,
                    gradient clip 1.0, max 50 epochs, early-stop patience 15
    Sequences     : window=60, stride=10  →  ~23 300 training sequences
    Focal loss    : γ=2.0, α=0.75
    Score fusion  : w_clf=0.55, w_recon=0.30, w_grad=0.15  (Eq. 6)
    CPD           : window=100, fusion (0.45/0.35/0.20),
                    min_affected=6, dedup_distance=200
    RCA weights   : PR=0.30, ANC=0.40, tier=0.20, AM=0.10  (Eq. 8)
    """

    # --- dataset ---
    N_SERVICES   = 50
    N_METRICS    = 6
    N_TIMESTAMPS = 5000
    RANDOM_SEED  = 42

    # --- model architecture (paper §III-B, Eq. 3-4) ---
    HIDDEN_DIM  = 128
    LSTM_LAYERS = 4       # 4-layer BiLSTM encoder
    ATTN_HEADS  = 4       # multi-head self-attention heads
    DROPOUT     = 0.1

    # --- training (paper §IV-A) ---
    BATCH_SIZE    = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-4
    MAX_EPOCHS    = 50
    EARLY_STOP    = 15
    GRAD_CLIP     = 1.0
    LR_T0         = 10    # cosine annealing warm restart T_0

    # --- sequence preparation (paper §III-A / Table I) ---
    WINDOW_SIZE = 60
    STRIDE      = 10      # dense stride → ~23 300 train sequences

    # --- focal loss (paper Eq. 5 / [19]) ---
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.75

    # --- multi-task loss weight (paper Eq. 5) ---
    LAMBDA_CLS = 0.5      # total = recon + 0.5 * FL

    # --- anomaly score fusion (paper Eq. 6) ---
    W_CLF   = 0.55
    W_RECON = 0.30
    W_GRAD  = 0.15

    # --- change-point detection (paper §III-C) ---
    CPD_WINDOW      = 100
    CPD_MIN_AFF     = 6     # min services that must change
    CPD_THRESH_MULT = 0.5   # threshold = mean + 0.5*std
    CPD_DISTANCE    = 120   # peak min distance (timesteps)
    CPD_PROMINENCE  = 0.05
    CPD_DEDUP       = 200   # merge detections within 200 ts

    # --- causal discovery (paper §III-B, Algorithm 1) ---
    CD_PROP_THR = 0.35    # forward propagation strength threshold τ
    CD_ASYM_THR = 0.30    # asymmetry margin Δ
    CD_P_VAL    = 0.03    # Granger p-value threshold
    CD_MI_THR   = 0.20    # mutual information threshold
    CD_MAX_LAG  = 6       # maximum propagation lag

    # --- root cause analysis (paper §III-D, Eq. 8) ---
    RCA_W_PR          = 0.30
    RCA_W_ANC         = 0.40
    RCA_W_TIER        = 0.20
    RCA_W_ANOM        = 0.10
    RCA_PAGERANK_ALPHA = 0.85

    # --- output ---
    OUTPUT_DIR = Path("./outputs")
    FIGURE_DPI = 150

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
