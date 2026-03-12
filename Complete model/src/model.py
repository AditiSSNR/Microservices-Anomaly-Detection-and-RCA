# BiLSTM + Multi-Head Attention anomaly detector (paper §III-B, Fig. 2).

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

# Focal loss (Lin et al. [19], paper Eq. 5)

def focal_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    gamma:  float = Config.FOCAL_GAMMA,
    alpha:  float = Config.FOCAL_ALPHA,
) -> torch.Tensor:
    """
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    γ=2.0, α=0.75 — downweights easy negatives, focuses on hard anomalous
    windows (paper §III-B, Eq. 5).
    """
    bce     = F.binary_cross_entropy(pred, target, reduction="none")
    p_t     = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()

# BiLSTM anomaly detector

class BiLSTMAnomalyDetector(nn.Module):
    """
    Architecture (paper §III-B, Fig. 2):

    1. 4-layer bidirectional LSTM encoder (hidden=128)     — Eq. (3)
    2. Layer normalisation on encoder output
    3. 4-head multi-head self-attention                    — Eq. (4)
       context c = mean-pool of attention output
    4. Single-layer LSTM decoder → linear reconstruction head
    5. 3-layer GELU MLP classifier head → σ(·) anomaly score

    Joint loss (Eq. 5):
        L = L_MSE + 0.5 · FL(γ=2, α=0.75)
    """

    def __init__(
        self,
        n_metrics:  int   = Config.N_METRICS,
        hidden_dim: int   = Config.HIDDEN_DIM,
        n_layers:   int   = Config.LSTM_LAYERS,
        n_heads:    int   = Config.ATTN_HEADS,
        dropout:    float = Config.DROPOUT,
    ):
        super().__init__()

        # --- Encoder (Eq. 3) ---
        self.encoder = nn.LSTM(
            input_size    = n_metrics,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # --- Multi-head self-attention (Eq. 4) ---
        self.mha = nn.MultiheadAttention(
            embed_dim   = hidden_dim * 2,
            num_heads   = n_heads,
            dropout     = dropout,
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
                p.data[n // 4 : n // 2].fill_(1.0)  # forget-gate bias = 1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_metrics)

        Returns
        -------
        dict with keys: reconstruction, anomaly_score, attention, encoding
        """
        enc, _   = self.encoder(x)           # (B, T, 2H)
        enc      = self.layer_norm(enc)

        attn_out, attn_w = self.mha(enc, enc, enc)
        context  = attn_out.mean(dim=1)      # (B, 2H)

        dec_in   = context.unsqueeze(1).expand(-1, x.size(1), -1)
        dec, _   = self.decoder(dec_in)
        recon    = self.output(dec)           # (B, T, M)

        score    = self.classifier(context).squeeze(-1)  # (B,)

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
        """
        Eq. (5): L = L_MSE(recon, x) + λ_cls · FL(ŝ, y)
        λ_cls = 0.5, γ = 2.0, α = 0.75
        """
        recon_loss = F.mse_loss(outputs["reconstruction"], inputs)
        cls_loss   = focal_loss(outputs["anomaly_score"], labels)
        total      = recon_loss + Config.LAMBDA_CLS * cls_loss
        return {
            "total":          total,
            "reconstruction": recon_loss,
            "classification": cls_loss,
        }
