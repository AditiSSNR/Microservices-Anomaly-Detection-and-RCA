# Sequence preparation, model training, and score computation (paper §III-B).

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression

from .config import Config


# Sequence preparation

def prepare_sequences(
    data:        np.ndarray,
    labels:      np.ndarray,
    window_size: int = Config.WINDOW_SIZE,
    stride:      int = Config.STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window extraction.
    window=60, stride=10  →  ~23 300 training sequences (paper Table I).

    Parameters
    ----------
    data   : (n_services, n_timestamps, n_metrics)
    labels : (n_timestamps, n_services)

    Returns
    -------
    X          (n_seqs, window_size, n_metrics)
    y          (n_seqs,)   binary anomaly label per window
    sids       (n_seqs,)   service index
    stimes     (n_seqs,)   window start timestamp
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
            X[idx]      = data[s, start : start + window_size, :]
            y[idx]      = labels[start : start + window_size, s].max()
            sids[idx]   = s
            stimes[idx] = start
            idx += 1

    return X, y, sids, stimes

# Training loop

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
    AdamW (lr=1e-4, wd=1e-4) + cosine annealing warm restarts (T_0=10,
    T_mult=2) + gradient clipping 1.0 + early stopping patience 15.
    All settings match paper §IV-A.

    Returns the model on CPU with best-validation-loss weights loaded.
    """
    device = Config.device()
    model  = model.to(device)
    pin    = device.type == "cuda"

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.LR_T0, T_mult=2, eta_min=1e-6
    )

    best_val = float("inf")
    patience = 0
    best_sd  = None

    if verbose:
        print(f"\nTraining on {device} ...")

    for epoch in range(1, epochs + 1):
        # --- train ---
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

        # --- validate ---
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
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={tr_loss:.4f} | val={vl_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

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

# Score computation 

def _norm(v: np.ndarray) -> np.ndarray:
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def compute_scores(
    model: nn.Module, X: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """
    Paper Eq. (6) three-component ensemble:
        ŝ = 0.55·norm(clf) + 0.30·norm(recon_err) + 0.15·norm(|Δclf|)

    The gradient term |Δclf| captures temporal transitions in the
    classifier output, rewarding sustained anomalous bursts.
    """
    model.eval()
    clf_list, recon_list = [], []

    for i in range(0, len(X), batch_size):
        bx  = torch.FloatTensor(X[i : i + batch_size])
        out = model(bx)
        clf_list.append(out["anomaly_score"].numpy())
        re = ((out["reconstruction"] - bx) ** 2).mean(dim=(1, 2))
        recon_list.append(re.numpy())

    clf   = np.concatenate(clf_list)
    recon = np.concatenate(recon_list)
    grad  = np.abs(np.diff(clf, prepend=clf[0]))

    return (
        Config.W_CLF   * _norm(clf)
      + Config.W_RECON * _norm(recon)
      + Config.W_GRAD  * _norm(grad)
    )

# Isotonic-regression score calibration 

def calibrate_scores(
    raw_val:   np.ndarray,
    y_val:     np.ndarray,
    raw_test:  np.ndarray,
) -> Tuple[np.ndarray, IsotonicRegression]:
    """
    Fit isotonic regression on validation scores, apply to test scores.
    Threshold search is then performed on the calibrated test scores.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)
    return ir.predict(raw_test), ir

# Full (n_timestamps × n_services) anomaly score matrix

def build_score_matrix(
    model:       nn.Module,
    dataset:     dict,
    window_size: int = Config.WINDOW_SIZE,
    stride:      int = Config.STRIDE,
) -> np.ndarray:
    """
    Back-project per-sequence scores onto the original timeline via max-pooling.
    Returns an (n_timestamps, n_services) matrix used by causal discovery and RCA.
    """
    X_full, _, sids, stimes = prepare_sequences(
        dataset["data"], dataset["anomaly_labels"],
        window_size=window_size, stride=stride,
    )
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
