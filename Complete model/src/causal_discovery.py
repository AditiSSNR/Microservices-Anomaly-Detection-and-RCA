# Propagation-based causal discovery pipeline

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .config import Config

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


class AnomalyPropagationCausalDiscovery:
    """
    Five-stage causal discovery pipeline (paper §III-B, Algorithm 1):

    Stage 1 — Propagation mining
        Forward propagation strength (Eq. 7):
            P(i→j) = |{t ∈ E_i : ∃t' ∈ [t+1, t+L] s.t. t' ∈ E_j}| / |E_i|
        Edge placed if P(i→j) > τ = 0.35.

    Stage 2 — Tier filter
        Removes backward edges and edges skipping >2 tiers.

    Stage 3 — Asymmetry test
        Keeps edge i→j only if P(i,j) > P(j,i) + Δ,  Δ = 0.30.

    Stage 4 — Granger causality gate
        Retains edge if min p-value over lags 1-5 < 0.03.

    Stage 5 — Mutual information gate
        Retains edge if max lagged MI (lags 1-5) > 0.20.
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

    def discover_causal_graph(
        self,
        data:                    np.ndarray,
        anomaly_score_threshold: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        data                    : (n_services, n_timestamps, n_metrics)
        anomaly_score_threshold : threshold applied to bilstm_anomaly_scores

        Returns
        -------
        adjacency_matrix  (n_services, n_services)  binary
        edge_strengths    (n_services, n_services)  float propagation strengths
        """
        events          = self._detect_anomaly_events(anomaly_score_threshold)
        raw, strengths  = self._mine_propagation(events)
        tier_f          = self._tier_filter(raw)
        asym_f          = self._asymmetry_filter(tier_f, strengths)
        granger_f       = self._granger_filter(asym_f, data)
        final           = self._mi_filter(granger_f, data)
        return final, strengths

    # Stage 1 helpers

    def _detect_anomaly_events(self, threshold: float) -> Dict[int, set]:
        if self.bilstm_anomaly_scores is None:
            return {s: set() for s in range(self.n_services)}
        nt, ns = self.bilstm_anomaly_scores.shape
        events = {}
        for s in range(ns):
            idx = np.where(self.bilstm_anomaly_scores[:, s] > threshold)[0]
            events[s] = set(idx[idx < nt].tolist())
        return events

    def _mine_propagation(
        self,
        events:  Dict[int, set],
        max_lag: int = Config.CD_MAX_LAG,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised forward propagation strength (Eq. 7)."""
        ns  = self.n_services
        nt  = Config.N_TIMESTAMPS
        mat = np.zeros((ns, nt), dtype=bool)
        for s, ev in events.items():
            if ev:
                idx = np.array(sorted(ev), dtype=int)
                idx = idx[idx < nt]
                mat[s, idx] = True

        counts    = mat.sum(axis=1)
        graph     = np.zeros((ns, ns))
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

    # Stage 2 — Tier filter

    def _tier_filter(self, graph: np.ndarray) -> np.ndarray:
        """Remove backward edges and those skipping >2 tiers."""
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

    # Stage 3 — Asymmetry test

    def _asymmetry_filter(
        self, graph: np.ndarray, strengths: np.ndarray
    ) -> np.ndarray:
        """Keep i→j only if P(i,j) > P(j,i) + Δ  (Δ=0.30)."""
        out  = graph.copy()
        mask = (graph == 1) & (strengths.T >= strengths - Config.CD_ASYM_THR)
        out[mask] = 0
        return out
    # Stage 4 — Granger causality

    def _granger_filter(
        self, graph: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Retain edge if min p-value over lags 1-5 < 0.03."""
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
                    ok  = any(
                        res[lag][0]["ssr_ftest"][1] < Config.CD_P_VAL
                        for lag in range(1, 6)
                    )
                    if not ok:
                        out[i, j] = 0
                except Exception:
                    out[i, j] = 0
        return out
    # Stage 5 — Mutual information

    def _mi_filter(
        self, graph: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Retain edge if max lagged MI (lags 1-5) > 0.20."""
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
                            random_state=42,
                        )[0]
                        for lag in range(1, 6)
                        if len(agg[i]) > lag + 100
                    )
                    if max_mi < Config.CD_MI_THR:
                        out[i, j] = 0
                except Exception:
                    pass
        return out
