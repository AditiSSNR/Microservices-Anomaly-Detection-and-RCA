# Graph-based root cause localizer (paper §III-D, Eq. 8-9).

from typing import Dict, List

import numpy as np
import networkx as nx

from .config import Config


class GraphBasedRootCauseAnalyzer:
    """
    Multi-signal root cause localizer (paper §III-D, Eq. 8):

        score(s) = 0.30·PR(s) + 0.40·ANC(s) + 0.20·tier(s) + 0.10·AM(s)

    PR(s)
        Personalized PageRank on the reversed dependency graph with
        personalization vector proportional to anomaly scores.
        Damping factor α = 0.85.

    ANC(s)   — Eq. (9)
        ANC(v) = Σ_{a ∈ anomalous} AM(a) / (dist(v,a)^0.5 + 1)
        Rewards nodes whose downstream descendants are anomalous.

    tier(s)
        Architectural prior: frontend=1.0, api=0.8, business=0.6,
        data=0.3, database=0.1.

    AM(s)
        Per-service anomaly magnitude = 90th-percentile of non-zero
        values in the score-matrix column (paper §III-D).
    """

    def __init__(
        self,
        causal_graph:  nx.DiGraph,
        service_tiers: Dict[str, List[int]],
    ):
        self.causal_graph  = causal_graph
        self.service_tiers = service_tiers

    def localize_root_cause(
        self,
        anomaly_scores:   np.ndarray,
        causal_strengths: np.ndarray,
        top_k: int = 15,
    ) -> Dict:
        """
        Parameters
        ----------
        anomaly_scores   : (n_services,)  per-service anomaly magnitude AM
        causal_strengths : (n_services, n_services)  from causal discovery
        top_k            : number of candidates to return

        Returns
        -------
        dict with keys: root_causes (list[int]), scores (list[float]),
                        anomalous_services (list[int])
        """
        n         = len(anomaly_scores)
        anomalous = np.where(anomaly_scores > 0.5)[0]

        if len(anomalous) == 0:
            return {
                "root_causes":        [],
                "scores":             [],
                "anomalous_services": [],
            }

        pr   = self._pagerank(anomalous, anomaly_scores)
        anc  = self._ancestor(anomalous, anomaly_scores)
        tier = np.array([self._tier_weight(i) for i in range(n)])

        # Eq. (8)
        combined = (
            Config.RCA_W_PR   * pr
          + Config.RCA_W_ANC  * anc
          + Config.RCA_W_TIER * tier
          + Config.RCA_W_ANOM * anomaly_scores
        )
        ranked = np.argsort(combined)[-top_k:][::-1]

        return {
            "root_causes":        ranked.tolist(),
            "scores":             combined[ranked].tolist(),
            "anomalous_services": anomalous.tolist(),
        }

    def _pagerank(
        self, anomalous: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Personalized PageRank on reversed graph (paper §III-D)."""
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

    def _ancestor(
        self, anomalous: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """
        Eq. (9):
            ANC(v) = Σ_{a∈anomalous} AM(a) / (dist(v,a)^0.5 + 1)
        """
        n      = len(scores)
        result = np.zeros(n)
        for a in anomalous:
            try:
                lengths = dict(
                    nx.single_target_shortest_path_length(
                        self.causal_graph, a
                    )
                )
                for anc, dist in lengths.items():
                    if anc != a and dist > 0:
                        result[anc] += scores[a] / (dist ** 0.5 + 1)
            except Exception:
                pass
        return result

    def _tier_weight(self, service: int) -> float:
        """Architectural tier prior (paper §III-D)."""
        weights = {
            "frontend": 1.0,
            "api":      0.8,
            "business": 0.6,
            "data":     0.3,
            "database": 0.1,
        }
        for tier, svcs in self.service_tiers.items():
            if service in svcs:
                return weights.get(tier, 0.5)
        return 0.5
