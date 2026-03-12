# Synthetic microservices dataset (paper §III-A, Eq. 1-2).

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from .config import Config


class MicroservicesDataGenerator:
    """
    Generates a synthetic 50-service dataset matching the paper:

    Baseline (Eq. 1):
        x(t) = 50 + 20·sin(2π t/288) + 10·sin(2π t/2016) + 5·sin(2π t/12)

    Cascade failures (Eq. 2):
        magnitude(d) = (3.5 + U[0,3]) · (1 – 0.05·d)
        75 scenarios; root causes from frontend/api/business tiers.

    Change points:
        8 coordinated events (mean-shift / variance-change / both),
        each affecting 15-30 services.

    Five-tier topology: frontend → api → business → data → database
    """

    def __init__(
        self,
        n_services:   int = Config.N_SERVICES,
        n_metrics:    int = Config.N_METRICS,
        n_timestamps: int = Config.N_TIMESTAMPS,
        seed:         int = Config.RANDOM_SEED,
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
        self.metric_names = [
            "cpu", "memory", "latency", "throughput", "errors", "connections"
        ]

    def _create_causal_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_services))

        for fe in self.tiers["frontend"]:
            for t in np.random.choice(
                self.tiers["api"], size=np.random.randint(2, 4), replace=False
            ):
                G.add_edge(fe, t)

        for api in self.tiers["api"]:
            for t in np.random.choice(
                self.tiers["business"],
                size=np.random.randint(3, 6),
                replace=False,
            ):
                G.add_edge(api, t)

        for bus in self.tiers["business"]:
            if np.random.rand() > 0.2:
                for t in np.random.choice(
                    self.tiers["data"],
                    size=np.random.randint(1, 3),
                    replace=False,
                ):
                    G.add_edge(bus, t)

        for dat in self.tiers["data"]:
            for t in np.random.choice(
                self.tiers["database"],
                size=np.random.randint(1, 2),
                replace=False,
            ):
                G.add_edge(dat, t)

        return G

    def generate(self) -> Dict:
        """
        Returns
        -------
        dict with keys:
            data            (n_services, n_timestamps, n_metrics)  float32
            edge_index      (2, n_edges)                           int
            anomaly_labels  (n_timestamps, n_services)             float32
            root_causes     list of dicts
            change_points   list of dicts
            causal_graph    nx.DiGraph
            metric_names    list[str]
            tiers           dict[str, list[int]]
        """
        # Eq. (1) — multi-scale sinusoidal baseline
        t    = np.arange(self.n_timestamps)
        base = (
            50
            + 20 * np.sin(2 * np.pi * t / 288)
            + 10 * np.sin(2 * np.pi * t / (288 * 7))
            +  5 * np.sin(2 * np.pi * t / 12)
        )
        data = base[np.newaxis, :, np.newaxis] + np.random.randn(
            self.n_services, self.n_timestamps, self.n_metrics
        ) * 3

        anomaly_labels, root_causes = self._inject_cascade_failures(data)
        change_points               = self._inject_change_points(data)

        # z-score normalisation (joint service-time axis)
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

    def _inject_cascade_failures(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Eq. (2): magnitude(d) = (3.5 + U[0,3]) · (1 – 0.05·d)"""
        anomaly_labels = np.zeros((self.n_timestamps, self.n_services))
        root_causes    = []

        for scenario_id in range(75):
            if scenario_id < 15:
                root = np.random.choice(
                    self.tiers["frontend"] + self.tiers["api"]
                )
            else:
                root = np.random.choice(self.tiers["business"])

            start    = np.random.randint(500, self.n_timestamps - 500)
            affected = list(nx.descendants(self.causal_graph, root))
            affected = [root] + affected[: min(len(affected), 15)]

            for i, service in enumerate(affected):
                delay    = i * np.random.randint(2, 6)
                duration = np.random.randint(40, 100)
                if start + delay + duration >= self.n_timestamps:
                    continue
                magnitude = (3.5 + np.random.rand() * 3.0) * (1 - i * 0.05)
                sl = slice(start + delay, start + delay + duration)
                data[service, sl, :]        *= magnitude
                anomaly_labels[sl, service]  = 1

            root_causes.append({
                "scenario_id":       scenario_id,
                "root_cause":        root,
                "affected_services": affected,
                "timestamp":         start,
            })

        return anomaly_labels, root_causes

    def _inject_change_points(self, data: np.ndarray) -> List[Dict]:
        """8 coordinated change points, 15-30 services each."""
        change_points = []
        for cp_id in range(8):
            timestamp  = 800 + cp_id * 500
            n_affected = np.random.randint(15, 30)
            affected   = np.random.choice(
                self.n_services, size=n_affected, replace=False
            )
            for service in affected:
                kind = np.random.choice(
                    ["mean_shift", "variance_change", "both"]
                )
                if kind in ("mean_shift", "both"):
                    data[service, timestamp:, :] *= np.random.uniform(2.0, 3.0)
                if kind in ("variance_change", "both"):
                    mv  = data[service, timestamp:, :].mean(
                        axis=0, keepdims=True
                    )
                    dev = data[service, timestamp:, :] - mv
                    data[service, timestamp:, :] = (
                        mv + dev * np.random.uniform(2.5, 5.0)
                    )
            change_points.append({
                "timestamp":         timestamp,
                "affected_services": affected.tolist(),
                "n_affected":        n_affected,
            })
        return change_points
