
#import necessary libraries
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config


class Visualizer:
    """
    Generates three figures:
      1. change_points.png        — service metrics + fused CPD score
      2. performance_summary.png  — 2×2 bar chart of all four pipeline stages
      3. anomaly_distribution.png — tier-level time series + heatmap
    """

    def __init__(self, output_dir: Path = Config.OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-paper")

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

    def plot_change_points(
        self,
        data:      np.ndarray,
        cp_scores: np.ndarray,
        detected:  List[int],
        true_cps:  List[int],
    ):
        """Two-panel: sample service metrics (top) + fused CPD score (bottom)."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        agg = data.mean(axis=-1)[:5]
        for i in range(5):
            axes[0].plot(agg[i], alpha=0.6, lw=1, label=f"S{i}")
        for t in true_cps:
            axes[0].axvline(t, color="red", ls="--", alpha=0.5, lw=1.5)
        axes[0].set_title(
            "Service Metrics + True Change Points", fontweight="bold"
        )
        axes[0].legend(fontsize=8);  axes[0].grid(alpha=0.3)

        axes[1].plot(cp_scores, color="navy", lw=1.2, label="Fused CPD Score")
        for d in detected:
            axes[1].axvline(
                d, color="green", lw=1.5, alpha=0.7,
                label="Detected" if d == detected[0] else "",
            )
        for t in true_cps:
            axes[1].axvline(
                t, color="red", ls="--", alpha=0.5, lw=1.5,
                label="True" if t == true_cps[0] else "",
            )
        axes[1].set_title("Fused Change Point Score", fontweight="bold")
        axes[1].legend(fontsize=8);  axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "change_points.png",
            dpi=Config.FIGURE_DPI, bbox_inches="tight",
        )
        plt.close()
        print("  ✓ change_points.png")

    def plot_performance_summary(self, results: Dict):
        """2×2 bar chart for all four pipeline components."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        def _bar(ax, data: Dict, title: str, ylim: float = 1.0):
            keys   = list(data.keys())
            vals   = [data[k] for k in keys]
            colors = [
                "#45B7D1" if v >= 0.9 else
                "#4ECDC4" if v >= 0.7 else
                "#FF6B6B"
                for v in vals
            ]
            bars = ax.bar(keys, vals, color=colors,
                          edgecolor="black", alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=9,
                )
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
             {
                 "Hit@1":  rca["hit_at_1"],
                 "Hit@3":  rca["hit_at_3"],
                 "Hit@5":  rca["hit_at_5"],
                 "Hit@10": rca["hit_at_10"],
             },
             "Root Cause Analysis", ylim=1.2)

        plt.suptitle("System Performance Summary",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_summary.png",
            dpi=Config.FIGURE_DPI, bbox_inches="tight",
        )
        plt.close()
        print("  ✓ performance_summary.png")

    def plot_anomaly_distribution(self, dataset: Dict):
        """Tier-level smoothed anomaly counts + full service×time heatmap."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        labels    = dataset["anomaly_labels"]

        for tier_name, svcs in dataset["tiers"].items():
            counts = labels[:, svcs].sum(axis=1)
            sm     = np.convolve(counts, np.ones(50) / 50, mode="same")
            axes[0].plot(sm, label=tier_name.capitalize(), lw=1.5)
        axes[0].set_title(
            "Anomaly Distribution by Tier", fontweight="bold"
        )
        axes[0].legend();  axes[0].grid(alpha=0.3)

        ds = 10
        im = axes[1].imshow(
            labels[::ds].T, aspect="auto", cmap="YlOrRd"
        )
        axes[1].set_title(
            "Anomaly Heatmap (Services × Time)", fontweight="bold"
        )
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "anomaly_distribution.png",
            dpi=Config.FIGURE_DPI, bbox_inches="tight",
        )
        plt.close()
        print("  ✓ anomaly_distribution.png")