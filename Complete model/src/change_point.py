# Multi-detector change point fusion (paper §III-C).

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

from .config import Config


class MultiServiceChangePointDetector:
    """
    Three-detector fusion (paper §III-C):

    Detector A (CUSUM)
        Per-service mean-shift and variance-ratio statistics in a sliding
        window of half-width W=100. Counts services with significant change
        weighted by mean change magnitude.

    Detector B (Energy ratio)
        E_ratio(t) = log(E_after / E_before),  E = mean(x²).
        Score = mean|ratio| × count(|ratio| > 0.2).

    Detector C (KL divergence)
        KL(P||Q) estimated from 20-bin histograms, averaged over 20 randomly
        sampled services.

    Fusion weights (paper §III-C):
        fused = 0.45·A + 0.35·B + 0.20·C

    Peak detection:
        height = mean + 0.5·std  (clipped to [0.04, 0.55])
        distance = 120 ts,  prominence = 0.05

    Confirmation:
        Peaks confirmed if ≥ min_affected=6 services exceed per-service
        threshold;  detections within 200 ts are merged (paper §III-C).
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

    def detect(
        self, data: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Parameters
        ----------
        data : (n_services, n_timestamps, n_metrics)

        Returns
        -------
        detected_timestamps  list[int]
        fused_scores         (n_timestamps,)
        """
        ns, nt, nm = data.shape
        agg        = data.mean(axis=-1)          # (ns, nt)

        A = self._detector_A(agg)
        B = self._detector_B(agg)
        C = self._detector_C(agg)

        fused  = 0.45 * A + 0.35 * B + 0.20 * C
        thresh = fused.mean() + self.thresh_mult * fused.std()
        thresh = np.clip(thresh, 0.04, 0.55)

        peaks, _ = find_peaks(
            fused,
            height     = thresh,
            distance   = self.distance,
            prominence = self.prominence,
        )

        confirmed = [
            p for p in peaks
            if self._count_affected(agg, p) >= self.min_affected
        ]

        # Merge detections within 200 ts
        deduped = []
        for p in sorted(confirmed):
            if not any(abs(p - q) < self.dedup for q in deduped):
                deduped.append(p)

        return deduped, fused

    def evaluate(
        self,
        data:      np.ndarray,
        true_cps:  List[int],
        tolerance: int = 150,
    ) -> Dict:
        """
        Match detected change points to true ones within ±tolerance ts.

        Returns dict with precision, recall, f1, n_detected, n_correct,
        n_true, scores, detected.
        """
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
            "precision":  prec,
            "recall":     rec,
            "f1":         f1,
            "n_detected": len(detected),
            "n_correct":  n_correct,
            "n_true":     len(true_cps),
            "scores":     scores,
            "detected":   detected,
        }

    # Internal detectors

    def _detector_A(self, agg: np.ndarray) -> np.ndarray:
        """CUSUM: count services with significant mean-shift or variance change."""
        ns, nt = agg.shape
        w      = self.window
        scores = np.zeros(nt)

        cum    = np.cumsum(agg, axis=1)
        cum_sq = np.cumsum(agg ** 2, axis=1)
        cp_    = np.concatenate([np.zeros((ns, 1)), cum],    axis=1)
        cps_   = np.concatenate([np.zeros((ns, 1)), cum_sq], axis=1)

        def _stats(c, cs, s, e):
            n   = e - s
            sv  = c[:, e]  - c[:, s]
            s2  = cs[:, e] - cs[:, s]
            mu  = sv / n
            var = np.maximum(s2 / n - mu ** 2, 0)
            return mu, np.sqrt(var) + 1e-8

        for t in range(w, nt - w):
            bm, bs  = _stats(cp_, cps_, t - w, t)
            am, as_ = _stats(cp_, cps_, t,     t + w)
            mc      = np.abs(am - bm) / bs
            vr      = np.abs(np.log(as_ / bs))
            hit     = (mc / bs > 0.3) | (vr > 0.3)
            nc      = hit.sum()
            if nc >= 5:
                scores[t] = nc * (mc + vr)[hit].mean()

        return scores / (scores.max() + 1e-8)

    def _detector_B(self, agg: np.ndarray) -> np.ndarray:
        """Energy-ratio detector."""
        ns, nt = agg.shape
        w      = self.window
        scores = np.zeros(nt)

        for t in range(w, nt - w):
            before = agg[:, t - w : t]
            after  = agg[:, t :     t + w]
            e_b    = (before ** 2).mean(axis=1) + 1e-8
            e_a    = (after  ** 2).mean(axis=1) + 1e-8
            ratio  = np.log(e_a / e_b)
            scores[t] = (
                np.abs(ratio).mean() * (np.abs(ratio) > 0.2).sum()
            )

        return scores / (scores.max() + 1e-8)

    def _detector_C(self, agg: np.ndarray, bins: int = 20) -> np.ndarray:
        """KL-divergence on 20 sampled services."""
        ns, nt = agg.shape
        w      = self.window
        scores = np.zeros(nt)
        sample = np.random.choice(ns, size=min(20, ns), replace=False)

        for t in range(w, nt - w):
            kl_sum = 0.0
            for s in sample:
                b  = agg[s, t - w : t]
                a  = agg[s, t :     t + w]
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

    def _count_affected(self, agg: np.ndarray, t: int) -> int:
        """Count services whose statistics change significantly at timestamp t."""
        ns, nt = agg.shape
        w      = self.window
        if t < w or t >= nt - w:
            return 0
        before = agg[:, t - w : t]
        after  = agg[:, t :     t + w]
        mc     = np.abs(after.mean(1) - before.mean(1))
        sb     = before.std(1) + 1e-8
        sa     = after.std(1)  + 1e-8
        vr     = np.abs(np.log(sa / sb))
        return int(((mc / sb > 0.25) | (vr > 0.25)).sum())
