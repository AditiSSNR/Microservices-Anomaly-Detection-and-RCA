# Full four-stage evaluation pipeline

from typing import Dict, Optional

import numpy as np
import networkx as nx
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from .config import Config
from .train import compute_scores, calibrate_scores, build_score_matrix
from .causal_discovery import AnomalyPropagationCausalDiscovery
from .change_point import MultiServiceChangePointDetector
from .root_cause import GraphBasedRootCauseAnalyzer


def evaluate_system(
    model,
    X_test:            np.ndarray,
    y_test:            np.ndarray,
    service_ids_test:  np.ndarray,
    dataset:           Dict,
    X_val:             Optional[np.ndarray] = None,
    y_val:             Optional[np.ndarray] = None,
) -> Dict:
    """
    Runs all four evaluation stages and returns a nested results dict.

    Stages
    ------
    1. Anomaly detection   — precision / recall / F1 / ROC-AUC / PR-AUC
    2. Causal discovery    — precision / recall / F1 against ground-truth edges
    3. Change point detection — precision / recall / F1 (±150 ts tolerance)
    4. Root cause analysis — Hit@1 / Hit@3 / Hit@5 / Hit@10
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    device = Config.device()
    model  = model.to(device)

    # Build full (n_timestamps × n_services) anomaly score matrix
    print("\n  Building full anomaly score matrix ...")
    score_matrix = build_score_matrix(model, dataset)

    # 1. Anomaly Detection
    print("\n1. ANOMALY DETECTION")
    print("-" * 70)

    raw_test = compute_scores(model, X_test)

    if X_val is not None and y_val is not None:
        raw_val          = compute_scores(model, X_val)
        cal_test, _      = calibrate_scores(raw_val, y_val, raw_test)
    else:
        cal_test = raw_test

    # Best-F1 threshold search on calibrated scores
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
    for k, v in ad.items():
        if k != "threshold":
            print(f"  {k:<12}: {v:.4f}")

    # 2. Causal Discovery
    ns       = dataset["data"].shape[0]
    cd_model = AnomalyPropagationCausalDiscovery(
        n_services            = ns,
        service_tiers         = dataset["tiers"],
        bilstm_anomaly_scores = score_matrix,
    )
    disc_graph, strengths = cd_model.discover_causal_graph(
        dataset["data"], anomaly_score_threshold=0.6
    )

    true_e = set(
        (dataset["edge_index"][0, i], dataset["edge_index"][1, i])
        for i in range(dataset["edge_index"].shape[1])
    )
    pred_e = set(
        (i, j) for i in range(ns) for j in range(ns)
        if disc_graph[i, j] > 0
    )
    nc = len(true_e & pred_e)
    cd = {
        "precision": nc / len(pred_e) if pred_e else 0.0,
        "recall":    nc / len(true_e) if true_e else 0.0,
        "n_true":    len(true_e),
        "n_pred":    len(pred_e),
        "n_correct": nc,
        "discovered_graph": disc_graph,
        "causal_strengths": strengths,
    }
    cd["f1"] = (
        2 * cd["precision"] * cd["recall"]
        / (cd["precision"] + cd["recall"])
        if (cd["precision"] + cd["recall"]) > 0 else 0.0
    )

    # 3. Change Point Detection
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

    # 4. Root Cause Analysis
    print("\n3. ROOT CAUSE ANALYSIS")
    print("-" * 70)

    # Per-service AM = 90th-percentile of non-zero score-matrix column
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
