"""Hozumi & Wei's 5-NN classification protocol (25M1786957 / arXiv:2412.20202, sec 2.2):
exclude families with < 15 total sequences, stratified 5-fold CV repeated over 30 random
seeds, 5-NN MAJORITY VOTE, macro-averaged ACC/BA/F1/AUC-ROC/Recall/Precision.

Copied verbatim from psd_v2/paper_5nn_eval.py -- package-agnostic, operates on any full
pairwise distance matrix + labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def make_splits(y, seed, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))


def filter_min_family_size(y, min_size=15):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= min_size]
    return np.isin(y, keep_classes)


def knn_vote_proba(D_block, y_ref, k, n_classes):
    n_ref = D_block.shape[1]
    k_eff = min(k, n_ref)
    idx = np.argpartition(D_block, kth=k_eff - 1, axis=1)[:, :k_eff]
    neighbor_labels = y_ref[idx]
    proba = np.zeros((D_block.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        proba[:, c] = (neighbor_labels == c).sum(axis=1) / k_eff
    return proba


def paper_style_metrics(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    ba = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    return {"ACC": acc, "BA": ba, "F1": f1, "AUC_ROC": auc, "Recall": ba, "Precision": prec}


def paper_5nn_eval(D_full, y_full, k=5, min_family_size=15, n_splits=5, n_seeds=30, seed0=0):
    y_full = np.asarray(y_full)
    keep = filter_min_family_size(y_full, min_family_size)
    D = D_full[np.ix_(keep, keep)]
    classes, y = np.unique(y_full[keep], return_inverse=True)
    n_classes = len(classes)
    n = len(y)

    per_seed_metrics = []
    for s in range(n_seeds):
        folds = make_splits(y, seed=seed0 + s, n_splits=n_splits)
        y_pred_all = np.empty(n, dtype=np.int64)
        y_proba_all = np.empty((n, n_classes), dtype=np.float64)
        for train_idx, test_idx in folds:
            D_block = D[np.ix_(test_idx, train_idx)]
            proba = knn_vote_proba(D_block, y[train_idx], k, n_classes)
            y_proba_all[test_idx] = proba
            y_pred_all[test_idx] = np.argmax(proba, axis=1)
        per_seed_metrics.append(paper_style_metrics(y, y_pred_all, y_proba_all))

    mean_metrics = {key: float(np.mean([m[key] for m in per_seed_metrics]))
                    for key in per_seed_metrics[0]}
    return {"per_seed": per_seed_metrics, "mean": mean_metrics,
            "n_sequences_used": int(n), "n_classes_used": int(n_classes)}
