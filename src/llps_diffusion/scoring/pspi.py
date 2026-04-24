from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_crl(sym_score_ab: float, sym_score_ba: float) -> float:
    return (sym_score_ab + sym_score_ba) / 2.0


def compute_cfg_gap(cond_logits: np.ndarray, uncond_logits: np.ndarray) -> float:
    return float(np.linalg.norm(cond_logits - uncond_logits, ord=2))


def fit_pspi_fuser(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500)
    clf.fit(x, y)
    return clf


def predict_pspi(clf: LogisticRegression, feat: np.ndarray) -> float:
    prob = clf.predict_proba(feat.reshape(1, -1))[0, 1]
    return float(prob)
