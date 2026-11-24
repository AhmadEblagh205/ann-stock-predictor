"""Model fitting and walk-forward evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import walk_forward_splits


@dataclass
class WalkForwardResult:
    model: str
    folds: list[dict]
    metrics: dict
    y_true: np.ndarray
    probability: np.ndarray
    baseline_probability: np.ndarray


def _metric_dict(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    y = np.asarray(y, dtype=int)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "positive_rate": float(y.mean()),
    }


def build_model(name: str, seed: int = 0):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=0.002,
                max_iter=1000,
                random_state=seed,
            ),
        )
    raise ValueError("model name must be 'logistic' or 'mlp'")


def evaluate_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 3,
    min_train_frac: float = 0.4,
    seed: int = 0,
) -> WalkForwardResult:
    """Evaluate a fixed model specification on expanding OOS folds.

    The probabilistic baseline is the historical positive rate estimated only
    from each fold's training observations.  This replaces V1's test-fold
    majority baseline, which used the future labels to choose the benchmark.
    """
    all_y, all_p, all_base = [], [], []
    folds = []
    for fold_id, (tr, te) in enumerate(
        walk_forward_splits(len(X), n_folds=n_folds, min_train_frac=min_train_frac), start=1
    ):
        model = build_model(model_name, seed=seed + fold_id)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        base_rate = float(np.mean(y[tr]))
        p_base = np.full(len(te), base_rate, dtype=float)
        fm = _metric_dict(y[te], p)
        bm = _metric_dict(y[te], p_base)
        always_up_accuracy = float(np.mean(y[te]))
        folds.append(
            {
                "fold": fold_id,
                "train_size": int(len(tr)),
                "test_size": int(len(te)),
                "train_positive_rate": base_rate,
                "test_positive_rate": float(np.mean(y[te])),
                "model": fm,
                "historical_base_rate": bm,
                "always_up_accuracy": always_up_accuracy,
            }
        )
        all_y.append(y[te])
        all_p.append(p)
        all_base.append(p_base)

    yy = np.concatenate(all_y).astype(int)
    pp = np.concatenate(all_p)
    bb = np.concatenate(all_base)
    return WalkForwardResult(
        model=model_name,
        folds=folds,
        metrics={
            "model": _metric_dict(yy, pp),
            "historical_base_rate": _metric_dict(yy, bb),
            "always_up_accuracy": float(np.mean(yy)),
        },
        y_true=yy,
        probability=pp,
        baseline_probability=bb,
    )
