import numpy as np

from data import make_windows, synthetic_market
from modeling import build_model, evaluate_walk_forward


def test_historical_base_rate_uses_training_labels_only():
    market = synthetic_market(n_days=500, seed=0, scenario="linear")
    X, y = make_windows(market.prices, 2)
    res = evaluate_walk_forward(X, y, "logistic")
    first = res.folds[0]
    n_train = first["train_size"]
    assert np.isclose(first["train_positive_rate"], y[:n_train].mean())


def test_models_return_probabilities():
    market = synthetic_market(n_days=1600, seed=1, scenario="linear")
    X, y = make_windows(market.prices, 2)
    for name in ("logistic", "mlp"):
        model = build_model(name, seed=1)
        model.fit(X[:1000], y[:1000])
        p = model.predict_proba(X[1000:])[:, 1]
        assert np.all((p >= 0) & (p <= 1))


def test_future_label_changes_do_not_alter_training_base_rate_for_first_fold():
    market = synthetic_market(n_days=500, seed=2, scenario="linear")
    X, y = make_windows(market.prices, 2)
    res1 = evaluate_walk_forward(X, y, "logistic")
    n_train = res1.folds[0]["train_size"]
    y2 = y.copy()
    y2[n_train:] = 1 - y2[n_train:]
    res2 = evaluate_walk_forward(X, y2, "logistic")
    assert res1.folds[0]["train_positive_rate"] == res2.folds[0]["train_positive_rate"]
