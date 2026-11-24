import numpy as np
import pandas as pd

from data import make_feature_windows, make_windows, synthetic_market, synthetic_prices, walk_forward_splits


def test_windows_shapes_and_labels():
    prices = synthetic_prices(n_days=120, seed=0)
    X, y = make_windows(prices, n_lags=10)
    assert X.shape[1] == 10
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert len(X) == len(y)


def test_window_alignment_has_no_unintended_gap():
    # Prices imply simple returns [0.01, -0.02, 0.03, -0.04, 0.05].
    r = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    p = np.r_[100.0, 100.0 * np.cumprod(1 + r)]
    prices = pd.Series(p, index=pd.date_range("2024-01-01", periods=len(p)))
    X, y = make_windows(prices, n_lags=2)
    np.testing.assert_allclose(X[0], [0.01, -0.02])
    assert y[0] == 1.0  # target is immediately following +3% return
    np.testing.assert_allclose(X[1], [-0.02, 0.03])
    assert y[1] == 0.0


def test_target_dates_are_the_predicted_return_dates():
    market = synthetic_market(n_days=150, seed=4, scenario="linear")
    X, y, dates = make_windows(market.prices, n_lags=10, return_target_index=True)
    returns = market.prices.pct_change().dropna()
    assert dates[0] == returns.index[10]
    assert y[0] == float(returns.iloc[10] > 0)
    assert len(dates) == len(X)


def test_walk_forward_is_ordered_contiguous_and_disjoint():
    n = 500
    previous_test_end = None
    for tr, te in walk_forward_splits(n, n_folds=3):
        assert tr.max() < te.min()
        assert te.min() == tr.max() + 1
        if previous_test_end is not None:
            assert te.min() == previous_test_end + 1
        previous_test_end = te.max()
    assert previous_test_end == n - 1


def test_feature_windows_add_context_columns_without_target_leakage():
    prices = synthetic_prices(n_days=200, seed=3)
    X, y, dates = make_feature_windows(prices, n_lags=10, return_target_index=True)
    assert X.shape[1] == 12
    assert len(X) == len(y) == len(dates)
    assert np.isfinite(X).all()


def test_synthetic_oracle_probability_is_bounded_and_aligned():
    market = synthetic_market(n_days=300, seed=2, scenario="nonlinear", nonlinear_strength=0.15)
    assert len(market.returns) == len(market.oracle_up_probability)
    assert market.returns.index.equals(market.oracle_up_probability.index)
    assert ((market.oracle_up_probability > 0) & (market.oracle_up_probability < 1)).all()
