"""Data loading, synthetic market generation and leak-free window construction."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticMarket:
    """Controlled synthetic market with known one-step direction probabilities."""

    prices: pd.Series
    returns: pd.Series
    oracle_up_probability: pd.Series
    conditional_mean: pd.Series
    noise_sigma: float
    scenario: str


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in x])


def load_prices(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    csv: str | None = None,
    synthetic: bool = False,
    seed: int = 2,
) -> pd.Series:
    """Load an adjusted-close series or the controlled synthetic fixture.

    CSV mode expects one numeric price column with dates in the first column.
    yfinance is imported lazily and is never required by CI.
    """
    if synthetic:
        return synthetic_market(seed=seed).prices
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        if df.shape[1] < 1:
            raise ValueError("CSV must contain at least one price column")
        s = pd.to_numeric(df.iloc[:, 0], errors="raise").sort_index()
        return _validate_prices(s, min_length=30)
    import yfinance as yf  # lazy optional dependency

    out = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if out.empty or "Close" not in out:
        raise ValueError(f"No Close prices returned for {ticker!r}")
    return _validate_prices(out["Close"].squeeze().dropna(), min_length=30)


def _validate_prices(prices: pd.Series, min_length: int = 2) -> pd.Series:
    prices = pd.Series(prices, copy=True).astype(float).dropna().sort_index()
    if len(prices) < min_length:
        raise ValueError(f"At least {min_length} prices are required")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Price index must be unique and increasing")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")
    return prices


def synthetic_market(
    n_days: int = 2500,
    seed: int = 2,
    scenario: str = "linear",
    phi: float = 0.12,
    nonlinear_strength: float = 0.0,
    drift: float = 0.00015,
    sigma: float = 0.010,
) -> SyntheticMarket:
    """Generate a controlled market with known P(next return > 0 | history).

    Parameters
    ----------
    scenario:
        ``no_signal``: iid Gaussian returns with constant drift.
        ``linear``: AR(1)-style conditional mean ``drift + phi*r[t-1]``.
        ``nonlinear``: the linear term plus a bounded two-lag interaction.

    The process is deliberately synthetic. It is designed to test whether an
    evaluation pipeline can recover known predictability, not to mimic a real
    market or establish an investment edge.
    """
    if n_days < 100:
        raise ValueError("n_days must be at least 100")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if scenario not in {"no_signal", "linear", "nonlinear"}:
        raise ValueError("scenario must be no_signal, linear or nonlinear")

    rng = np.random.default_rng(seed)
    n_ret = n_days - 1
    r = np.zeros(n_ret, dtype=float)
    mu = np.zeros(n_ret, dtype=float)
    eps = rng.normal(0.0, sigma, n_ret)

    for t in range(n_ret):
        if scenario == "no_signal" or t == 0:
            m = drift
        else:
            m = drift + phi * r[t - 1]
            if scenario == "nonlinear" and t >= 2:
                # Bounded interaction: same-signed recent moves increase the
                # next conditional mean; opposite-signed moves decrease it.
                interaction = np.tanh(3.0 * (r[t - 1] / sigma) * (r[t - 2] / sigma))
                m += nonlinear_strength * sigma * interaction
        mu[t] = m
        r[t] = m + eps[t]
        # Keep the simple-return price process valid in extreme custom inputs.
        r[t] = max(r[t], -0.95)

    dates = pd.bdate_range("2016-01-04", periods=n_days)
    price_values = np.empty(n_days, dtype=float)
    price_values[0] = 100.0
    price_values[1:] = 100.0 * np.cumprod(1.0 + r)
    prices = pd.Series(price_values, index=dates, name="close")
    returns = pd.Series(r, index=dates[1:], name="return")
    oracle = pd.Series(_normal_cdf(mu / sigma), index=dates[1:], name="oracle_up_probability")
    cond = pd.Series(mu, index=dates[1:], name="conditional_mean")
    return SyntheticMarket(prices, returns, oracle, cond, sigma, scenario)


def synthetic_prices(n_days: int = 2500, seed: int = 2) -> pd.Series:
    """Backward-compatible synthetic price helper using the linear scenario."""
    return synthetic_market(n_days=n_days, seed=seed, scenario="linear").prices


def _returns(prices: pd.Series) -> pd.Series:
    prices = _validate_prices(prices)
    return prices.pct_change().dropna()


def make_windows(
    prices: pd.Series,
    n_lags: int = 10,
    return_target_index: bool = False,
):
    """Build leak-free lag windows for one-step-ahead direction prediction.

    For target return ``r[t]``, features are exactly
    ``[r[t-n_lags], ..., r[t-1]]``.  V1 accidentally used ``r[t+1]`` as the
    target, leaving an unintended one-return gap between the features and the
    label.  This function makes the alignment explicit.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    rets = _returns(prices)
    r = rets.to_numpy()
    if len(r) <= n_lags:
        raise ValueError("Not enough returns for requested lag window")
    X, y, target_dates = [], [], []
    for target in range(n_lags, len(r)):
        X.append(r[target - n_lags : target])
        y.append(float(r[target] > 0))
        target_dates.append(rets.index[target])
    out = (np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    if return_target_index:
        return (*out, pd.Index(target_dates))
    return out


def make_feature_windows(
    prices: pd.Series,
    n_lags: int = 10,
    return_target_index: bool = False,
):
    """Lagged returns plus volatility/momentum known before the target return."""
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    rets = _returns(prices)
    r = rets.to_numpy()
    vol = rets.rolling(10).std().to_numpy()
    mom = rets.rolling(5).sum().to_numpy()
    start = max(n_lags, 10)
    if len(r) <= start:
        raise ValueError("Not enough returns for requested feature window")

    X, y, target_dates = [], [], []
    for target in range(start, len(r)):
        # Context features end at target-1, so target-day information cannot
        # leak into the predictor.
        row = list(r[target - n_lags : target]) + [vol[target - 1], mom[target - 1]]
        X.append(row)
        y.append(float(r[target] > 0))
        target_dates.append(rets.index[target])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Feature construction produced non-finite values")
    if return_target_index:
        return X, y, pd.Index(target_dates)
    return X, y


def walk_forward_splits(n: int, n_folds: int = 3, min_train_frac: float = 0.4):
    """Expanding-window train/test index pairs with contiguous OOS folds."""
    if n < 10:
        raise ValueError("n must be at least 10")
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must lie in (0, 1)")
    start = int(n * min_train_frac)
    remaining = n - start
    if remaining < n_folds:
        raise ValueError("Not enough observations for requested folds")
    fold = remaining // n_folds
    for i in range(n_folds):
        tr_end = start + i * fold
        te_end = tr_end + fold if i < n_folds - 1 else n
        yield np.arange(0, tr_end), np.arange(tr_end, te_end)
