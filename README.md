# Stock Direction Predictability Laboratory

[![CI](https://github.com/AhmadEblagh205/ann-stock-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/ann-stock-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project asks when a small neural network adds information beyond a linear model for next-day stock direction. It began as a next-day ANN predictor. I corrected a target-alignment bug and a baseline that used future labels, then added simulated markets where I choose whether the signal is absent, linear or nonlinear. It is an educational project, not a trading system, and I make no claim of a real-market edge.

## What it does

- builds features from the previous ten daily returns, with optional trailing volatility and momentum;
- evaluates logistic regression and a small MLP `(16, 8)` with expanding walk-forward folds;
- reports accuracy, ROC-AUC, Brier score and log loss, against an always-up rule and a training-only base-rate forecast;
- simulates three kinds of market: no signal, linear, and a nonlinear two-lag interaction;
- optionally runs on a real ticker through `--ticker`.

## Results

The repeated experiment uses simulated data: 12 independent worlds per scenario, 2,400 trading days each, three walk-forward folds. The figures below are medians over the worlds.

| Scenario | Logistic AUC | MLP AUC | Oracle AUC |
|---|---:|---:|---:|
| No signal | 0.499 | 0.498 | 0.500 |
| Linear | 0.571 | 0.563 | 0.572 |
| Nonlinear | 0.515 | 0.649 | 0.660 |

With no signal both models sit at an AUC of 0.5, as they should. On the linear signal logistic regression does at least as well as the MLP, which beat it in 1 of 12 worlds. The MLP gains on the nonlinear signal, where it won in all 12 worlds and its median Brier score is 0.234 against 0.250 for logistic regression. None of this says anything about real markets.

Full output is in `reports/robustness.json`, `reports/robustness_worlds.csv` and `reports/results.json`.

## How to run

```bash
pip install -r requirements.txt

# Quick leak-free demo
python src/train.py --synthetic --model mlp
python src/train.py --synthetic --model logistic

# Repeated simulated study
python src/robustness.py

# Optional exploratory market-data mode
python src/train.py --ticker SPY --model mlp

pytest -q
```

## Notes

The target is `[r(t-L), ..., r(t-1)] -> sign(r(t))`, and a test pins that alignment. TensorFlow is not in the core dependencies, and `requirements-tensorflow.txt` is only for Keras or LSTM extensions that are not part of these results. Equations, benchmark definitions, the two corrections and the interpretation are in docs/METHODOLOGY.md.

## Licence

MIT, see [LICENSE](LICENSE).
