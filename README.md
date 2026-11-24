# Stock Direction Predictability Laboratory

[![CI](https://github.com/AhmadEblagh205/ann-stock-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/ann-stock-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A compact walk-forward laboratory for a deliberately difficult question: **when does a flexible neural network add information beyond market drift and a linear return model?**

The project started as a next-day ANN predictor.  The audited version is intentionally more sceptical.  It fixes a one-return target-alignment bug, removes a future-aware “majority” benchmark, evaluates probability quality as well as accuracy, and adds repeated **known-truth synthetic worlds** where the data-generating signal can be switched off, made linear, or made nonlinear.

This is an independent educational/statistical project.  It is **not** a live trading system and does not claim a real-market forecasting edge.

## Main findings

The repeated synthetic experiment uses 12 independent worlds per scenario, 2,400 trading days per world, three expanding walk-forward folds and two fixed model classes: logistic regression and a small MLP.

<!-- RESULTS_START -->
Run `python src/robustness.py` to regenerate the numerical summary and figures.  The checked-in results are generated from the audited code.
<!-- RESULTS_END -->

The interpretation matters more than a single accuracy number:

1. **No planted signal should look like no signal.**  The known-truth oracle has AUC 0.5, so any apparent neural-network “edge” is sampling noise.
2. **A neural net does not deserve a win for a mostly linear signal.**  Logistic regression is the natural benchmark.
3. **Extra flexibility becomes useful only when the DGP contains genuinely nonlinear structure and there is enough data to learn it.**
4. **Accuracy is not enough.**  Every fold is also compared with a training-only historical base-rate probability forecast using Brier score and log loss.

The synthetic oracle `Phi(mu_t/sigma)` is evaluation-only and never appears in the model features.

## Two audit corrections that materially changed the project

### 1. One-return target gap

Version 1 used the previous `L` returns but labelled each row with `r[t+1]`, unintentionally skipping `r[t]`.  V2 uses the intended one-step mapping:

```text
[r(t-L), ..., r(t-1)]  ->  sign(r(t))
```

A hand-constructed regression test now pins that alignment.

### 2. Future-aware “majority baseline”

Version 1 computed the majority class **inside each test fold** and called it an always-up/majority baseline.  That uses future test labels to decide which class the benchmark predicts.  V2 separates two honest comparators:

- **always up**, fixed before seeing the fold;
- **historical base-rate probability**, estimated from the training labels only.

## Methods

The CI-supported models are:

- standardised logistic regression;
- standardised scikit-learn MLP `(16, 8)`.

Features are the previous ten daily returns.  An optional dense-feature mode also adds trailing volatility and momentum known before the target return.  The repository reports pooled walk-forward accuracy, ROC-AUC, Brier score and log loss.

Three synthetic worlds provide known truth:

- iid/no-signal;
- linear AR-style predictability;
- nonlinear two-lag interaction.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for equations, target alignment, benchmark definitions.

## Run

```bash
pip install -r requirements.txt

# Quick leak-free demo
python src/train.py --synthetic --model mlp
python src/train.py --synthetic --model logistic

# Repeated known-truth study used in the README
python src/robustness.py

# Optional exploratory market-data mode
python src/train.py --ticker SPY --model mlp
```

TensorFlow is deliberately outside the core CI dependency set.  Install `requirements-tensorflow.txt` only if extending the project with Keras/LSTM experiments; those optional paths are not part of the audited headline results.

## Tests

`pytest` now checks:

- exact lag/target alignment;
- target-date alignment;
- context-feature leakage guards;
- contiguous expanding walk-forward splits;
- training-only base-rate construction;
- future-label independence of the baseline;
- bounded model probabilities;
- known-truth no-signal and nonlinear oracle behaviour.

## What this project does **not** establish

It does not establish profitable stock prediction, market efficiency/inefficiency, a production trading signal or a live neural-network strategy.  Real-market inference would need stronger data provenance, corporate-action handling, universe construction and multiple-testing controls.

## Licence

MIT, see [LICENSE](LICENSE).
