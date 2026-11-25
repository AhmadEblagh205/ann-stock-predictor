# Methodology

## Prediction target and the v1 alignment bug

Let `r_t` be the simple return from close `t-1` to close `t`.  For a lag length `L`, the intended one-step problem is

`X_t = [r_{t-L}, ..., r_{t-1}]  ->  y_t = 1(r_t > 0)`.

Version 1 instead paired `[r_{t-L}, ..., r_{t-1}]` with `1(r_{t+1}>0)`, leaving an unintended one-return gap.  Version 2 makes the target date explicit and includes regression tests with hand-constructed returns.

## Walk-forward evaluation

The first 40% of windows form the initial training set.  The remaining 60% is divided into three contiguous out-of-sample folds.  Training expands after each fold; no future observation enters a prior fit.

Two baselines are reported:

1. **Always up**: a fixed directional rule.  Its test accuracy is simply the realised fraction of positive returns.
2. **Historical base-rate probability**: every test observation receives `p = mean(y_train)` from that fold's historical training set.  This is the correct probabilistic no-feature comparator for Brier/log-loss.

Version 1 instead chose the majority class from each test fold itself.  That benchmark used future labels to decide whether “up” or “down” was the majority and therefore was not an available ex-ante rule.

## Metrics

Accuracy alone can be misleading when the unconditional up probability differs from 50%.  The audited evaluation reports:

- accuracy at probability threshold 0.5;
- ROC-AUC for ranking;
- Brier score for probability quality;
- log loss for probabilistic sharpness/calibration.

A model should not receive credit merely for tracking market drift; its probability forecasts should also beat the training-only base-rate forecast.

## Models

The core CI-supported comparison is deliberately simple:

- logistic regression after standardisation;
- a small feed-forward MLP `(16, 8)` after standardisation.

There is no hyperparameter search in the headline robustness experiment.  This isolates the question of model class rather than adding a multiple-testing layer.

## Known-truth synthetic markets

Three controlled processes are used.  They are research fixtures, not market models.

- **No signal**: iid Gaussian returns with a small constant drift.
- **Linear**: the conditional mean contains an AR(1)-style term.
- **Nonlinear**: a weak linear term plus a bounded interaction of the last two standardised returns.

Because the conditional mean `mu_t` and innovation SD `sigma` are known, the true one-step up probability is

`P(r_t > 0 | history) = Phi(mu_t / sigma)`.

That probability is an **oracle ceiling** used only for evaluation.  It never enters model features or fitting.

The key falsification logic is:

- in the no-signal world, neither learned model should reliably beat a no-feature probability forecast;
- under a mostly linear signal, a neural network should not automatically beat logistic regression;
- only when genuinely nonlinear structure exists should extra model flexibility have a reason to help.

## Real-market mode

`--ticker` and `--csv` exist so the same evaluation harness can be run on market data, but the repository does **not** publish a current empirical trading-edge claim.  Historical-equity evaluation requires careful data provenance, corporate-action handling, universe definition and multiple-testing control that are outside this compact laboratory.

## Limitations

- one asset at a time;
- fixed lag architecture and fixed model hyperparameters;
- no transaction-cost/trading-rule layer here;
- no probability calibration study;
- synthetic returns are intentionally simplified;
- optional TensorFlow sequence models are not part of CI-supported headline results.

## Notes moved from the README

Two corrections changed the project. First, version 1 labelled each row of the previous L returns with r[t+1], which skipped r[t]. Version 2 maps [r(t-L), ..., r(t-1)] to sign(r(t)), and a hand-constructed regression test pins the alignment. Second, version 1 computed the majority class inside each test fold and called it a baseline, which uses the future test labels. Version 2 uses two comparators: always up, fixed before seeing the fold, and a historical base-rate probability estimated from the training labels only.

Interpretation of the synthetic experiment: with no planted signal the oracle AUC is 0.5, so any apparent neural-network edge is sampling noise. A neural network should not be expected to beat logistic regression on a mostly linear signal. Extra flexibility helps only when the data-generating process has nonlinear structure and there is enough data to learn it. Every fold is also compared with the training-only base-rate forecast using Brier score and log loss. The oracle Phi(mu_t/sigma) is used for evaluation only and never appears in the model features.

The tests cover lag and target alignment, context-feature leakage guards, contiguous expanding walk-forward splits, training-only base-rate construction, independence of the baseline from future labels, bounded probabilities, and the no-signal and nonlinear oracle behaviour.

TensorFlow is outside the core CI dependency set. requirements-tensorflow.txt is only for optional Keras/LSTM extensions, which are not part of the headline results. Real-market inference would need better data provenance, corporate-action handling, universe construction and multiple-testing controls.
