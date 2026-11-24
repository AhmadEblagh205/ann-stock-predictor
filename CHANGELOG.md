# Changelog

## 0.3.0 — defensibility and known-truth audit

- Fixed the one-return feature/target alignment gap.
- Replaced the future-aware test-fold majority benchmark with fixed always-up and training-only historical-base-rate comparators.
- Added ROC-AUC, Brier score and log loss alongside accuracy.
- Added logistic regression as the linear benchmark to the MLP.
- Added controlled no-signal, linear and nonlinear synthetic worlds with an evaluation-only Bayes oracle.
- Added repeated-world robustness experiments and two generated figures.
- Added explicit data/model validation and methodology documentation.
- Expanded CI-supported tests.
- Moved TensorFlow out of the core dependency file; optional deep-learning extensions are no longer presented as audited headline results.

## 0.2.0

Added an LSTM backend and a context-feature mode in the original project iteration.  The audited 0.3.0 release does not rely on TensorFlow for its headline claims.

## 0.1.0

Initial release.
