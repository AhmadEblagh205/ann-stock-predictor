"""One-step stock-direction evaluation with leak-free walk-forward splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import load_prices, make_feature_windows, make_windows
from modeling import evaluate_walk_forward

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", choices=["logistic", "mlp"], default="mlp")
    ap.add_argument("--features", action="store_true", help="add rolling volatility and 5-day momentum")
    ap.add_argument("--lags", type=int, default=10)
    args = ap.parse_args()

    prices = load_prices(args.ticker, csv=args.csv, synthetic=args.synthetic)
    maker = make_feature_windows if args.features else make_windows
    X, y = maker(prices, n_lags=args.lags)
    result = evaluate_walk_forward(X, y, args.model)

    for fold in result.folds:
        fm = fold["model"]
        print(
            f"fold {fold['fold']}: acc {fm['accuracy']:.3f}, AUC {fm['roc_auc']:.3f}, "
            f"Brier {fm['brier']:.3f}; always-up acc {fold['always_up_accuracy']:.3f}"
        )

    summary = {
        "model": args.model,
        "lags": args.lags,
        "context_features": bool(args.features),
        "folds": result.folds,
        "pooled_oos": result.metrics,
        "note": "Historical base-rate probabilities use training labels only; always-up is a fixed directional benchmark.",
    }
    out = ROOT / "reports" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    m = result.metrics["model"]
    b = result.metrics["historical_base_rate"]
    print(
        f"\npooled OOS: acc {m['accuracy']:.3f}, AUC {m['roc_auc']:.3f}, Brier {m['brier']:.3f}; "
        f"base-rate Brier {b['brier']:.3f}; always-up acc {result.metrics['always_up_accuracy']:.3f}"
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
