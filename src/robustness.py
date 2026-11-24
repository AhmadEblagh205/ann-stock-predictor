"""Known-truth robustness experiments for stock-direction predictability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import make_windows, synthetic_market
from modeling import evaluate_walk_forward

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _oracle_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    p = np.clip(np.asarray(p), 1e-8, 1 - 1e-8)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def evaluate_world(seed: int, scenario: str, n_days: int = 2400, nonlinear_strength: float = 0.45) -> list[dict]:
    kwargs = {}
    if scenario == "linear":
        kwargs.update(phi=0.18, nonlinear_strength=0.0)
    elif scenario == "nonlinear":
        kwargs.update(phi=0.06, nonlinear_strength=nonlinear_strength)
    elif scenario == "no_signal":
        kwargs.update(phi=0.0, nonlinear_strength=0.0)
    market = synthetic_market(n_days=n_days, seed=seed, scenario=scenario, **kwargs)
    X, y, target_dates = make_windows(market.prices, n_lags=2, return_target_index=True)

    rows = []
    # pooled OOS indices are the final 60% under the default split; align oracle
    # probabilities by date to the exact evaluated targets.
    first_oos = int(len(X) * 0.4)
    oracle_p = market.oracle_up_probability.reindex(target_dates[first_oos:]).to_numpy()
    oracle_y = y[first_oos:].astype(int)
    om = _oracle_metrics(oracle_y, oracle_p)

    for model in ("logistic", "mlp"):
        res = evaluate_walk_forward(X, y, model, seed=seed * 11)
        mm = res.metrics["model"]
        bm = res.metrics["historical_base_rate"]
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "model": model,
                "accuracy": mm["accuracy"],
                "roc_auc": mm["roc_auc"],
                "brier": mm["brier"],
                "log_loss": mm["log_loss"],
                "base_brier": bm["brier"],
                "always_up_accuracy": res.metrics["always_up_accuracy"],
                "oracle_auc": om["roc_auc"],
                "oracle_brier": om["brier"],
            }
        )
    return rows


def run(n_worlds: int = 12, n_days: int = 2400) -> tuple[pd.DataFrame, dict]:
    rows = []
    for scenario in ("no_signal", "linear", "nonlinear"):
        for seed in range(n_worlds):
            rows.extend(evaluate_world(seed, scenario, n_days=n_days))
    df = pd.DataFrame(rows)

    summary = {}
    for scenario, sdf in df.groupby("scenario"):
        summary[scenario] = {}
        for model, mdf in sdf.groupby("model"):
            summary[scenario][model] = {
                "median_auc": float(mdf["roc_auc"].median()),
                "median_brier": float(mdf["brier"].median()),
                "median_brier_improvement_vs_base": float((mdf["base_brier"] - mdf["brier"]).median()),
                "median_accuracy": float(mdf["accuracy"].median()),
                "median_always_up_accuracy": float(mdf["always_up_accuracy"].median()),
            }
        piv = sdf.pivot(index="seed", columns="model", values="roc_auc")
        summary[scenario]["mlp_minus_logistic_auc"] = {
            "median": float((piv["mlp"] - piv["logistic"]).median()),
            "mlp_wins_fraction": float((piv["mlp"] > piv["logistic"]).mean()),
        }
        first = sdf.groupby("seed").first()
        summary[scenario]["oracle"] = {
            "median_auc": float(first["oracle_auc"].median()),
            "median_brier": float(first["oracle_brier"].median()),
        }

    return df, summary


def plot(df: pd.DataFrame):
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = ["no_signal", "linear", "nonlinear"]
    labels = ["No signal", "Linear AR", "Nonlinear interaction"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(order))
    width = 0.32
    for j, model in enumerate(["logistic", "mlp"]):
        med = [df[(df.scenario == s) & (df.model == model)].roc_auc.median() for s in order]
        lo = [df[(df.scenario == s) & (df.model == model)].roc_auc.quantile(0.25) for s in order]
        hi = [df[(df.scenario == s) & (df.model == model)].roc_auc.quantile(0.75) for s in order]
        pos = x + (j - 0.5) * width
        ax.bar(pos, med, width=width, label=model.title())
        ax.errorbar(pos, med, yerr=[np.array(med)-np.array(lo), np.array(hi)-np.array(med)], fmt="none", capsize=4)
    oracle = [df[df.scenario == s].drop_duplicates("seed").oracle_auc.median() for s in order]
    ax.plot(x, oracle, marker="o", linestyle="--", label="Known-truth oracle")
    ax.axhline(0.5, linestyle=":", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Pooled walk-forward ROC-AUC")
    ax.set_title("Model complexity helps only when the data-generating signal is nonlinear")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "signal_structure_robustness.png", dpi=160)
    plt.close(fig)

    # Show probability quality relative to the training-only base-rate forecast.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = []
    tick = []
    for s, label in zip(order, labels):
        for model in ("logistic", "mlp"):
            sub = df[(df.scenario == s) & (df.model == model)]
            vals.append(sub.base_brier - sub.brier)
            tick.append(f"{label}\n{model}")
    ax.boxplot(vals, tick_labels=tick, showfliers=False)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_ylabel("Brier improvement vs historical base-rate forecast")
    ax.set_title("Probability forecasts should beat a training-only base-rate benchmark")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "brier_improvement.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--days", type=int, default=2400)
    args = ap.parse_args()
    df, summary = run(args.worlds, args.days)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "robustness_worlds.csv", index=False)
    (REPORTS / "robustness.json").write_text(json.dumps({"n_worlds": args.worlds, "n_days": args.days, "summary": summary}, indent=2))
    plot(df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
