"""
End-to-end, reproducible analysis workflow.

    python run_analysis.py                 # download data, run everything
    python run_analysis.py --offline       # reuse data/sp500_raw.csv
    python run_analysis.py --k 4 --seed 7  # try other settings

Outputs (written to outputs/):
    regimes.csv          one row per day: price, features, cluster, regime, split
    regime_profile.csv   average characteristics of each regime
    k_sweep.csv          inertia / silhouette for k = 2..6 (diagnostic)
    metrics.json         settings, silhouette scores, sizes, latest regime
    figures/*.html       interactive Plotly charts (open in any browser)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DEFAULT_TICKER, load_prices
from src.features import CLUSTER_FEATURES, build_features, model_ready
from src.regimes import (
    chronological_split,
    fit_model,
    k_sweep,
    name_clusters,
    persistence_stats,
    predict,
    profile_clusters,
    safe_silhouette,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S&P 500 regime detection with K-Means")
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--k", type=int, default=3, help="number of clusters (default 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8, help="share of dates used to fit")
    p.add_argument("--offline", action="store_true", help="use cached CSV, no download")
    return p.parse_args()


def save_eda_figures(feat: pd.DataFrame) -> None:
    """Exploratory charts: index level, daily returns, rolling volatility."""
    fig = px.line(feat, y="close", title="S&P 500 index level")
    fig.update_layout(yaxis_title="Index level", xaxis_title=None)
    fig.write_html(FIG_DIR / "01_price.html", include_plotlyjs="cdn")

    fig = px.line(feat, y="ret_1d", title="Daily simple returns")
    fig.update_layout(yaxis_tickformat=".1%", yaxis_title="Return", xaxis_title=None)
    fig.write_html(FIG_DIR / "02_daily_returns.html", include_plotlyjs="cdn")

    fig = px.histogram(feat.dropna(subset=["ret_1d"]), x="ret_1d", nbins=150,
                       title="Distribution of daily returns")
    fig.update_layout(xaxis_tickformat=".1%", xaxis_title="Daily return")
    fig.write_html(FIG_DIR / "03_return_histogram.html", include_plotlyjs="cdn")

    fig = px.line(feat, y=["vol_20", "vol_60"], title="Annualised rolling volatility")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Volatility", xaxis_title=None)
    fig.write_html(FIG_DIR / "04_volatility.html", include_plotlyjs="cdn")


def save_model_figures(df: pd.DataFrame, sweep: pd.DataFrame) -> None:
    fig = px.scatter(df, x="vol_20", y="mom_20", color="regime", opacity=0.6,
                     title="Regimes in feature space (20d volatility vs 20d momentum)")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    fig.write_html(FIG_DIR / "05_regime_scatter.html", include_plotlyjs="cdn")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["silhouette"], mode="lines+markers"))
    fig.update_layout(title="Silhouette score by k (training period)", xaxis_title="k",
                      yaxis_title="Silhouette")
    fig.write_html(FIG_DIR / "06_k_sweep.html", include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Data
    prices = load_prices(DATA_DIR / "sp500_raw.csv", args.ticker, args.years, args.offline)
    print(f"[data] {len(prices)} trading days: {prices.index[0].date()} to {prices.index[-1].date()}")

    # 2) Features (trailing windows only)
    feat = build_features(prices)
    save_eda_figures(feat)
    df = model_ready(feat)

    # 3) Fit on the earlier period, assign the later period out-of-sample
    train, test = chronological_split(df, args.train_frac)
    scaler, model = fit_model(train[CLUSTER_FEATURES], args.k, args.seed)
    df["cluster"] = predict(scaler, model, df[CLUSTER_FEATURES])
    df["split"] = "train"
    df.loc[test.index, "split"] = "test"
    print(f"[model] K-Means k={args.k}, seed={args.seed}; fitted on {len(train)} days "
          f"({train.index[0].date()} to {train.index[-1].date()}), "
          f"assigned {len(test)} later days out-of-sample.")

    # 4) Interpret clusters from TRAINING-period averages, then name them
    train_profile = profile_clusters(df[df["split"] == "train"])
    names = name_clusters(train_profile)
    df["regime"] = df["cluster"].map(names)

    profile = profile_clusters(df)  # full-period profile for reporting
    profile.insert(0, "regime", profile.index.map(names))
    persistence = persistence_stats(df["regime"])
    profile = profile.join(persistence, on="regime")

    # 5) Evaluation
    is_train = df["split"] == "train"
    X_train = scaler.transform(df.loc[is_train, CLUSTER_FEATURES])
    X_test = scaler.transform(df.loc[~is_train, CLUSTER_FEATURES])
    sil_train = safe_silhouette(X_train, df.loc[is_train, "cluster"].to_numpy())
    sil_test = safe_silhouette(X_test, df.loc[~is_train, "cluster"].to_numpy())
    sweep = k_sweep(train[CLUSTER_FEATURES], range(2, 7), args.seed)
    save_model_figures(df, sweep)

    latest = df.iloc[-1]
    metrics = {
        "ticker": args.ticker,
        "data_start": str(prices.index[0].date()),
        "data_end": str(prices.index[-1].date()),
        "k": args.k,
        "seed": args.seed,
        "features": CLUSTER_FEATURES,
        "train_period": [str(train.index[0].date()), str(train.index[-1].date())],
        "test_period": [str(test.index[0].date()), str(test.index[-1].date())] if len(test) else None,
        "silhouette_train": sil_train,
        "silhouette_test": sil_test,
        "cluster_sizes": {names[int(c)]: int(n) for c, n in df["cluster"].value_counts().sort_index().items()},
        "regime_switches": int((df["regime"] != df["regime"].shift()).sum() - 1),
        "latest": {
            "date": str(df.index[-1].date()),
            "regime": latest["regime"],
            "close": float(latest["close"]),
            "vol_20": float(latest["vol_20"]),
            "mom_20": float(latest["mom_20"]),
            "mdd_60": float(latest["mdd_60"]),
        },
    }

    # 6) Save
    out_cols = ["close", "ret_1d", "log_ret"] + CLUSTER_FEATURES + ["drawdown", "cluster", "regime", "split"]
    df[out_cols].to_csv(OUT_DIR / "regimes.csv", index_label="Date")
    profile.to_csv(OUT_DIR / "regime_profile.csv", index_label="cluster")
    sweep.to_csv(OUT_DIR / "k_sweep.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 7) Console report (ASCII only, so it prints in any Windows console)
    pd.set_option("display.width", 160)
    show = profile.copy()
    show["ret_1d"] = (show["ret_1d"] * 100).round(3)  # daily returns are small; keep 3 dp
    for c in ["vol_20", "vol_60", "mom_20", "mdd_60", "share"]:
        show[c] = (show[c] * 100).round(1)
    show["avg_run_days"] = show["avg_run_days"].round(1)
    print("\n=== Regime profile (full period; returns, vol, momentum, drawdown in %) ===")
    print(show.to_string())
    print("\n=== k sweep on training data (diagnostic) ===")
    print(sweep.round(3).to_string(index=False))
    fmt = lambda s: "n/a" if s is None else f"{s:.3f}"
    print(f"\nSilhouette (train): {fmt(sil_train)}   Silhouette (test, out-of-sample): {fmt(sil_test)}")
    print(f"Regime switches over the period: {metrics['regime_switches']}")
    print(f"Latest regime ({metrics['latest']['date']}): {metrics['latest']['regime']}")
    print(f"\nSaved outputs to {OUT_DIR}")
    print("Labels are derived from the table above - check they make sense before quoting them.")


if __name__ == "__main__":
    main()
