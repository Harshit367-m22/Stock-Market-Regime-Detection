"""
Regime detection with K-Means, plus evaluation and data-driven labelling.

Leakage control
---------------
The StandardScaler and K-Means are fitted on the EARLIER part of history
(the training period) only. Later dates are then assigned to the nearest
learned centroid, exactly as a new day would be in practice. Scaling with
statistics from the whole sample would let future volatility levels shape
how past days are classified.

Labelling
---------
Cluster numbers are arbitrary. Labels are derived from each cluster's
OBSERVED training-period averages (volatility rank, sign of return and
momentum, depth of drawdown). If the numbers don't support a clean story,
the label says "mixed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_COLUMNS = ["ret_1d", "vol_20", "vol_60", "mom_20", "mdd_60"]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date (never shuffle time series)."""
    if not 0.5 <= train_frac < 1.0:
        raise ValueError("train_frac should be between 0.5 and 1.0")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def fit_model(train_X: pd.DataFrame, n_clusters: int, seed: int) -> tuple[StandardScaler, KMeans]:
    """Fit scaler + K-Means on training features only."""
    scaler = StandardScaler().fit(train_X)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    model.fit(scaler.transform(train_X))
    return scaler, model


def predict(scaler: StandardScaler, model: KMeans, X: pd.DataFrame) -> np.ndarray:
    return model.predict(scaler.transform(X))


def safe_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette needs >= 2 distinct labels and more samples than labels."""
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return None
    return float(silhouette_score(X_scaled, labels))


def k_sweep(train_X: pd.DataFrame, ks: range, seed: int) -> pd.DataFrame:
    """Diagnostic only: inertia and silhouette for several k on the training data."""
    X = StandardScaler().fit_transform(train_X)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": safe_silhouette(X, km.labels_)})
    return pd.DataFrame(rows)


def profile_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Average feature values, size and share of each cluster."""
    prof = df.groupby(cluster_col)[PROFILE_COLUMNS].mean()
    prof["days"] = df.groupby(cluster_col).size()
    prof["share"] = prof["days"] / prof["days"].sum()
    return prof


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """
    Turn observed cluster averages into descriptive labels.

    volatility level : rank of mean vol_20 (lowest = Low, highest = High,
                       anything in between = Moderate)
    direction        : 'positive' if mean daily return AND mean momentum > 0,
                       'negative' if both < 0, otherwise 'mixed / transitional'
    stressed         : highest-volatility cluster that is negative AND has the
                       deepest average rolling drawdown
    """
    k = len(profile)
    vol_rank = profile["vol_20"].rank(method="first")
    deepest_dd = profile["mdd_60"].idxmin()
    labels: dict[int, str] = {}

    for cluster, row in profile.iterrows():
        rank = vol_rank[cluster]
        level = "Low" if rank == 1 else ("High" if rank == k else "Moderate")
        positive = row["ret_1d"] > 0 and row["mom_20"] > 0
        negative = row["ret_1d"] < 0 and row["mom_20"] < 0

        if level == "High" and negative and cluster == deepest_dd:
            label = "High-volatility stressed"
        elif positive:
            label = f"{level}-volatility positive"
        elif negative:
            label = f"{level}-volatility negative"
        else:
            label = f"{level}-volatility mixed / transitional"
        labels[int(cluster)] = label

    # With k > 3 two clusters can get the same text; keep them distinguishable.
    counts = pd.Series(labels).value_counts()
    for c, lab in labels.items():
        if counts[lab] > 1:
            labels[c] = f"{lab} (cluster {c})"
    return labels


def persistence_stats(regimes: pd.Series) -> pd.DataFrame:
    """Average length (in trading days) of uninterrupted runs of each regime."""
    run_id = (regimes != regimes.shift()).cumsum().rename("run_id")
    runs = regimes.groupby(run_id).agg(["first", "size"])
    runs.columns = ["regime", "length"]
    runs = runs.reset_index(drop=True)
    stats = runs.groupby("regime")["length"].agg(["count", "mean", "max"])
    stats.columns = ["episodes", "avg_run_days", "longest_run_days"]
    return stats
