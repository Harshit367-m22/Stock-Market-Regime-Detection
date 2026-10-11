"""
Feature engineering for regime detection.

Every feature at date t uses ONLY prices up to and including t (trailing
windows, no centring, no negative shifts). That is what prevents future
data leaking into the features.

Features
--------
ret_1d   : simple daily return, P_t / P_{t-1} - 1           (used to describe regimes)
log_ret  : log daily return, ln(P_t / P_{t-1})              (used for volatility)
vol_20   : 20-day std of log returns, annualised by sqrt(252)
vol_60   : 60-day std of log returns, annualised by sqrt(252)
mom_20   : 20-day momentum, P_t / P_{t-20} - 1
mdd_60   : maximum drawdown inside the trailing 60-day window (<= 0)
drawdown : drawdown from the all-time high so far (<= 0; dashboard only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Inputs to K-Means. The noisy single-day return is deliberately left out:
# it would make cluster assignments flip day to day. It is still used to
# describe each regime afterwards (average daily return).
CLUSTER_FEATURES = ["vol_20", "vol_60", "mom_20", "mdd_60"]


def _window_max_drawdown(prices: np.ndarray) -> float:
    """Worst peak-to-trough fall within one window of prices (a value <= 0)."""
    running_peak = np.maximum.accumulate(prices)
    return float(np.min(prices / running_peak - 1.0))


def build_features(
    prices: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 60,
    drawdown_window: int = 60,
) -> pd.DataFrame:
    """Compute all features from a price table that has a 'Close' column."""
    close = prices["Close"].astype(float)
    out = pd.DataFrame(index=prices.index)
    out["close"] = close

    out["ret_1d"] = close.pct_change()
    out["log_ret"] = np.log(close / close.shift(1))

    annualise = np.sqrt(TRADING_DAYS)
    out[f"vol_{short_window}"] = (
        out["log_ret"].rolling(short_window, min_periods=short_window).std() * annualise
    )
    out[f"vol_{long_window}"] = (
        out["log_ret"].rolling(long_window, min_periods=long_window).std() * annualise
    )

    out[f"mom_{short_window}"] = close / close.shift(short_window) - 1.0

    out[f"mdd_{drawdown_window}"] = close.rolling(
        drawdown_window, min_periods=drawdown_window
    ).apply(_window_max_drawdown, raw=True)

    out["drawdown"] = close / close.cummax() - 1.0  # expanding peak: past data only

    # Division by zero/invalid logs would produce +/-inf; treat them as missing.
    return out.replace([np.inf, -np.inf], np.nan)


def model_ready(features: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up rows (where rolling windows are incomplete) and any NaNs."""
    needed = CLUSTER_FEATURES + ["ret_1d"]
    clean = features.dropna(subset=needed).copy()
    dropped = len(features) - len(clean)
    print(f"[features] {len(clean)} usable rows ({dropped} warm-up/invalid rows dropped).")
    return clean
