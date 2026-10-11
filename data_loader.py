"""
Data collection: download daily S&P 500 (^GSPC) prices from Yahoo Finance.

Design choices
--------------
* yfinance is an unofficial, free wrapper around Yahoo Finance. It can fail
  (network issues, rate limits, empty responses), so we retry with a growing
  pause and fall back to a locally cached CSV if one exists.
* Recent yfinance versions return MultiIndex columns such as
  ('Close', '^GSPC') even for a single ticker, and use auto_adjust=True by
  default (so there is no 'Adj Close' column). `clean_prices` handles both
  the old and the new layouts.
* We do NOT reindex to a full business-day calendar. Missing dates are
  exchange holidays; inserting them would create fake zero-return days.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKER = "^GSPC"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_prices(
    ticker: str = DEFAULT_TICKER,
    years: int = 10,
    retries: int = 3,
    pause_seconds: float = 5.0,
) -> pd.DataFrame:
    """Download raw daily OHLCV data. Raises RuntimeError if all attempts fail."""
    import yfinance as yf  # imported here so offline runs don't need network setup

    # yfinance treats `end` as exclusive, so add one day to include today.
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[data] Downloading {ticker} (attempt {attempt}/{retries}) ...")
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,  # explicit, so behaviour doesn't depend on the version default
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
            last_error = RuntimeError("Yahoo Finance returned an empty table.")
        except Exception as exc:  # network errors, rate limits, parsing errors
            last_error = exc
        if attempt < retries:
            wait = pause_seconds * attempt
            print(f"[data] Download failed ({last_error}). Retrying in {wait:.0f}s ...")
            time.sleep(wait)

    raise RuntimeError(f"Could not download {ticker} after {retries} attempts: {last_error}")


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns, dates and values into a tidy daily price table."""
    df = raw.copy()

    # 1) Flatten MultiIndex columns: keep whichever level holds 'Close'.
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = [str(v) for v in df.columns.get_level_values(level)]
            if "Close" in values or "Adj Close" in values:
                df.columns = values
                break
    df.columns = [str(c).strip() for c in df.columns]

    # Older/other settings may give 'Adj Close' instead of an adjusted 'Close'.
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' column found. Columns were: {list(df.columns)}")

    # 2) Dates: datetime, timezone-naive, midnight, sorted, unique.
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 3) Values: numeric, finite, strictly positive close.
    keep = [c for c in PRICE_COLUMNS if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df["Close"] <= 0, "Close"] = np.nan
    n_missing = int(df["Close"].isna().sum())
    if n_missing:
        print(f"[data] Dropping {n_missing} rows with missing/invalid Close.")
    df = df.dropna(subset=["Close"])

    df.index.name = "Date"
    if df.empty:
        raise ValueError("No valid price rows remain after cleaning.")
    return df


def load_prices(
    cache_path: str | Path,
    ticker: str = DEFAULT_TICKER,
    years: int = 10,
    offline: bool = False,
) -> pd.DataFrame:
    """
    Return clean daily prices.

    * offline=True  -> read the cached CSV only (no network).
    * offline=False -> download; save to cache; if the download fails and a
                       cache exists, fall back to the cache with a warning.
    """
    cache_path = Path(cache_path)

    def _read_cache() -> pd.DataFrame:
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
        return clean_prices(cached)

    if offline:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Offline mode requested but no cache at {cache_path}. Run once without --offline."
            )
        print(f"[data] Offline mode: reading {cache_path}")
        return _read_cache()

    try:
        prices = clean_prices(download_prices(ticker, years))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        print(f"[data] Saved {len(prices)} rows to {cache_path}")
        return prices
    except Exception as exc:
        if cache_path.exists():
            print(f"[data] WARNING: download failed ({exc}). Using cached file {cache_path}.")
            return _read_cache()
        raise
