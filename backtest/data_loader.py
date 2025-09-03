# backtest/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def fetch_data(csv_path: Optional[str | Path] = None, symbol: Optional[str] = None,
        # reserved for MT5/other loaders
        timeframe: Optional[str] = None,  # reserved for MT5/other loaders
        start: Optional[str] = None, end: Optional[str] = None,
        server_tz: str = "Europe/Athens",  # FTMO server timezone
) -> pd.DataFrame:
    """
    Load OHLC(V) data from CSV.
    Returns tz-aware index in `server_tz`.
    Required columns: open, high, low, close. Volume optional.
    """
    if not csv_path:
        raise NotImplementedError(
            "MT5/other sources not implemented; use CSV via csv_path.")
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Detect time column
    time_col = next(
        (c for c in ("date", "datetime", "timestamp", "time") if c in df.columns), None)
    if time_col is None:
        raise KeyError("CSV missing time column: one of date/datetime/timestamp/time")

    # Parse to UTC, drop NaT, set as index, convert to server_tz
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(server_tz)

    # Normalize columns
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    # Coerce numeric + add volume if absent
    for c in required + (["volume"] if "volume" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # Clean
    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")]

    # Slice in the same tz as index
    tz = df.index.tz
    if start:
        ts = pd.Timestamp(start)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        df = df.loc[ts:]
    if end:
        ts = pd.Timestamp(end)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        df = df.loc[:ts]

    return df
