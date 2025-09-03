# backtest/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def fetch_data(
    csv_path: Optional[str | Path] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    server_tz: str = "Europe/Athens",  # FTMO server timezone
) -> pd.DataFrame:
    """
    Load OHLC(V) data from CSV (or future MT5).
    Always return tz-aware index in `server_tz`.
    Required columns: open, high, low, close. Volume optional.
    """
    if csv_path:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # find a time column
        time_col = None
        for col in ("date", "datetime", "timestamp", "time"):
            if col in df.columns:
                time_col = col
                break
        if time_col is None:
            raise KeyError("CSV missing time column: date/datetime/timestamp/time")

        # parse to UTC then convert to server_tz
        idx = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=[time_col]).copy()
        df.index = idx
        # convert to server tz (consistent downstream)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(server_tz)
        # drop the time column if still present
        if time_col in df.columns:
            df = df.drop(columns=[time_col])
    else:
        raise NotImplementedError("MT5 fetch not implemented; use CSV.")

    # normalize columns
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]

    # ensure start/end are same tz as index
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
