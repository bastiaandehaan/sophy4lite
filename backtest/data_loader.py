from __future__ import annotations
from pathlib import Path
import pandas as pd


def fetch_data(
    *,
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from CSV and filter on date range.

    CSV schema (required):
        time, open, high, low, close, volume
    """
    if csv_path is None:
        raise ValueError("csv_path is required for data loading")

    df = pd.read_csv(csv_path, parse_dates=["time"])
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column (UTC timestamps)")

    df = df.set_index("time").sort_index()

    if start:
        df = df.loc[pd.Timestamp(start) :]
    if end:
        df = df.loc[: pd.Timestamp(end)]

    # keep only needed columns; rename/standardize if needed
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    return df[needed].copy()
