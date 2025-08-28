from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def fetch_data(
    csv_path: Optional[str | Path] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Laadt OHLC(V) data van CSV of (placeholder) MT5.
    Vereist kolommen: open, high, low, close. Volume optioneel.
    """
    if csv_path:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # tijdkolom detecteren
        for col in ("date", "datetime", "timestamp", "time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                df = df.dropna(subset=[col]).set_index(col)
                break
        else:
            raise KeyError("CSV mist tijdkolom: date/datetime/timestamp/time")
    else:
        # Placeholder voor MT5 fetch (implementeer als nodig)
        raise NotImplementedError("MT5 data fetch nog niet geïmplementeerd; gebruik CSV.")

    # kolommen normaliseren
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Vereiste kolommen ontbreken: {missing}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]

    # Zorg dat start/end dezelfde tijdzone krijgen als de index
    tz = df.index.tz

    if start:
        ts = pd.Timestamp(start)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)  # align naar index-tz
        df = df.loc[ts:]

    if end:
        ts = pd.Timestamp(end)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        df = df.loc[:ts]

    return df
