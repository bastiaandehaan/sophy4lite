from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def fetch_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Leest OHLC(V) CSV en geeft DataFrame met DatetimeIndex (UTC).
    Vereist kolommen: open, high, low, close. Volume optioneel.
    """
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
    return df
