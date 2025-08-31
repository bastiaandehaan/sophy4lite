# utils/data_health.py
from __future__ import annotations
import pandas as pd

def health_stats(df: pd.DataFrame, expected_freq: str = "15T") -> dict:
    """
    Kale feiten over index-regulariteit (geen thresholds/policy):
      - gaps: aantal index-deltas ≠ expected
      - top: top-3 afwijkende deltas met counts
      - dups: aantal duplicate timestamps
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"error": "index!=DatetimeIndex"}

    exp = pd.to_timedelta(expected_freq)
    diffs = df.index.to_series().diff().dropna()
    off = diffs[diffs != exp]
    top = off.value_counts().head(3)
    return {
        "gaps": int(len(off)),
        "dups": int(df.index.duplicated().sum()),
        "top": {str(delta): int(cnt) for delta, cnt in top.items()},
    }

def health_line(df: pd.DataFrame, expected_freq: str = "15T") -> str:
    """Compacte 1-regel logstring op basis van health_stats."""
    st = health_stats(df, expected_freq=expected_freq)
    if "error" in st:
        return f"DATA {{error:{st['error']}}}"
    top = " ".join([f"{k}x{v}" for k, v in st["top"].items()])
    return f"DATA {{gaps:{st['gaps']} dups:{st['dups']} top:[{top}]}}"
