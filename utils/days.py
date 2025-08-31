# utils/days.py
from __future__ import annotations
import pandas as pd
from typing import Tuple

def day_counts(df: pd.DataFrame) -> pd.Series:
    """Aantal bars per kalenderdag die in de data voorkomen."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    return df.index.normalize().value_counts().sort_index()

def trading_days(df: pd.DataFrame) -> pd.Index:
    """Unieke dagen die daadwerkelijk bars hebben (geen lege/weekenddagen)."""
    return df.index.normalize().unique()

def summarize_day_health(df: pd.DataFrame) -> Tuple[int, int, float]:
    """
    Geeft (days, min_bars, mean_bars) terug.
    Alleen feiten loggen; geen auto-skip hier.
    """
    cnts = day_counts(df)
    if cnts.empty:
        return (0, 0, 0.0)
    return (int(len(cnts)), int(cnts.min()), float(cnts.mean()))
