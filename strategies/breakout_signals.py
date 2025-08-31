# strategies/breakout_signals.py
from __future__ import annotations
import pandas as pd
from typing import Optional

def breakout_long(close: pd.Series, high: pd.Series, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
    """
    Long-signaal als close > rolling max van vorige highs.
    Bar-gebaseerd, gap-safe. Geen tijd-assumpties, geen fills.
    Default: min_periods=window//2 zodat herstel na outage niet onnodig traag is.
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    prev_high = high.shift(1).rolling(window=window, min_periods=min_periods).max()
    return (close > prev_high).fillna(False)

def breakout_short(close: pd.Series, low: pd.Series, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    prev_low = low.shift(1).rolling(window=window, min_periods=min_periods).min()
    return (close < prev_low).fillna(False)
