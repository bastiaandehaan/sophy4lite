# strategies/breakout_signals.py
from __future__ import annotations
import pandas as pd


def breakout_long(close: pd.Series, high: pd.Series, window: int = 20) -> pd.Series:
    """
    Klassiek bar-based breakout: True als close > rolling max van vorige highs.
    Voor exploratie; kan vaker dan 1x per dag vuren.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not close.index.equals(high.index):
        raise ValueError("close/high index mismatch")

    prev_high = high.shift(1)
    lvl = prev_high.rolling(window, min_periods=window).max()
    sig = (close > lvl) & lvl.notna()
    return sig.astype(bool).rename("breakout_long")


def _prev_day_high_series(high: pd.Series) -> pd.Series:
    """
    Voor elke rij: de high van de VORIGE kalenderdag.
    Eenvoudig, vectorized en tz-safe (werkt op de index zelf).
    """
    if not isinstance(high.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")

    day = high.index.normalize()              # één label per dag
    daily_highs = high.groupby(day).max()     # index: unieke dagen
    prev_daily_highs = daily_highs.shift(1)   # vorige dag

    # Map de vorige-dag-high terug naar elke rij:
    # .loc met 'day' (duplicate indexlabels) dupliceert de juiste waarde per rij.
    prev_per_row = pd.Series(prev_daily_highs.loc[day].values, index=high.index, name="prev_day_high")
    return prev_per_row


def opening_breakout_long(
    close: pd.Series,
    high: pd.Series,
    open_window_bars: int = 4,
    confirm: str = "close",  # "close" of "wick"
) -> pd.Series:
    """
    Maximaal 1 entry per dag in het openingsvenster (eerste N bars).
    Breakout-niveau = previous-day high.

    - confirm="close":  close > prev_day_high  (strict)
    - confirm="wick" :  high  > prev_day_high  (strict)

    Retourneert een boolean Series met ten hoogste 1 True per dag.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not close.index.equals(high.index):
        raise ValueError("close/high index mismatch")

    day = close.index.normalize()
    bar_idx = day.groupby(day).cumcount()   # 0,1,2,... per dag
    in_open = bar_idx < int(open_window_bars)

    prev_day_high = _prev_day_high_series(high)

    if confirm == "wick":
        raw = (high > prev_day_high)
    else:
        raw = (close > prev_day_high)

    raw = raw & prev_day_high.notna() & in_open

    # Alleen de EERSTE True per dag behouden
    first_of_day = raw & ~raw.groupby(day).cummax().shift(fill_value=False)
    return first_of_day.astype(bool).rename("opening_breakout_long")
