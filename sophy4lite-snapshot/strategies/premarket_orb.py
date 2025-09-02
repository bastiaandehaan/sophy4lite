# strategies/premarket_orb.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
from zoneinfo import ZoneInfo

@dataclass(frozen=True)
class ORBParams:
    # Lokale cash‑open (HH:MM) en bijbehorende tijdzone
    session_open_local: str = "09:00"       # DAX: "09:00"; Dow: "09:30"
    session_tz: str = "Europe/Berlin"       # DAX; voor Dow "America/New_York"
    premarket_minutes: int = 60             # Trader Tom range = 60 minuten
    confirm: str = "close"                  # "close" of "wick"
    one_trade_per_day: bool = True          # max 1 trade per dag
    allow_both_sides: bool = True           # zowel long als short

def _first_idx(s: pd.Series) -> Optional[pd.Timestamp]:
    """Returneer de eerste True‑index (of None)."""
    if not s.any():
        return None
    return s[s].index[0]

def premarket_orb_entries(
    df: pd.DataFrame,
    p: ORBParams,
) -> Tuple[pd.Series, pd.Series]:
    """
    Bepaal pre‑market high/low en geef twee boolean Series terug:
    entries_long en entries_short.
    """
    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not req.issubset(df.columns):
        miss = req.difference(df.columns)
        raise KeyError(f"df mist kolommen: {sorted(miss)}")

    df = df.sort_index()
    loc_idx = df.index.tz_convert(ZoneInfo(p.session_tz))
    loc_day = loc_idx.normalize()

    el = pd.Series(False, index=df.index, name="entries_long")
    es = pd.Series(False, index=df.index, name="entries_short")

    hh, mm = map(int, p.session_open_local.split(":"))
    for d in pd.unique(loc_day):
        open_loc = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=ZoneInfo(p.session_tz))
        pre_start = open_loc - pd.Timedelta(minutes=p.premarket_minutes)

        pre_mask = (loc_idx >= pre_start) & (loc_idx < open_loc)
        ses_mask = (loc_idx >= open_loc) & (loc_day == d)

        pre = df.loc[pre_mask]
        ses = df.loc[ses_mask]
        if pre.empty or ses.empty:
            continue

        hi = float(pre["high"].max())
        lo = float(pre["low"].min())

        if p.confirm == "wick":
            cond_long = ses["high"] > hi
            cond_short = ses["low"] < lo
        else:
            cond_long = ses["close"] > hi
            cond_short = ses["close"] < lo

        t_long = _first_idx(cond_long)
        t_short = _first_idx(cond_short)

        if p.one_trade_per_day:
            if t_long is not None and (t_short is None or t_long <= t_short):
                el.loc[t_long] = True
            elif t_short is not None:
                es.loc[t_short] = True
        else:
            if t_long is not None:
                el.loc[t_long] = True
            if t_short is not None:
                es.loc[t_short] = True

    return el.astype(bool), es.astype(bool)
