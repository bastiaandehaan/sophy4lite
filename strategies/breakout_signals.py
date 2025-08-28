from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal
import warnings
import numpy as np
import pandas as pd

Mode = Literal["close_confirm", "pending_stop"]

@dataclass(frozen=True)
class SymbolSpec:
    value_per_point: float
    min_tick: float

DEFAULT_SPECS = {    "DE40": SymbolSpec(value_per_point=1.0, min_tick=0.1),
    "GER40": SymbolSpec(value_per_point=1.0, min_tick=0.1),  # Voor FTMO
    "GER40.cash": SymbolSpec(value_per_point=1.0, min_tick=0.1),  # Voor FTMO

}

@dataclass
class BreakoutParams:
    broker_tz: str = "UTC"          # broker/server TZ; pandas regelt DST
    session_start_h: int = 7        # bv. 07:00-07:59
    session_end_h: int = 8          # exclusief
    cancel_at_h: int = 17
    min_range_pts: float = 10.0     # skip quiet days
    vol_percentile: float = 0.0     # 0 = uit; anders bv. 40
    vol_lookback: int = 50
    confirm_bars: int = 1           # #gesloten bars voorbij level (alleen close_confirm)
    mode: Mode = "close_confirm"    # of "pending_stop"
    atr_period: int = 14
    atr_sl_mult: float = 1.0
    atr_tp_mult: float = 1.5        # default RR >= 1:1.5
    offset_pts: float = 0.0         # buffer boven/onder high/low
    spread_pts: float = 0.0         # extra kostenmodel (optioneel)

    def __post_init__(self) -> None:
        if self.mode == "close_confirm" and self.confirm_bars <= 0:
            raise ValueError("confirm_bars must be > 0 in close_confirm mode")
        if self.min_range_pts < 0:
            raise ValueError("min_range_pts cannot be negative")
        if self.vol_percentile < 0 or self.vol_percentile > 100:
            raise ValueError("vol_percentile must be in [0,100]")
        if self.atr_period < 2:
            raise ValueError("atr_period must be >= 2")

def _require_tz(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("DataFrame index must be tz-aware (localize/convert it before calling).")

def _win(df: pd.DataFrame, d: pd.Timestamp, h0: int, h1: int) -> pd.DataFrame:
    """Sessievenster [h0:00, h1:00). Gebruik minuten i.p.v. -1s hacks."""
    start = pd.Timestamp(d.year, d.month, d.day, h0, 0, tz=df.index.tz)
    end   = pd.Timestamp(d.year, d.month, d.day, h1, 0, tz=df.index.tz) - pd.Timedelta(minutes=1)
    return df.loc[start:end]

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def daily_levels(df: pd.DataFrame, symbol: str, p: BreakoutParams) -> List[Dict]:
    """
    Bepaal per dag:
      - hi/lo van sessievenster
      - ATR (t/m sessie-einde) → SL/TP in punten
      - volume-percentiel (t/m sessie-einde)
    Verwacht: df is al in broker TZ en tz-aware (geen dubbele tz-conversies).
    """
    _require_tz(df)
    out: List[Dict] = []
    days = pd.DatetimeIndex(df.index.date).unique()
    has_vol = "volume" in df.columns
    vol = df["volume"] if has_vol else None

    for d in days:
        dt = pd.Timestamp(d, tz=df.index.tz)
        w = _win(df, dt, p.session_start_h, p.session_end_h)
        if w.empty:
            continue

        hi, lo = float(w["high"].max()), float(w["low"].min())
        rng = hi - lo
        if rng < p.min_range_pts:
            continue

        # ATR tot sessie-einde (geen forward bias)
        df_upto = df.loc[:w.index.max()]
        atr = _atr(df_upto, p.atr_period).dropna()
        if atr.empty:
            warnings.warn(f"[{symbol}] Skipping {d}: insufficient data for ATR(period={p.atr_period}).")
            continue
        atr_last = float(atr.iloc[-1])
        sl_pts = atr_last * p.atr_sl_mult
        tp_pts = atr_last * p.atr_tp_mult

        # volume pctl tot sessie-einde
        vol_ok = True
        if p.vol_percentile > 0 and has_vol:
            base = vol.loc[:w.index.max()].tail(p.vol_lookback)
            if base.size >= 5:
                thr = np.percentile(base.dropna().values, p.vol_percentile)
                recent = float(base.iloc[-1])
                vol_ok = bool(recent >= thr)

        out.append({
            "day": d, "symbol": symbol,
            "hi": hi, "lo": lo,
            "buy_lvl": hi + p.offset_pts,
            "sell_lvl": lo - p.offset_pts,
            "sl_pts": float(sl_pts), "tp_pts": float(tp_pts),
            "vol_ok": vol_ok,
            "cancel_at": pd.Timestamp(d.year, d.month, d.day, p.cancel_at_h, 0, tz=df.index.tz),
            "w_end": w.index.max(),
        })
    return out

def confirm_pass(df_after: pd.DataFrame, level: float, side: str, n: int) -> bool:
    """Vereist n gesloten bars ná het breekmoment volledig voorbij het level (geen look-ahead)."""
    if n <= 0:
        return True
    closes = df_after["close"].head(n)
    if closes.size < n:
        return False
    return bool((closes > level).all()) if side == "BUY" else bool((closes < level).all())