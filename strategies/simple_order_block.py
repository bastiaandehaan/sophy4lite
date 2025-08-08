import pandas as pd
import numpy as np
from typing import Tuple
from .base_strategy import BaseStrategy

class SimpleOrderBlockStrategy(BaseStrategy):
    def __init__(self, ob_lookback: int=5, body_mult: float=1.5,
                 rsi_min: int=25, rsi_max: int=75,
                 use_htf: bool=True, htf_ma:int=20,
                 sl_fixed_percent: float=0.01, tp_fixed_percent: float=0.03):
        self.ob_lookback = ob_lookback
        self.body_mult = body_mult
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.use_htf = use_htf
        self.htf_ma = htf_ma
        self.sl_fixed_percent = sl_fixed_percent
        self.tp_fixed_percent = tp_fixed_percent

    def _rsi(self, s: pd.Series, n: int=14) -> pd.Series:
        delta = s.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / (down.replace(0, np.nan))
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        avg_body = body.rolling(self.ob_lookback).mean()
        strong_bull = (c > o) & (body > self.body_mult * avg_body)
        rsi = self._rsi(c).fillna(50)
        rsi_ok = (rsi.between(self.rsi_min, self.rsi_max))
        # Simple trend proxy: price above 20MA
        trend_ok = c > c.rolling(self.htf_ma).mean() if self.use_htf else pd.Series(True, index=c.index)
        entries = (strong_bull & rsi_ok & trend_ok)
        sl_stop = pd.Series(self.sl_fixed_percent, index=c.index).where(entries, 0.0)
        tp_stop = pd.Series(self.tp_fixed_percent, index=c.index).where(entries, 0.0)
        return entries.astype(bool).fillna(False), sl_stop, tp_stop
