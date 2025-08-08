import pandas as pd
import numpy as np
from typing import Tuple
from .base_strategy import BaseStrategy

class BollongStrategy(BaseStrategy):
    def __init__(self, window: int=50, std_dev: float=2.0, ma_long: int=100,
                 atr_window: int=14, max_positions: int=3,
                 use_trailing_stop: bool=False, atr_sl_mult: float=2.0, atr_tp_mult: float=3.0):
        self.window = window
        self.std_dev = std_dev
        self.ma_long = ma_long
        self.atr_window = atr_window
        self.max_positions = max_positions
        self.use_trailing_stop = use_trailing_stop
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        return tr.rolling(n).mean()

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        c = df["close"]
        ma = c.rolling(self.window).mean()
        sd = c.rolling(self.window).std()
        upper = ma + self.std_dev * sd
        lower = ma - self.std_dev * sd
        long_trend = c > c.rolling(self.ma_long).mean()
        atr = self._atr(df, self.atr_window)
        vol_ok = atr / c < 0.02  # avoid extreme vol regimes (~2% ATR)
        breakout = c > upper
        entries = (breakout & long_trend & vol_ok)
        # ATR-based SL/TP as percent of price
        sl_pct = (self.atr_sl_mult * atr / c).clip(lower=0).fillna(0)
        tp_pct = (self.atr_tp_mult * atr / c).clip(lower=0).fillna(0)
        # Only valid when entry; otherwise keep last valid percentages for portfolio engine
        sl_stop = sl_pct.where(entries, 0.0)
        tp_stop = tp_pct.where(entries, 0.0)
        entries = entries.astype(bool).fillna(False)
        return entries, sl_stop, tp_stop
