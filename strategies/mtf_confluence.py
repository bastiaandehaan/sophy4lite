# strategies/mtf_confluence.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
import pandas as pd


class Bias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass(frozen=True)
class MTFParams:
    """Multi-timeframe strategy parameters"""
    # Trend detection
    ema_period: int = 20
    atr_period: int = 14
    # Structure levels
    structure_lookback: int = 20
    zone_buffer: float = 0.2  # ATR multiplier for zones
    # Entry timing
    momentum_threshold: float = 0.6
    # Risk management
    atr_mult_sl: float = 1.5
    atr_mult_tp: float = 2.5
    # Signal quality
    min_confluence_score: float = 0.65
    # Optional EV filter (off by default to avoid bias)
    use_ev_in_filter: bool = False
    min_ev_R: float = 0.0


class MTFSignals:
    """Multi-timeframe signal generator (look-ahead safe)."""

    def __init__(self, params: MTFParams = MTFParams()):
        self.params = params
        self.expectancy_history: List[float] = []

    # ---------- Timeframe construction ----------
    def create_timeframes(self, df_m1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create synthetic timeframes from M1 data.
        Returns dict with 'M1','M5','M15','H1'.
        Resampling is right-labeled/closed to prevent look-ahead.
        """
        if not isinstance(df_m1.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        req = {"open", "high", "low", "close"}
        miss = req.difference(df_m1.columns)
        if miss:
            raise KeyError(f"DataFrame missing columns: {sorted(miss)}")

        tfs: Dict[str, pd.DataFrame] = {"M1": df_m1.copy()}
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_m1.columns else "last",
        }
        tfs["M5"] = df_m1.resample("5min", label="right", closed="right").agg(agg).dropna()
        tfs["M15"] = df_m1.resample("15min", label="right", closed="right").agg(agg).dropna()
        tfs["H1"] = df_m1.resample("60min", label="right", closed="right").agg(agg).dropna()
        return tfs

    # ---------- H1 bias ----------
    def calculate_bias(self, df_h1: pd.DataFrame) -> pd.Series:
        ema = df_h1["close"].ewm(span=self.params.ema_period, adjust=False).mean()
        h, l, c = df_h1["high"], df_h1["low"], df_h1["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.params.atr_period, adjust=False).mean()

        bias = pd.Series(index=df_h1.index, dtype=object)
        bias[c > ema + atr * 0.1] = Bias.BULLISH
        bias[c < ema - atr * 0.1] = Bias.BEARISH
        bias[bias.isna()] = Bias.NEUTRAL
        return bias

    # ---------- M15 structure ----------
    def find_structure_levels(self, df_m15: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.structure_lookback
        roll_high = df_m15["high"].rolling(lookback).max()
        roll_low = df_m15["low"].rolling(lookback).min()

        h, l, c = df_m15["high"], df_m15["low"], df_m15["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.params.atr_period, adjust=False).mean()

        buffer = atr * self.params.zone_buffer
        structure = pd.DataFrame(index=df_m15.index)
        structure["resistance_high"] = roll_high + buffer
        structure["resistance_low"] = roll_high - buffer
        structure["support_high"] = roll_low + buffer
        structure["support_low"] = roll_low - buffer
        structure["atr"] = atr
        return structure

    # ---------- M5 patterns ----------
    def detect_setup_patterns(self, df_m5: pd.DataFrame) -> pd.DataFrame:
        patterns = pd.DataFrame(index=df_m5.index)

        # Bullish
        patterns["hammer"] = (
            (df_m5["close"] > df_m5["open"])
            & ((df_m5["high"] - df_m5["close"]) < (df_m5["close"] - df_m5["open"]) * 0.3)
            & ((df_m5["open"] - df_m5["low"]) > (df_m5["close"] - df_m5["open"]) * 2)
        )
        patterns["bullish_engulfing"] = (
            (df_m5["close"] > df_m5["open"])
            & (df_m5["open"].shift() > df_m5["close"].shift())
            & (df_m5["close"] > df_m5["open"].shift())
            & (df_m5["open"] < df_m5["close"].shift())
        )

        # Bearish
        patterns["shooting_star"] = (
            (df_m5["open"] > df_m5["close"])
            & ((df_m5["low"] - df_m5["close"]) < (df_m5["open"] - df_m5["close"]) * 0.3)
            & ((df_m5["high"] - df_m5["open"]) > (df_m5["open"] - df_m5["close"]) * 2)
        )
        patterns["bearish_engulfing"] = (
            (df_m5["open"] > df_m5["close"])
            & (df_m5["close"].shift() > df_m5["open"].shift())
            & (df_m5["open"] > df_m5["close"].shift())
            & (df_m5["close"] < df_m5["open"].shift())
        )

        # Strengths
        patterns["bullish_strength"] = (
            patterns["hammer"].astype(int) + patterns["bullish_engulfing"].astype(int)
        ) / 2.0
        patterns["bearish_strength"] = (
            patterns["shooting_star"].astype(int) + patterns["bearish_engulfing"].astype(int)
        ) / 2.0
        return patterns

    # ---------- M1 micro-momentum ----------
    def calculate_momentum(self, df_m1: pd.DataFrame, window: int = 10) -> pd.Series:
        roc = df_m1["close"].pct_change(window)
        momentum = roc / roc.rolling(50).std()
        momentum = momentum.clip(-2, 2) / 2  # scale to [-1,1]
        return momentum.fillna(0)

    # ---------- Confluence score ----------
    def calculate_confluence_score(
        self,
        bias: Bias,
        at_support: bool,
        at_resistance: bool,
        pattern_strength: float,
        momentum: float,
    ) -> float:
        score = 0.0
        # Bias alignment (40%)
        if bias == Bias.BULLISH and at_support:
            score += 0.4
        elif bias == Bias.BEARISH and at_resistance:
            score += 0.4
        elif bias == Bias.NEUTRAL:
            score += 0.1
        # Pattern strength (30%)
        score += float(pattern_strength) * 0.3
        # Momentum (30%)
        if bias == Bias.BULLISH and momentum > self.params.momentum_threshold:
            score += 0.3
        elif bias == Bias.BEARISH and momentum < -self.params.momentum_threshold:
            score += 0.3
        elif abs(momentum) > self.params.momentum_threshold:
            score += 0.15
        return float(min(score, 1.0))

    # ---------- EV (calc only; optional as filter) ----------
    def calculate_expected_value(
        self,
        entry: float,
        sl: float,
        tp: float,
        win_rate: float = 0.58,
    ) -> float:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk == 0:
            return 0.0
        ev = (win_rate * reward) - ((1 - win_rate) * risk)
        return ev / risk  # in R

    # ---------- Signal generation ----------
    def generate_signals(
        self,
        df_m1: pd.DataFrame,
        session_start: str = "09:00",
        session_end: str = "17:00",
    ) -> pd.DataFrame:
        """
        Look-ahead safe mapping:
        - any higher-TF feature must be shifted by 1 bar **before** reindex to M1.
        """
        tfs = self.create_timeframes(df_m1)
        bias_h1 = self.calculate_bias(tfs["H1"])
        structure_m15 = self.find_structure_levels(tfs["M15"])
        patterns_m5 = self.detect_setup_patterns(tfs["M5"])
        momentum_m1 = self.calculate_momentum(tfs["M1"])

        signals = pd.DataFrame(index=df_m1.index)

        # CRITICAL: lag higher-TF by one closed bar before mapping to M1
        signals["bias"] = bias_h1.shift(1).reindex(df_m1.index, method="ffill")
        signals["resistance"] = structure_m15["resistance_low"].shift(1).reindex(df_m1.index, method="ffill")
        signals["support"] = structure_m15["support_high"].shift(1).reindex(df_m1.index, method="ffill")
        signals["atr_m15"] = structure_m15["atr"].shift(1).reindex(df_m1.index, method="ffill")
        signals["bullish_pattern"] = patterns_m5["bullish_strength"].shift(1).reindex(df_m1.index, method="ffill")
        signals["bearish_pattern"] = patterns_m5["bearish_strength"].shift(1).reindex(df_m1.index, method="ffill")
        signals["momentum"] = momentum_m1  # M1 = no shift

        # Location vs zones
        signals["at_support"] = (
            (df_m1["low"] <= signals["support"])
            & (df_m1["close"] > signals["support"] - signals["atr_m15"] * 0.5)
        )
        signals["at_resistance"] = (
            (df_m1["high"] >= signals["resistance"])
            & (df_m1["close"] < signals["resistance"] + signals["atr_m15"] * 0.5)
        )

        # Scores
        signals["long_score"] = signals.apply(
            lambda r: self.calculate_confluence_score(
                r["bias"], r["at_support"], False, r["bullish_pattern"], r["momentum"]
            ) if pd.notna(r["bias"]) else 0.0,
            axis=1,
        )
        signals["short_score"] = signals.apply(
            lambda r: self.calculate_confluence_score(
                r["bias"], False, r["at_resistance"], r["bearish_pattern"], -r["momentum"]
            ) if pd.notna(r["bias"]) else 0.0,
            axis=1,
        )

        # Session filter (DST-safe)
        in_session_index = df_m1.between_time(session_start, session_end, include_end=False).index
        in_session_mask = df_m1.index.isin(in_session_index)

        # SL/TP levels (using lagged M15 ATR)
        signals["long_sl"] = df_m1["close"] - signals["atr_m15"] * self.params.atr_mult_sl
        signals["long_tp"] = df_m1["close"] + signals["atr_m15"] * self.params.atr_mult_tp
        signals["short_sl"] = df_m1["close"] + signals["atr_m15"] * self.params.atr_mult_sl
        signals["short_tp"] = df_m1["close"] - signals["atr_m15"] * self.params.atr_mult_tp

        # EV (calculated; optionally used as filter)
        signals["long_ev"] = signals.apply(
            lambda r: self.calculate_expected_value(df_m1.loc[r.name, "close"], r["long_sl"], r["long_tp"])
            if r.name in df_m1.index else 0.0,
            axis=1,
        )
        signals["short_ev"] = signals.apply(
            lambda r: self.calculate_expected_value(df_m1.loc[r.name, "close"], r["short_sl"], r["short_tp"])
            if r.name in df_m1.index else 0.0,
            axis=1,
        )

        # Entry filter(s)
        base_long = in_session_mask & (signals["long_score"] >= self.params.min_confluence_score)
        base_short = in_session_mask & (signals["short_score"] >= self.params.min_confluence_score)

        if self.params.use_ev_in_filter:
            signals["long_entry"] = base_long & (signals["long_ev"] >= self.params.min_ev_R)
            signals["short_entry"] = base_short & (signals["short_ev"] >= self.params.min_ev_R)
        else:
            signals["long_entry"] = base_long
            signals["short_entry"] = base_short

        return signals

    # ---------- Daily best (optional) ----------
    def filter_best_daily_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
        filtered = signals.copy()
        filtered["best_score"] = filtered[["long_score", "short_score"]].max(axis=1)
        filtered["best_side"] = filtered.apply(
            lambda r: "long" if r["long_score"] > r["short_score"] else "short",
            axis=1,
        )
        daily_best = (
            filtered.groupby(filtered.index.date)
            .apply(lambda g: g.nlargest(1, "best_score").index[0] if len(g) > 0 else None)
            .dropna()
        )
        filtered["long_entry"] = False
        filtered["short_entry"] = False
        for ts in daily_best.to_list():
            if filtered.loc[ts, "best_side"] == "long":
                filtered.loc[ts, "long_entry"] = True
            else:
                filtered.loc[ts, "short_entry"] = True
        return filtered
