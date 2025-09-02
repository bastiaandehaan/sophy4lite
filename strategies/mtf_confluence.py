# strategies/mtf_confluence.py
"""
Multi-Timeframe Confluence Strategy voor DAX
Combineert H1 bias, M15 structure, M5 setup en M1 entry timing
FIXED: Syntax errors, look-ahead bias, EV calculation
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from enum import Enum


class Bias(Enum):
    """Market bias from higher timeframe"""
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
    structure_lookback: int = 20  # bars voor support/resistance
    zone_buffer: float = 0.2  # ATR multiplier voor zones

    # Entry timing
    momentum_threshold: float = 0.6  # Minimum momentum voor entry

    # Risk management
    atr_mult_sl: float = 1.5
    atr_mult_tp: float = 2.5

    # Signal quality
    min_confluence_score: float = 0.65  # Minimum score voor trade

    # NIEUW: Adaptive win rate tracking
    use_adaptive_winrate: bool = True
    default_winrate: float = 0.5  # Conservatief: 50% default


class MTFSignals:
    """Multi-timeframe signal generator"""

    def __init__(self, params: MTFParams = MTFParams()):
        self.params = params
        self.win_rate_history: List[float] = []

    def create_timeframes(self, df_m1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creëer synthetische timeframes uit M1 data
        Returns dict met 'M1', 'M5', 'M15', 'H1'
        """
        if not isinstance(df_m1.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        # M1 is origineel
        tfs = {'M1': df_m1.copy()}

        # CRITICAL: Use label='right' to prevent look-ahead
        # This ensures each bar contains only data up to its close time
        tfs['M5'] = df_m1.resample('5min', label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum' if 'volume' in df_m1.columns else 'last'}).dropna()

        tfs['M15'] = df_m1.resample('15min', label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum' if 'volume' in df_m1.columns else 'last'}).dropna()

        tfs['H1'] = df_m1.resample('60min', label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum' if 'volume' in df_m1.columns else 'last'}).dropna()

        return tfs

    def calculate_bias(self, df_h1: pd.DataFrame) -> pd.Series:
        """H1 timeframe: Bepaal overall market bias"""
        ema = df_h1['close'].ewm(span=self.params.ema_period, adjust=False).mean()

        # ATR voor volatility context
        h, l, c = df_h1['high'], df_h1['low'], df_h1['close']
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        atr = tr.ewm(span=self.params.atr_period, adjust=False).mean()

        # Bias bepalen
        bias = pd.Series(index=df_h1.index, dtype=object)
        bias[c > ema + atr * 0.1] = Bias.BULLISH
        bias[c < ema - atr * 0.1] = Bias.BEARISH
        bias[bias.isna()] = Bias.NEUTRAL

        return bias

    def find_structure_levels(self, df_m15: pd.DataFrame) -> pd.DataFrame:
        """M15 timeframe: Identificeer support/resistance zones"""
        lookback = self.params.structure_lookback

        # Rolling high/low voor structure
        roll_high = df_m15['high'].rolling(lookback).max()
        roll_low = df_m15['low'].rolling(lookback).min()

        # ATR voor zone sizing
        h, l, c = df_m15['high'], df_m15['low'], df_m15['close']
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                       axis=1).max(axis=1)
        atr = tr.ewm(span=self.params.atr_period, adjust=False).mean()

        # Support/resistance zones (met buffer)
        buffer = atr * self.params.zone_buffer

        structure = pd.DataFrame(index=df_m15.index)
        structure['resistance_high'] = roll_high + buffer
        structure['resistance_low'] = roll_high - buffer
        structure['support_high'] = roll_low + buffer
        structure['support_low'] = roll_low - buffer
        structure['atr'] = atr

        return structure

    def detect_setup_patterns(self, df_m5: pd.DataFrame) -> pd.DataFrame:
        """M5 timeframe: Detecteer reversal patterns bij structure"""
        patterns = pd.DataFrame(index=df_m5.index)

        # Simplified pattern detection (meer robuust)
        # Bullish: Close near high + higher than previous
        patterns['bullish_momentum'] = ((df_m5['close'] > df_m5['open']) & (
                    df_m5['close'] > df_m5['close'].shift(1)) & (
                                                    (df_m5['close'] - df_m5['low']) > (
                                                        df_m5['high'] - df_m5[
                                                    'close']) * 2)).astype(float)

        # Bearish: Close near low + lower than previous  
        patterns['bearish_momentum'] = ((df_m5['close'] < df_m5['open']) & (
                    df_m5['close'] < df_m5['close'].shift(1)) & (
                                                    (df_m5['high'] - df_m5['close']) > (
                                                        df_m5['close'] - df_m5[
                                                    'low']) * 2)).astype(float)

        # Smooth patterns over 3 bars voor robuustheid
        patterns['bullish_strength'] = patterns['bullish_momentum'].rolling(3,
                                                                            min_periods=1).mean()
        patterns['bearish_strength'] = patterns['bearish_momentum'].rolling(3,
                                                                            min_periods=1).mean()

        return patterns

    def calculate_momentum(self, df_m1: pd.DataFrame, window: int = 10) -> pd.Series:
        """M1 timeframe: Bereken micro-momentum voor entry timing"""
        # Rate of change
        roc = df_m1['close'].pct_change(window)

        # Robuuste normalisatie
        rolling_std = roc.rolling(50, min_periods=20).std()
        momentum = roc / rolling_std.where(rolling_std > 1e-8, 1.0)
        momentum = momentum.clip(-2, 2) / 2  # Cap op ±2 std

        return momentum.fillna(0)

    def calculate_confluence_score(self, bias: Bias, at_support: bool,
            at_resistance: bool, pattern_strength: float, momentum: float) -> float:
        """Bereken confluence score [0,1] voor trade kwaliteit"""
        score = 0.0

        # Bias alignment (40% weight)
        if bias == Bias.BULLISH and at_support:
            score += 0.4
        elif bias == Bias.BEARISH and at_resistance:
            score += 0.4
        elif bias == Bias.NEUTRAL:
            score += 0.1

        # Pattern strength (30% weight)
        score += min(pattern_strength, 1.0) * 0.3

        # Momentum confirmation (30% weight)
        if bias == Bias.BULLISH and momentum > self.params.momentum_threshold:
            score += 0.3
        elif bias == Bias.BEARISH and momentum < -self.params.momentum_threshold:
            score += 0.3
        elif abs(momentum) > self.params.momentum_threshold:
            score += 0.15

        return min(score, 1.0)

    def get_adaptive_winrate(self) -> float:
        """Calculate adaptive win rate from history"""
        if not self.params.use_adaptive_winrate:
            return self.params.default_winrate

        if len(self.win_rate_history) < 20:  # Need minimum sample
            return self.params.default_winrate

        # Exponentially weighted average of recent win rates
        weights = np.exp(-np.arange(len(self.win_rate_history)) * 0.1)
        weights = weights / weights.sum()
        return float(np.average(self.win_rate_history, weights=weights))

    def calculate_expected_value(self, entry: float, sl: float, tp: float) -> float:
        """
        Bereken expected value van een trade setup
        FIXED: Gebruikt adaptive win rate of conservatieve default
        """
        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk <= 0:
            return 0

        # R-multiple (risk-reward ratio)
        r_multiple = reward / risk

        # Use adaptive or default win rate
        win_rate = self.get_adaptive_winrate()

        # Expected value in R-multiples
        # EV = P(win) * R_reward - P(loss) * 1R
        ev_r = (win_rate * r_multiple) - (1 - win_rate)

        return ev_r  # Returns expected R-multiple

    def generate_signals(self, df_m1: pd.DataFrame, session_start: str = "09:00",
            session_end: str = "17:00") -> pd.DataFrame:
        """
        Genereer MTF confluence signals
        FIXED: Proper forward-fill zonder look-ahead
        """
        # Create timeframes  
        tfs = self.create_timeframes(df_m1)

        # Calculate components per timeframe
        bias_h1 = self.calculate_bias(tfs['H1'])
        structure_m15 = self.find_structure_levels(tfs['M15'])
        patterns_m5 = self.detect_setup_patterns(tfs['M5'])
        momentum_m1 = self.calculate_momentum(tfs['M1'])

        # Initialize signals
        signals = pd.DataFrame(index=df_m1.index)

        # CRITICAL FIX: Use shift(1) before reindex to prevent look-ahead
        # This ensures we only see the COMPLETED higher timeframe bar
        signals['bias'] = bias_h1.shift(1).reindex(df_m1.index, method='ffill')
        signals['resistance'] = structure_m15['resistance_low'].shift(1).reindex(
            df_m1.index, method='ffill')
        signals['support'] = structure_m15['support_high'].shift(1).reindex(df_m1.index,
                                                                            method='ffill')
        signals['atr_m15'] = structure_m15['atr'].shift(1).reindex(df_m1.index,
                                                                   method='ffill')
        signals['bullish_pattern'] = patterns_m5['bullish_strength'].shift(1).reindex(
            df_m1.index, method='ffill')
        signals['bearish_pattern'] = patterns_m5['bearish_strength'].shift(1).reindex(
            df_m1.index, method='ffill')
        signals['momentum'] = momentum_m1  # M1 momentum is realtime

        # Check of price bij support/resistance is
        signals['at_support'] = ((df_m1['low'] <= signals['support']) & (
                    df_m1['close'] > signals['support'] - signals['atr_m15'] * 0.5))
        signals['at_resistance'] = ((df_m1['high'] >= signals['resistance']) & (
                    df_m1['close'] < signals['resistance'] + signals['atr_m15'] * 0.5))

        # Calculate confluence scores
        signals['long_score'] = signals.apply(
            lambda r: self.calculate_confluence_score(r['bias'], r['at_support'], False,
                r['bullish_pattern'], r['momentum']) if pd.notna(r['bias']) else 0,
            axis=1)

        signals['short_score'] = signals.apply(
            lambda r: self.calculate_confluence_score(r['bias'], False,
                r['at_resistance'], r['bearish_pattern'], -r['momentum']) if pd.notna(
                r['bias']) else 0, axis=1)

        # Session filter
        in_session = ((df_m1.index.hour >= int(session_start.split(':')[0])) & (
                    df_m1.index.hour < int(session_end.split(':')[0])))

        # Generate entry signals
        signals['long_entry'] = (in_session & (
                    signals['long_score'] >= self.params.min_confluence_score) & (
                                             signals[
                                                 'momentum'] > self.params.momentum_threshold))

        signals['short_entry'] = (in_session & (
                    signals['short_score'] >= self.params.min_confluence_score) & (
                                              signals[
                                                  'momentum'] < -self.params.momentum_threshold))

        # Calculate SL/TP levels with ATR floor
        atr_floor = 5.0  # Minimum ATR in points voor DAX
        atr_safe = signals['atr_m15'].fillna(atr_floor).clip(lower=atr_floor)

        signals['long_sl'] = df_m1['close'] - atr_safe * self.params.atr_mult_sl
        signals['long_tp'] = df_m1['close'] + atr_safe * self.params.atr_mult_tp
        signals['short_sl'] = df_m1['close'] + atr_safe * self.params.atr_mult_sl
        signals['short_tp'] = df_m1['close'] - atr_safe * self.params.atr_mult_tp

        # Expected value calculation
        signals['long_ev'] = signals.apply(
            lambda r: self.calculate_expected_value(df_m1.loc[r.name, 'close'],
                r['long_sl'], r['long_tp']) if r['long_entry'] else 0, axis=1)

        signals['short_ev'] = signals.apply(
            lambda r: self.calculate_expected_value(df_m1.loc[r.name, 'close'],
                r['short_sl'], r['short_tp']) if r['short_entry'] else 0, axis=1)

        return signals

    def filter_best_daily_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Filter voor max 1 trade per dag (beste confluence score)"""
        filtered = signals.copy()

        # Combineer scores
        filtered['best_score'] = filtered[['long_score', 'short_score']].max(axis=1)
        filtered['best_side'] = filtered.apply(
            lambda r: 'long' if r['long_score'] >= r['short_score'] else 'short',
            axis=1)

        # Group by date, keep best signal
        for day, day_data in filtered.groupby(filtered.index.date):
            if len(day_data) == 0:
                continue

            # Find best signal of the day
            best_idx = day_data['best_score'].idxmax()

            # Clear all signals except best
            day_indices = day_data.index
            filtered.loc[day_indices, 'long_entry'] = False
            filtered.loc[day_indices, 'short_entry'] = False

            # Set only the best signal
            if pd.notna(best_idx) and filtered.loc[best_idx, 'best_score'] > 0:
                if filtered.loc[best_idx, 'best_side'] == 'long':
                    filtered.loc[best_idx, 'long_entry'] = True
                else:
                    filtered.loc[best_idx, 'short_entry'] = True

        return filtered
































