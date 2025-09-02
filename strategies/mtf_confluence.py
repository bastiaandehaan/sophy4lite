Multi - Timeframe
Confluence
Strategy
voor
DAX
Combineert
H1
bias, M15
structure, M5
setup
en
M1
entry
timing
Met
expectancy - based
scoring
en
FTMO
risk
management
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
from enum import Enum

class Bias(Enum):
    """
Market
bias
from higher timeframe

"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

@dataclass(frozen=True)
class MTFParams:
    """
Multi - timeframe
strategy
parameters
"""
    # Trend detection
    ema_period: int = 20
    atr_period: int = 14

    # Structure levels
    structure_lookback: int = 20  # bars voor support/resistance
    zone_buffer: float = 0.2      # ATR multiplier voor zones

    # Entry timing
    momentum_threshold: float = 0.6  # Minimum momentum voor entry

    # Risk management
    atr_mult_sl: float = 1.5
    atr_mult_tp: float = 2.5

    # Signal quality
    min_confluence_score: float = 0.65  # Minimum score voor trade

class MTFSignals:
    """
Multi - timeframe
signal
generator
"""

    def __init__(self, params: MTFParams = MTFParams()):
        self.params = params
        self.expectancy_history: List[float] = []

    def create_timeframes(self, df_m1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
Creëer
synthetische
timeframes
uit
M1
data
Retourneert
dict
met
'M1', 'M5', 'M15', 'H1'
"""
if not isinstance(df_m1.index, pd.DatetimeIndex):
    raise TypeError("Index must be DatetimeIndex")

# M1 is origineel
tfs = {'M1': df_m1.copy()}

# M5: 5-min bars
tfs['M5'] = df_m1.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum' if 'volume' in df_m1.columns else 'last'
}).dropna()

# M15: 15-min bars
tfs['M15'] = df_m1.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum' if 'volume' in df_m1.columns else 'last'
}).dropna()

# H1: 60-min bars
tfs['H1'] = df_m1.resample('60min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum' if 'volume' in df_m1.columns else 'last'
}).dropna()

return tfs

def calculate_bias(self, df_h1: pd.DataFrame) -> pd.Series:
"""
H1
timeframe: Bepaal
overall
market
bias
Returns
Series
met
Bias
enum
values
"""
ema = df_h1['close'].ewm(span=self.params.ema_period, adjust=False).mean()

# ATR voor volatility context
h = df_h1['high']
l = df_h1['low']
c = df_h1['close']
tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(span=self.params.atr_period, adjust=False).mean()

# Bias bepalen
bias = pd.Series(index=df_h1.index, dtype=object)

# Bullish: close > EMA + kleine buffer
bias[c > ema + atr * 0.1] = Bias.BULLISH

# Bearish: close < EMA - kleine buffer
bias[c < ema - atr * 0.1] = Bias.BEARISH

# Neutral: te dicht bij EMA
bias[bias.isna()] = Bias.NEUTRAL

return bias

def find_structure_levels(self, df_m15: pd.DataFrame) -> pd.DataFrame:
"""
M15
timeframe: Identificeer
support / resistance
zones
Returns
DataFrame
met
support / resistance
levels
per
bar
"""
lookback = self.params.structure_lookback

# Rolling high/low voor structure
roll_high = df_m15['high'].rolling(lookback).max()
roll_low = df_m15['low'].rolling(lookback).min()

# ATR voor zone sizing
h = df_m15['high']
l = df_m15['low']
c = df_m15['close']
tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
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
"""
M5
timeframe: Detecteer
reversal
patterns
bij
structure
Returns
DataFrame
met
pattern
signals
"""
patterns = pd.DataFrame(index=df_m5.index)

# Bullish patterns
patterns['hammer'] = (
    (df_m5['close'] > df_m5['open']) &
    ((df_m5['high'] - df_m5['close']) < (df_m5['close'] - df_m5['open']) * 0.3) &
    ((df_m5['open'] - df_m5['low']) > (df_m5['close'] - df_m5['open']) * 2)
)

patterns['bullish_engulfing'] = (
    (df_m5['close'] > df_m5['open']) &
    (df_m5['open'].shift() > df_m5['close'].shift()) &
    (df_m5['close'] > df_m5['open'].shift()) &
    (df_m5['open'] < df_m5['close'].shift())
)

# Bearish patterns
patterns['shooting_star'] = (
    (df_m5['open'] > df_m5['close']) &
    ((df_m5['low'] - df_m5['close']) < (df_m5['open'] - df_m5['close']) * 0.3) &
    ((df_m5['high'] - df_m5['open']) > (df_m5['open'] - df_m5['close']) * 2)
)

patterns['bearish_engulfing'] = (
    (df_m5['open'] > df_m5['close']) &
    (df_m5['close'].shift() > df_m5['open'].shift()) &
    (df_m5['open'] > df_m5['close'].shift()) &
    (df_m5['close'] < df_m5['open'].shift())
)

# Aggregate pattern strength
patterns['bullish_strength'] = (
    patterns['hammer'].astype(int) + 
    patterns['bullish_engulfing'].astype(int)
) / 2.0

patterns['bearish_strength'] = (
    patterns['shooting_star'].astype(int) + 
    patterns['bearish_engulfing'].astype(int)
) / 2.0

return patterns

def calculate_momentum(self, df_m1: pd.DataFrame, window: int = 10) -> pd.Series:
"""
M1
timeframe: Bereken
micro - momentum
voor
entry
timing
"""
# Rate of change
roc = df_m1['close'].pct_change(window)

# Normalize naar [-1, 1]
momentum = roc / roc.rolling(50).std()
momentum = momentum.clip(-2, 2) / 2  # Cap op ±2 std, scale naar ±1

return momentum.fillna(0)

def calculate_confluence_score(
self,
bias: Bias,
at_support: bool,
at_resistance: bool,
pattern_strength: float,
momentum: float
) -> float:
"""
Bereken
confluence
score[0, 1]
voor
trade
kwaliteit
"""
score = 0.0

# Bias alignment (40% weight)
if bias == Bias.BULLISH and at_support:
    score += 0.4
elif bias == Bias.BEARISH and at_resistance:
    score += 0.4
elif bias == Bias.NEUTRAL:
    score += 0.1  # Kleine score voor range trading

# Pattern strength (30% weight)
score += pattern_strength * 0.3

# Momentum confirmation (30% weight)
if bias == Bias.BULLISH and momentum > self.params.momentum_threshold:
    score += 0.3
elif bias == Bias.BEARISH and momentum < -self.params.momentum_threshold:
    score += 0.3
elif abs(momentum) > self.params.momentum_threshold:
    score += 0.15  # Half score voor sterke momentum zonder bias

return min(score, 1.0)

def calculate_expected_value(
self,
entry: float,
sl: float,
tp: float,
win_rate: float = 0.58  # Historische winrate
) -> float:
"""
Bereken
expected
value
van
een
trade
setup
"""
risk = abs(entry - sl)
reward = abs(tp - entry)

if risk == 0:
    return 0

# Expected value = (P(win) × reward) - (P(loss) × risk)
ev = (win_rate * reward) - ((1 - win_rate) * risk)

# Normalize naar R-multiple
return ev / risk

def generate_signals(
self,
df_m1: pd.DataFrame,
session_start: str = "09:00",
session_end: str = "17:00"
) -> pd.DataFrame:
"""
Genereer
MTF
confluence
signals
Returns
DataFrame
met
entry
signals
en
metadata
"""
# Create timeframes
tfs = self.create_timeframes(df_m1)

# Calculate components per timeframe
bias_h1 = self.calculate_bias(tfs['H1'])
structure_m15 = self.find_structure_levels(tfs['M15'])
patterns_m5 = self.detect_setup_patterns(tfs['M5'])
momentum_m1 = self.calculate_momentum(tfs['M1'])

# Align alle data naar M1 timeframe
signals = pd.DataFrame(index=df_m1.index)

# Forward fill higher timeframe data naar M1
signals['bias'] = bias_h1.reindex(df_m1.index, method='ffill')
signals['resistance'] = structure_m15['resistance_low'].reindex(df_m1.index, method='ffill')
signals['support'] = structure_m15['support_high'].reindex(df_m1.index, method='ffill')
signals['atr_m15'] = structure_m15['atr'].reindex(df_m1.index, method='ffill')
signals['bullish_pattern'] = patterns_m5['bullish_strength'].reindex(df_m1.index, method='ffill')
signals['bearish_pattern'] = patterns_m5['bearish_strength'].reindex(df_m1.index, method='ffill')
signals['momentum'] = momentum_m1

# Check of price bij support/resistance is
signals['at_support'] = (
    (df_m1['low'] <= signals['support']) & 
    (df_m1['close'] > signals['support'] - signals['atr_m15'] * 0.5)
)
signals['at_resistance'] = (
    (df_m1['high'] >= signals['resistance']) & 
    (df_m1['close'] < signals['resistance'] + signals['atr_m15'] * 0.5)
)

# Calculate confluence scores
signals['long_score'] = signals.apply(
    lambda r: self.calculate_confluence_score(
        r['bias'],
        r['at_support'],
        False,
        r['bullish_pattern'],
        r['momentum']
    ) if pd.notna(r['bias']) else 0,
    axis=1
)

signals['short_score'] = signals.apply(
    lambda r: self.calculate_confluence_score(
        r['bias'],
        False,
        r['at_resistance'],
        r['bearish_pattern'],
        -r['momentum']
    ) if pd.notna(r['bias']) else 0,
    axis=1
)

# Filter op sessie tijden
in_session = (
    (df_m1.index.hour >= int(session_start.split(':')[0])) &
    (df_m1.index.hour < int(session_end.split(':')[0]))
)

# Generate signals alleen tijdens sessie met minimum score
signals['long_entry'] = (
    in_session &
    (signals['long_score'] >= self.params.min_confluence_score) &
    (signals['momentum'] > self.params.momentum_threshold)
)

signals['short_entry'] = (
    in_session &
    (signals['short_score'] >= self.params.min_confluence_score) &
    (signals['momentum'] < -self.params.momentum_threshold)
)

# Calculate SL/TP levels
signals['long_sl'] = df_m1['close'] - signals['atr_m15'] * self.params.atr_mult_sl
signals['long_tp'] = df_m1['close'] + signals['atr_m15'] * self.params.atr_mult_tp
signals['short_sl'] = df_m1['close'] + signals['atr_m15'] * self.params.atr_mult_sl
signals['short_tp'] = df_m1['close'] - signals['atr_m15'] * self.params.atr_mult_tp

# Expected value calculation
signals['long_ev'] = signals.apply(
    lambda r: self.calculate_expected_value(
        df_m1.loc[r.name, 'close'],
        r['long_sl'],
        r['long_tp']
    ) if r['long_entry'] else 0,
    axis=1
)

signals['short_ev'] = signals.apply(
    lambda r: self.calculate_expected_value(
        df_m1.loc[r.name, 'close'],
        r['short_sl'],
        r['short_tp']
    ) if r['short_entry'] else 0,
    axis=1
)

return signals

def filter_best_daily_signal(self, signals: pd.DataFrame) -> pd.DataFrame:
"""
Filter
voor
max
1
trade
per
dag(beste
confluence
score)
"""
filtered = signals.copy()

# Combineer long en short scores
filtered['best_score'] = filtered[['long_score', 'short_score']].max(axis=1)
filtered['best_side'] = filtered.apply(
    lambda r: 'long' if r['long_score'] > r['short_score'] else 'short',
    axis=1
)

# Group by date, keep best
daily_best = filtered.groupby(filtered.index.date).apply(
    lambda g: g.nlargest(1, 'best_score').index[0] if len(g) > 0 else None
).dropna()

# Reset signals, alleen beste per dag behouden
filtered['long_entry'] = False
filtered['short_entry'] = False

for idx in daily_best:
    if filtered.loc[idx, 'best_side'] == 'long':
        filtered.loc[idx, 'long_entry'] = True
    else:
        filtered.loc[idx, 'short_entry'] = True

return filtered


# backtest/mtf_exec.py
"""
Backtest
executor
voor
MTF
confluence
strategy
Integreert
met
Sophy4Lite
's bestaande infrastructure
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np
import pandas as pd

from strategies.mtf_confluence import MTFSignals, MTFParams
from strategies.breakout_params import SymbolSpec, DEFAULT_SPECS
from utils.position import _size

@dataclass(frozen=True)
class MTFExecCfg:
    """
Config
voor
MTF
backtest
execution
"""
    equity0: float = 20_000.0
    fee_rate: float = 0.0002      # 2 bps per leg
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    specs: Dict[str, SymbolSpec] = None
    risk_frac: float = 0.01        # 1% risk per trade (FTMO safe)

    def get_spec(self, symbol: str) -> SymbolSpec:
        specs = self.specs or DEFAULT_SPECS
        if symbol in specs:
            return specs[symbol]
        base = symbol.split(".")[0]
        if base in specs:
            return specs[base]
        return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)

def backtest_mtf_confluence(
    df: pd.DataFrame,
    symbol: str,
    params: MTFParams,
    cfg: MTFExecCfg,
    session_start: str = "09:00",
    session_end: str = "17:00",
    max_trades_per_day: int = 1
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """
Backtest
MTF
confluence
strategy

Returns:
- equity: Series
met
equity
curve
- trades: DataFrame
met
trade
details
- metrics: Dict
met
performance
metrics
"""
req = {"open", "high", "low", "close"}
if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError("Index must be DatetimeIndex")
if not req.issubset(df.columns):
    miss = req.difference(df.columns)
    raise KeyError(f"DataFrame missing columns: {sorted(miss)}")

df = df.sort_index().copy()
spec = cfg.get_spec(symbol)

# Generate MTF signals
mtf = MTFSignals(params)
signals = mtf.generate_signals(df, session_start, session_end)

if max_trades_per_day == 1:
    signals = mtf.filter_best_daily_signal(signals)

# Initialize tracking
eq = pd.Series(index=df.index, dtype="float64")
equity = float(cfg.equity0)
trades = []

in_pos = False
side = None
entry_px = sl_px = tp_px = np.nan
size = 0.0
entry_time = None
entry_score = 0.0
entry_ev = 0.0
entry_leg_fee = 0.0

# Process bars
for ts, row in df.iterrows():
    h, l, c = float(row["high"]), float(row["low"]), float(row["close"])

    if not in_pos:
        # Check for entry signals
        if signals.loc[ts, 'long_entry']:
            side = "long"
            effective_entry = c + cfg.entry_slip_pts
            sl_raw = signals.loc[ts, 'long_sl']
            tp_raw = signals.loc[ts, 'long_tp']
            entry_score = signals.loc[ts, 'long_score']
            entry_ev = signals.loc[ts, 'long_ev']

        elif signals.loc[ts, 'short_entry']:
            side = "short"
            effective_entry = c - cfg.entry_slip_pts
            sl_raw = signals.loc[ts, 'short_sl']
            tp_raw = signals.loc[ts, 'short_tp']
            entry_score = signals.loc[ts, 'short_score']
            entry_ev = signals.loc[ts, 'short_ev']
        else:
            eq.loc[ts] = equity
            continue

        # Validate and size position
        if (side == "long" and sl_raw >= effective_entry) or \
           (side == "short" and sl_raw <= effective_entry):
            eq.loc[ts] = equity
            continue

        this_size = _size(equity, effective_entry, sl_raw, spec.point_value, cfg.risk_frac)
        if this_size <= 0:
            eq.loc[ts] = equity
            continue

        # Enter position
        entry_px = effective_entry
        sl_px = sl_raw
        tp_px = tp_raw
        size = this_size
        entry_time = ts
        in_pos = True

        notional = entry_px * size * spec.point_value
        entry_leg_fee = abs(notional) * cfg.fee_rate

        eq.loc[ts] = equity
        continue

    # In position - check exits
    exit_reason = None
    exit_px = None

    if side == "long":
        if l <= sl_px:
            exit_reason, exit_px = "SL", sl_px - cfg.sl_slip_pts
        elif h >= tp_px:
            exit_reason, exit_px = "TP", tp_px + cfg.tp_slip_pts
    else:  # short
        if h >= sl_px:
            exit_reason, exit_px = "SL", sl_px + cfg.sl_slip_pts
        elif l <= tp_px:
            exit_reason, exit_px = "TP", tp_px - cfg.tp_slip_pts

    if exit_reason is None:
        eq.loc[ts] = equity
        continue

    # Calculate P&L
    notional_exit = exit_px * size * spec.point_value
    if side == "long":
        pnl_per_contract = (exit_px - entry_px) * spec.point_value
    else:
        pnl_per_contract = (entry_px - exit_px) * spec.point_value

    gross = pnl_per_contract * size
    exit_fee = abs(notional_exit) * cfg.fee_rate
    net = gross - entry_leg_fee - exit_fee

    equity += net

    # Record trade
    trades.append({
        "symbol": symbol,
        "side": side,
        "entry_time": entry_time,
        "entry_px": entry_px,
        "exit_time": ts,
        "exit_px": exit_px,
        "reason": exit_reason,
        "size": size,
        "sl_px": sl_px,
        "tp_px": tp_px,
        "pnl": net,
        "gross": gross,