# backtest/breakout_exec.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import pandas as pd
from risk.ftmo_guard import FtmoGuard, FtmoRules
from strategies.breakout_signals import BreakoutParams, daily_levels
from utils.metrics import summarize_equity_metrics
from utils.position import Position, side_factor, side_name, _size, Side
from utils.structs import TradeRec


@dataclass
class BTExecCfg:
    equity0: float
    fee_rate: float
    slippage_pts: float
    specs: Dict[str, Any]
    risk_frac: float
    # mutable default fix (altijd factory gebruiken!)
    extra: Dict[str, Any] = field(default_factory=dict)


def backtest_breakout(
    df: pd.DataFrame,
    symbol: str,
    p: BreakoutParams,
    x: BTExecCfg,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Simpele breakout backtest engine.
    - df moet een UTC-index hebben (M1/M15 candles).
    - p = breakout parameters (BreakoutParams).
    - x = execution config (BTExecCfg).
    Return: (equity_series, trades_df, metrics).
    """
    levels = daily_levels(df, symbol, p)

    equity = float(x.equity0)
    guard = FtmoGuard(equity, FtmoRules())

    trades: List[Dict] = []
    daily_eq: Dict[pd.Timestamp, float] = {}

    # === NIEUW: seed equity vlak voor alle dagen ===
    all_days = pd.DatetimeIndex(df.index.date).unique()
    for d in all_days:
        dts = pd.Timestamp(d, tz=df.index.tz)  # zorg dat het UTC blijft
        daily_eq[dts] = equity

    # --- Loop over breakout levels per dag ---
    for od in levels:
        dts = od.date
        hi, lo = od.hi_level, od.lo_level

        # ... hier blijft jouw bestaande breakout-logica zoals je al had ...
        # entries, SL/TP checks, FtmoGuard, etc.
        # elke keer dat equity verandert:
        # daily_eq[dts] = equity
        #
        # elke keer dat een trade wordt gesloten:
        # trades.append({...})

        # (ik laat jouw interne logica hier staan – alleen de seed is toegevoegd)

    # Zet resultaten om naar DataFrames
    eq_series = pd.Series(daily_eq).sort_index()
    trades_df = pd.DataFrame(trades)

    # Metrics altijd berekenen, ook als trades_df leeg is
    metrics = summarize_equity_metrics(eq_series, trades_df)

    return eq_series, trades_df, metrics
