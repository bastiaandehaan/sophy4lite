# backtest/breakout_exec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd

from config import logger
from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
from strategies.breakout_signals import opening_breakout_long
from utils.position import _size  # risk-based position sizing (float)


# ========= Config =========

@dataclass(frozen=True)
class BTExecCfg:
    equity0: float = 20_000.0           # startkapitaal
    fee_rate: float = 0.0002            # 2 bps per LEG (entry én exit)
    entry_slip_pts: float = 0.1         # adverse slip bij ENTRY (nieuw)
    sl_slip_pts: float = 0.5            # adverse slip bij SL
    tp_slip_pts: float = 0.0            # gunstig of neutraal bij TP
    specs: Dict[str, SymbolSpec] = None
    risk_frac: float = 0.01             # 1% per trade
    atr_n: int = 14                      # ATR lookback
    atr_floor: float = 1e-6              # minimale ATR om sizing te doen

    def get_spec(self, symbol: str) -> SymbolSpec:
        specs = self.specs or DEFAULT_SPECS
        if symbol in specs:
            return specs[symbol]
        base = symbol.split(".")[0]
        if base in specs:
            return specs[base]
        return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)


# ========= Helpers =========

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr.rename(f"ATR({n})")


def _entry_fee(notional: float, fee_rate: float) -> float:
    return abs(notional) * fee_rate  # enkele leg


def _exit_fee(notional: float, fee_rate: float) -> float:
    return abs(notional) * fee_rate  # enkele leg


def _sl_tp_on_entry(entry_px: float, atr_val: float, p: BreakoutParams) -> Tuple[float, float]:
    sl = entry_px - p.atr_mult_sl * atr_val
    tp = entry_px + p.atr_mult_tp * atr_val
    return sl, tp


# ========= Kern backtester =========

def backtest_breakout(
    df: pd.DataFrame,
    symbol: str,
    params: BreakoutParams,
    cfg: BTExecCfg,
    open_window_bars: int = 4,
    confirm: str = "close",
):
    """
    1 trade per dag: opening-breakout op previous-day high.
    Entry: eerste trigger in openingsvenster.
    Exit: SL of TP (ATR-multiples bij entry), intra-bar, SL prioriteit.
    Fees per leg (entry+exit), slippage (ENTRY/SL/TP apart), risk-based sizing.
    """
    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not req.issubset(df.columns):
        miss = req.difference(df.columns)
        raise KeyError(f"df mist kolommen: {sorted(miss)}")

    df = df.sort_index().copy()
    spec = cfg.get_spec(symbol)
    atr = _atr(df, n=cfg.atr_n)

    # Max 1 entry per dag in openingsvenster
    entries = opening_breakout_long(
        df["close"], df["high"], open_window_bars=open_window_bars, confirm=confirm
    )

    # Warmup: geen entries toestaan vóór een geldige ATR
    entries = entries & atr.notna()

    # Containers
    eq = pd.Series(index=df.index, dtype="float64")
    equity = float(cfg.equity0)
    trades = []

    in_pos = False
    entry_px = sl_px = tp_px = np.nan
    size = 0.0
    entry_time = None
    entry_leg_fee = 0.0  # fee op entry-notional

    for ts, row in df.iterrows():
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        a_val = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else None

        if not in_pos:
            # Entry: signaal + bruikbare ATR
            if entries.loc[ts] and (a_val is not None) and (a_val > cfg.atr_floor):
                sl_raw, tp_raw = _sl_tp_on_entry(c, a_val, params)
                if sl_raw >= c:  # guard tegen pathologische data
                    eq.loc[ts] = equity
                    continue

                this_size = _size(equity, c, sl_raw, spec.point_value, cfg.risk_frac)
                if this_size <= 0:
                    eq.loc[ts] = equity
                    continue

                # Slippage per type: ENTRY
                effective_entry = c + cfg.entry_slip_pts  # adverse richting voor long entry
                entry_px = effective_entry
                sl_px, tp_px = sl_raw, tp_raw
                size = this_size
                entry_time = ts
                in_pos = True

                notional = entry_px * size * spec.point_value
                entry_leg_fee = _entry_fee(notional, cfg.fee_rate)

                eq.loc[ts] = equity
                continue

            eq.loc[ts] = equity
            continue

        # In positie: SL -> TP
        exit_reason = None
        exit_px = None

        if l <= sl_px:
            exit_reason = "SL"
            exit_px = sl_px - cfg.sl_slip_pts  # adverse slip bij stop
        elif h >= tp_px:
            exit_reason = "TP"
            exit_px = tp_px + cfg.tp_slip_pts  # niet-nadelig (0 of klein positief)

        if exit_reason is None:
            eq.loc[ts] = equity
            continue

        # Realiseer P&L + exit fee
        notional_exit = exit_px * size * spec.point_value
        pnl_per_contract = (exit_px - entry_px) * spec.point_value
        gross = pnl_per_contract * size
        net = gross - entry_leg_fee - _exit_fee(notional_exit, cfg.fee_rate)

        equity += net

        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_px": entry_px,
            "exit_time": ts,
            "exit_px": exit_px,
            "reason": exit_reason,
            "size": size,
            "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
            "sl_px": sl_px,
            "tp_px": tp_px,
            "pnl": net,
            "gross": gross,
            "fees": entry_leg_fee + _exit_fee(notional_exit, cfg.fee_rate),
        })

        # Reset
        in_pos = False
        entry_px = sl_px = tp_px = np.nan
        size = 0.0
        entry_time = None
        entry_leg_fee = 0.0
        eq.loc[ts] = equity

    # Sluit open positie op laatste bar
    if in_pos and entry_time is not None:
        last_ts = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        # conservatief: sluit tegen last_close met entry-slip voor adverse richting
        exit_px = last_close - cfg.sl_slip_pts

        notional_exit = exit_px * size * spec.point_value
        pnl_per_contract = (exit_px - entry_px) * spec.point_value
        gross = pnl_per_contract * size
        net = gross - entry_leg_fee - _exit_fee(notional_exit, cfg.fee_rate)

        equity += net
        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_px": entry_px,
            "exit_time": last_ts,
            "exit_px": exit_px,
            "reason": "EOD",
            "size": size,
            "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
            "sl_px": sl_px,
            "tp_px": tp_px,
            "pnl": net,
            "gross": gross,
            "fees": entry_leg_fee + _exit_fee(notional_exit, cfg.fee_rate),
        })
        eq.iloc[-1] = equity

    # Equity carry
    eq = eq.ffill().fillna(cfg.equity0).rename("equity")

    trades_df = (
        pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
        if trades else pd.DataFrame(columns=[
            "symbol","entry_time","entry_px","exit_time","exit_px","reason",
            "size","atr_at_entry","sl_px","tp_px","pnl","gross","fees"
        ])
    )

    # Metrics
    def _max_dd(series: pd.Series):
        if len(series) == 0:
            return 0.0, 0
        roll_max = series.cummax()
        dd = series / roll_max - 1.0
        min_dd = float(dd.min())
        under = dd < 0
        dur = int(pd.Series(np.where(under, 1, 0), index=series.index)
                  .groupby((~under).cumsum()).sum().max() or 0)
        return min_dd * 100.0, dur

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) else 0.0
    max_dd_pct, dd_dur = _max_dd(eq)

    bar_rets = eq.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sd = float(bar_rets.std(ddof=1))
    if sd > 0:
        bars_per_day = df.index.normalize().value_counts().mean() if len(df) else 60.0
        sharpe = float(bar_rets.mean() / sd * math.sqrt(bars_per_day * 252.0))
    else:
        sharpe = 0.0

    metrics = {
        "n_trades": int(len(trades_df)),
        "final_equity": float(eq.iloc[-1]) if len(eq) else cfg.equity0,
        "return_total_pct": total_ret,
        "max_drawdown_pct": max_dd_pct,
        "dd_duration_bars": dd_dur,
        "sharpe": sharpe,
    }

    return eq, trades_df, metrics
