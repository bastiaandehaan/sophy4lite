# backtest/orb_exec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np
import pandas as pd

from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
from utils.position import _size

@dataclass(frozen=True)
class ORBExecCfg:
    equity0: float = 20_000.0
    fee_rate: float = 0.0002      # per leg
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    specs: Dict[str, SymbolSpec] = None
    risk_frac: float = 0.01
    atr_n: int = 14
    atr_floor: float = 1e-6

    def get_spec(self, symbol: str) -> SymbolSpec:
        specs = self.specs or DEFAULT_SPECS
        if symbol in specs:
            return specs[symbol]
        base = symbol.split(".")[0]
        if base in specs:
            return specs[base]
        return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().rename(f"ATR({n})")

def backtest_orb_bidirectional(
    df: pd.DataFrame,
    symbol: str,
    params: BreakoutParams,
    cfg: ORBExecCfg,
    entries_long: pd.Series,
    entries_short: pd.Series,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """
    Backtest pre-market ORB met long Ã©n short entries.
    Er wordt max 1 trade per dag gezet (eerste trigger wint).
    Intraâ€‘bar SLâ€‘prioriteit, ATRâ€‘based SL/TP, fees en slippage zoals in breakout_exec.
    """
    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not req.issubset(df.columns):
        miss = req.difference(df.columns)
        raise KeyError(f"df mist kolommen: {sorted(miss)}")
    if not entries_long.index.equals(df.index) or not entries_short.index.equals(df.index):
        raise ValueError("entries index mismatch met df.index")

    df = df.sort_index().copy()
    spec = cfg.get_spec(symbol)
    atr = _atr(df, n=cfg.atr_n)

    # ATRâ€‘warmup: geen entries als atr NaN of < floor
    entries_long = entries_long & atr.notna() & (atr > cfg.atr_floor)
    entries_short = entries_short & atr.notna() & (atr > cfg.atr_floor)

    eq = pd.Series(index=df.index, dtype="float64")
    equity = float(cfg.equity0)
    trades = []

    in_pos = False
    side = None  # "long" of "short"
    entry_px = sl_px = tp_px = np.nan
    size = 0.0
    entry_time = None
    entry_leg_fee = 0.0

    for ts, row in df.iterrows():
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        a_val = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else None

        if not in_pos:
            go_long = entries_long.loc[ts]
            go_short = entries_short.loc[ts]

            if (go_long or go_short) and (a_val is not None):
                side = "long" if go_long and not go_short else "short" if go_short and not go_long else "long"
                effective_entry = c + cfg.entry_slip_pts if side == "long" else c - cfg.entry_slip_pts
                if side == "long":
                    sl_raw = effective_entry - params.atr_mult_sl * a_val
                    tp_raw = effective_entry + params.atr_mult_tp * a_val
                else:
                    sl_raw = effective_entry + params.atr_mult_sl * a_val
                    tp_raw = effective_entry - params.atr_mult_tp * a_val
                if (side == "long" and sl_raw >= effective_entry) or (side == "short" and sl_raw <= effective_entry):
                    eq.loc[ts] = equity
                    continue
                this_size = _size(equity, effective_entry, sl_raw, spec.point_value, cfg.risk_frac)
                if this_size <= 0:
                    eq.loc[ts] = equity
                    continue

                entry_px = effective_entry
                sl_px, tp_px = sl_raw, tp_raw
                size = this_size
                entry_time = ts
                in_pos = True

                notional = entry_px * size * spec.point_value
                entry_leg_fee = abs(notional) * cfg.fee_rate

                eq.loc[ts] = equity
                continue

            eq.loc[ts] = equity
            continue

        # in positie â†’ SL first, dan TP
        exit_reason = None
        exit_px = None

        if side == "long":
            if l <= sl_px:
                exit_reason, exit_px = "SL", sl_px - cfg.sl_slip_pts
            elif h >= tp_px:
                exit_reason, exit_px = "TP", tp_px + cfg.tp_slip_pts
        else:
            if h >= sl_px:
                exit_reason, exit_px = "SL", sl_px + cfg.sl_slip_pts
            elif l <= tp_px:
                exit_reason, exit_px = "TP", tp_px - cfg.tp_slip_pts

        if exit_reason is None:
            eq.loc[ts] = equity
            continue

        notional_exit = exit_px * size * spec.point_value
        pnl_per_contract = (exit_px - entry_px) * spec.point_value if side == "long" else (entry_px - exit_px) * spec.point_value
        gross = pnl_per_contract * size
        net = gross - entry_leg_fee - abs(notional_exit) * cfg.fee_rate

        equity += net
        trades.append({
            "symbol": symbol, "side": side, "entry_time": entry_time,
            "entry_px": entry_px, "exit_time": ts, "exit_px": exit_px,
            "reason": exit_reason, "size": size, "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
            "sl_px": sl_px, "tp_px": tp_px, "pnl": net, "gross": gross,
            "fees": entry_leg_fee + abs(notional_exit) * cfg.fee_rate
        })

        in_pos = False
        side = None
        entry_px = sl_px = tp_px = np.nan
        size = 0.0
        entry_time = None
        entry_leg_fee = 0.0
        eq.loc[ts] = equity

    # eindbar â€“ conservatief sluiten
    if in_pos and entry_time is not None:
        last_ts = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        exit_px = (last_close - cfg.sl_slip_pts) if side == "long" else (last_close + cfg.sl_slip_pts)
        notional_exit = exit_px * size * spec.point_value
        pnl_per_contract = (exit_px - entry_px) * spec.point_value if side == "long" else (entry_px - exit_px) * spec.point_value
        gross = pnl_per_contract * size
        net = gross - entry_leg_fee - abs(notional_exit) * cfg.fee_rate
        equity += net

        trades.append({
            "symbol": symbol, "side": side or "NA", "entry_time": entry_time,
            "entry_px": entry_px, "exit_time": last_ts, "exit_px": exit_px, "reason": "EOD",
            "size": size, "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
            "sl_px": sl_px, "tp_px": tp_px, "pnl": net, "gross": gross,
            "fees": entry_leg_fee + abs(notional_exit) * cfg.fee_rate
        })
        eq.iloc[-1] = equity

    eq = eq.ffill().fillna(cfg.equity0).rename("equity")
    trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)

    def _max_dd(series: pd.Series):
        if len(series) == 0:
            return 0.0, 0
        roll_max = series.cummax()
        dd = series / roll_max - 1.0
        min_dd = float(dd.min())
        under = dd < 0
        dur = int(pd.Series(np.where(under, 1, 0), index=series.index).groupby((~under).cumsum()).sum().max() or 0)
        return min_dd * 100.0, dur

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) else 0.0
    max_dd_pct, dd_dur = _max_dd(eq)
    bar_rets = eq.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if bar_rets.std(ddof=1) > 0:
        bars_per_day = df.index.normalize().value_counts().mean() if len(df) else 60.0
        sharpe = float(bar_rets.mean() / bar_rets.std(ddof=1) * math.sqrt(bars_per_day * 252.0))
    else:
        sharpe = 0.0

    metrics = {
        "n_trades": int(len(trades_df)),
        "final_equity": float(eq.iloc[-1]),
        "return_total_pct": total_ret,
        "max_drawdown_pct": max_dd_pct,
        "dd_duration_bars": dd_dur,
        "sharpe": sharpe,
    }
    return eq, trades_df, metrics


