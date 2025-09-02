# backtest/mtf_exec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from strategies.mtf_confluence import MTFSignals, MTFParams
from utils.position import _size
from utils.specs import get_spec, SymbolSpec


@dataclass(frozen=True)
class MTFExecCfg:
    """Config for MTF backtest execution."""
    equity0: float = 20_000.0
    fee_rate: float = 0.0002      # 2 bps per leg
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    risk_frac: float = 0.01       # 1% per trade
    atr_floor: float = 1e-6       # guard voor ATR=0


def backtest_mtf_confluence(
    df: pd.DataFrame,
    symbol: str,
    params: MTFParams,
    cfg: MTFExecCfg,
    session_start: str = "09:00",
    session_end: str = "17:00",
    max_trades_per_day: int = 1,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """
    Backtest Multi-Timeframe Confluence strategie.
    Returns:
      - equity: Series (tz-aware index)
      - trades: DataFrame
      - metrics: dict
    """
    # --- Input checks ---
    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    miss = req.difference(df.columns)
    if miss:
        raise KeyError(f"DataFrame missing columns: {sorted(miss)}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    df = df.sort_index().copy()
    spec: SymbolSpec = get_spec(symbol)

    # --- Signals ---
    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_start, session_end)
    if max_trades_per_day == 1:
        signals = mtf.filter_best_daily_signal(signals)

    # --- State ---
    eq = pd.Series(index=df.index, dtype="float64")
    equity = float(cfg.equity0)
    trades = []

    in_pos = False
    side = None  # "long" | "short"
    entry_px = sl_px = tp_px = np.nan
    size = 0.0
    entry_time = None
    entry_score = 0.0
    entry_ev = 0.0
    entry_leg_fee = 0.0

    def _notional(px: float, qty: float) -> float:
        return px * qty * spec.point_value

    # --- Iterate bars ---
    for ts, row in df.iterrows():
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        if not in_pos:
            go_long = bool(signals.loc[ts, "long_entry"])
            go_short = bool(signals.loc[ts, "short_entry"])

            if not (go_long or go_short):
                eq.loc[ts] = equity
                continue

            if go_long:
                side = "long"
                effective_entry = c + cfg.entry_slip_pts
                sl_raw = float(signals.loc[ts, "long_sl"])
                tp_raw = float(signals.loc[ts, "long_tp"])
                entry_score = float(signals.loc[ts, "long_score"])
                entry_ev = float(signals.loc[ts, "long_ev"])
            else:
                side = "short"
                effective_entry = c - cfg.entry_slip_pts
                sl_raw = float(signals.loc[ts, "short_sl"])
                tp_raw = float(signals.loc[ts, "short_tp"])
                entry_score = float(signals.loc[ts, "short_score"])
                entry_ev = float(signals.loc[ts, "short_ev"])

            # Must define positive risk to size position
            if (side == "long" and sl_raw >= effective_entry) or (side == "short" and sl_raw <= effective_entry):
                eq.loc[ts] = equity
                side = None
                continue

            atr_val = float(signals.loc[ts, "atr_m15"])
            if not np.isfinite(atr_val) or atr_val < cfg.atr_floor:
                eq.loc[ts] = equity
                side = None
                continue

            # Size with broker lot rounding
            this_size = _size(
                equity,
                effective_entry,
                sl_raw,
                spec.point_value,
                cfg.risk_frac,
                lot_step=getattr(spec, "lot_step", 0.01),
            )
            if this_size <= 0:
                eq.loc[ts] = equity
                side = None
                continue

            # Enter
            entry_px = effective_entry
            sl_px = sl_raw
            tp_px = tp_raw
            size = this_size
            entry_time = ts
            in_pos = True

            entry_leg_fee = abs(_notional(entry_px, size)) * cfg.fee_rate
            eq.loc[ts] = equity
            continue

        # In position: evaluate exits
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

        # PnL + fees
        if side == "long":
            pnl_per_contract = (exit_px - entry_px) * spec.point_value
        else:
            pnl_per_contract = (entry_px - exit_px) * spec.point_value

        gross = pnl_per_contract * size
        exit_fee = abs(_notional(exit_px, size)) * cfg.fee_rate
        net = gross - entry_leg_fee - exit_fee
        equity += net

        trades.append(
            {
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
                "score": entry_score,
                "ev_R": entry_ev,
            }
        )

        # Reset state
        in_pos = False
        side = None
        entry_px = sl_px = tp_px = np.nan
        size = 0.0
        entry_time = None
        entry_score = 0.0
        entry_ev = 0.0
        entry_leg_fee = 0.0

        eq.loc[ts] = equity

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "symbol", "side", "entry_time", "entry_px", "exit_time", "exit_px",
            "reason", "size", "sl_px", "tp_px", "pnl", "gross", "score", "ev_R",
        ],
    )

    # Metrics
    n = int(len(trades_df))
    wins = int((trades_df["pnl"] > 0).sum()) if n else 0
    losses = int((trades_df["pnl"] < 0).sum()) if n else 0
    winrate = wins / n * 100 if n else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if n else 0.0
    gross_loss = -float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if n else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    pnl = float(trades_df["pnl"].sum()) if n else 0.0

    if n:
        risk_notional = (
            (abs(trades_df["entry_px"] - trades_df["sl_px"]) * trades_df["size"] * spec.point_value)
            .replace(0, np.nan)
        )
        expectancy_R = float((trades_df["pnl"] / risk_notional).mean())
        ret = trades_df["pnl"] / cfg.equity0
        sharpe_proxy = float(ret.mean() / (ret.std(ddof=1) + 1e-12))
    else:
        expectancy_R = np.nan
        sharpe_proxy = np.nan

    metrics = {
        "trades": n,
        "winrate_pct": round(winrate, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "pnl": round(pnl, 2),
        "equity_end": round(float(equity), 2),
        "expectancy_R": round(expectancy_R, 3) if np.isfinite(expectancy_R) else None,
        "sharpe_proxy": round(sharpe_proxy, 3) if np.isfinite(sharpe_proxy) else None,
    }

    return eq, trades_df, metrics
