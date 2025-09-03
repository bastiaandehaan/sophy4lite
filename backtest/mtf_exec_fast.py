# backtest/mtf_exec_fast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from strategies.mtf_confluence import MTFSignals, MTFParams
from utils.specs import get_spec, SymbolSpec


@dataclass(frozen=True)
class MTFExecCfg:
    equity0: float = 20_000.0
    fee_rate: float = 0.0002      # 2 bps per leg
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    risk_frac: float = 0.01       # 1% per trade
    atr_floor: float = 1e-6
    mark_to_market: bool = False  # value open positions at end


def backtest_mtf_confluence_fast(
    df: pd.DataFrame,
    symbol: str,
    params: MTFParams = MTFParams(),
    cfg: MTFExecCfg = MTFExecCfg(),
    session_start: str = "09:00",
    session_end: str = "17:00",
    max_trades_per_day: int = 1,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """FAST vectorized backtest of MTF Confluence strategy."""

    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not req.issubset(df.columns):
        miss = req.difference(df.columns)
        raise KeyError(f"DataFrame missing columns: {sorted(miss)}")

    df = df.sort_index().copy()
    spec = get_spec(symbol)

    # Generate signals
    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_start, session_end)
    if max_trades_per_day == 1:
        signals = mtf.filter_best_daily_signal(signals)

    # Merge data
    data = pd.concat([df[["open", "high", "low", "close"]], signals], axis=1)

    # Pre-calc entry prices with slippage
    data["long_entry_px"] = data["close"] + cfg.entry_slip_pts
    data["short_entry_px"] = data["close"] - cfg.entry_slip_pts

    # Risk pts
    data["long_risk_pts"] = np.abs(data["long_entry_px"] - data["long_sl"])
    data["short_risk_pts"] = np.abs(data["short_entry_px"] - data["short_sl"])

    # Collect trades (simple vectorized pass)
    trades = []
    equity = cfg.equity0
    eq_curve = []

    # Build entry table
    long_entries = data[data["long_entry"]].copy()
    short_entries = data[data["short_entry"]].copy()
    all_entries = pd.concat(
        [long_entries.assign(side="long"), short_entries.assign(side="short")]
    ).sort_index()

    for entry_time, row in all_entries.iterrows():
        if equity <= 0:
            break

        side = row["side"]
        if side == "long":
            entry_px = row["long_entry_px"]
            sl_px = row["long_sl"]
            tp_px = row["long_tp"]
            risk_pts = row["long_risk_pts"]
            score = row["long_score"]
            ev = row["long_ev"]
        else:
            entry_px = row["short_entry_px"]
            sl_px = row["short_sl"]
            tp_px = row["short_tp"]
            risk_pts = row["short_risk_pts"]
            score = row["short_score"]
            ev = row["short_ev"]

        if risk_pts <= 0 or not np.isfinite(risk_pts):
            continue

        size = (equity * cfg.risk_frac) / (risk_pts * spec.point_value)
        if spec.lot_step > 0:
            size = np.floor(size / spec.lot_step) * spec.lot_step
        if size <= 0:
            continue

        # Future bars from next minute (no same-bar exits)
        future = df.loc[df.index > entry_time]

        # Exit detection
        if side == "long":
            sl_hit = future[future["low"] <= sl_px]
            tp_hit = future[future["high"] >= tp_px]
        else:
            sl_hit = future[future["high"] >= sl_px]
            tp_hit = future[future["low"] <= tp_px]

        exit_time = None
        exit_reason = None
        exit_px = None

        if len(sl_hit) > 0 and len(tp_hit) > 0:
            if sl_hit.index[0] <= tp_hit.index[0]:
                exit_time = sl_hit.index[0]
                exit_reason = "SL"
                exit_px = sl_px - cfg.sl_slip_pts if side == "long" else sl_px + cfg.sl_slip_pts
            else:
                exit_time = tp_hit.index[0]
                exit_reason = "TP"
                exit_px = tp_px + cfg.tp_slip_pts if side == "long" else tp_px - cfg.tp_slip_pts
        elif len(sl_hit) > 0:
            exit_time = sl_hit.index[0]
            exit_reason = "SL"
            exit_px = sl_px - cfg.sl_slip_pts if side == "long" else sl_px + cfg.sl_slip_pts
        elif len(tp_hit) > 0:
            exit_time = tp_hit.index[0]
            exit_reason = "TP"
            exit_px = tp_px + cfg.tp_slip_pts if side == "long" else tp_px - cfg.tp_slip_pts

        # Optional mark-to-market close at end
        if exit_time is None and cfg.mark_to_market and len(future) > 0:
            exit_time = future.index[-1]
            exit_reason = "M2M"
            exit_px = future.iloc[-1]["close"]

        if exit_time is None:
            continue

        # P&L with fees
        if side == "long":
            gross = (exit_px - entry_px) * size * spec.point_value
        else:
            gross = (entry_px - exit_px) * size * spec.point_value
        entry_fee = abs(entry_px * size * spec.point_value) * cfg.fee_rate
        exit_fee = abs(exit_px * size * spec.point_value) * cfg.fee_rate
        net = gross - entry_fee - exit_fee

        equity += net
        eq_curve.append((exit_time, equity))

        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": entry_time,
                "entry_px": float(entry_px),
                "exit_time": exit_time,
                "exit_px": float(exit_px),
                "reason": exit_reason,
                "size": float(size),
                "sl_px": float(sl_px),
                "tp_px": float(tp_px),
                "sl_pts": float(abs(entry_px - sl_px)),
                "tp_pts": float(abs(tp_px - entry_px)),
                "pnl": float(net),
                "gross": float(gross),
                "fees": float(entry_fee + exit_fee),
                "score": float(score),
                "ev_R": float(ev),
            }
        )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Equity curve (mark trade exits)
    eq = pd.Series(index=df.index, data=cfg.equity0, dtype=float)
    for _, tr in trades_df.iterrows():
        eq.loc[tr["exit_time"]:] = eq.loc[tr["exit_time"]:] + tr["pnl"]

    # ---- Metrics (extended) ----
    n = len(trades_df)
    if n > 0:
        gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = -trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()
        pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

        # R-multiples
        risk_notional = trades_df["sl_pts"] * trades_df["size"] * spec.point_value
        r_multiples = trades_df["pnl"] / risk_notional.replace(0, np.nan)
        zero_r_trades = r_multiples.isna().sum()
        valid_r = r_multiples.dropna()
        actual_expectancy = valid_r.mean() if len(valid_r) > 0 else 0.0

        # Sharpe (approx)
        returns = trades_df["pnl"] / cfg.equity0
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

        # Max DD
        cum = (1 + returns).cumprod()
        runmax = cum.expanding().max()
        dd = (cum - runmax) / runmax
        max_dd = abs(dd.min()) * 100 if len(dd) > 0 else 0.0

        winrate = (trades_df["pnl"] > 0).mean() * 100.0
        expectancy_r = r_multiples.mean()
    else:
        pf = np.nan
        actual_expectancy = 0.0
        zero_r_trades = 0
        sharpe = 0.0
        max_dd = 0.0
        winrate = 0.0
        expectancy_r = 0.0

    metrics = {
        "trades": n,
        "winrate_pct": round(winrate, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "pnl": round(trades_df["pnl"].sum(), 2) if n > 0 else 0.0,
        "equity_end": round(float(eq.iloc[-1]), 2),
        "return_pct": round((eq.iloc[-1] / cfg.equity0 - 1) * 100, 2),
        "expectancy_R": round(expectancy_r, 3) if np.isfinite(expectancy_r) else 0.0,
        "actual_expectancy_R": round(actual_expectancy, 3),
        "trades_with_zero_r": int(zero_r_trades),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
    }

    return eq, trades_df, metrics
