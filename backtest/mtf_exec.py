# backtest/mtf_exec.py
"""
Backtest executor voor MTF confluence strategy
FIXED: Complete rewrite met proper imports en error handling
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from strategies.mtf_confluence import MTFSignals, MTFParams
from strategies.breakout_params import SymbolSpec, DEFAULT_SPECS
from utils.position import _size
from utils.metrics import summarize_equity_metrics


@dataclass(frozen=True)
class MTFExecCfg:
    """Config voor MTF backtest execution"""
    equity0: float = 20_000.0
    fee_rate: float = 0.0002  # 2 bps per leg
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    specs: Optional[Dict[str, SymbolSpec]] = None
    risk_frac: float = 0.01  # 1% risk per trade (FTMO safe)

    def get_spec(self, symbol: str) -> SymbolSpec:
        """Get symbol specification with fallback"""
        specs = self.specs or DEFAULT_SPECS
        if symbol in specs:
            return specs[symbol]

        # Try base symbol (e.g., "GER40" from "GER40.cash")
        base = symbol.split(".")[0]
        if base in specs:
            return specs[base]

        # Default spec
        return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)


def backtest_mtf_confluence(df: pd.DataFrame, symbol: str,
        params: MTFParams = MTFParams(), cfg: MTFExecCfg = MTFExecCfg(),
        session_start: str = "09:00", session_end: str = "17:00",
        max_trades_per_day: int = 1, verbose: bool = False) -> Tuple[
    pd.Series, pd.DataFrame, dict]:
    """
    Backtest MTF confluence strategy

    Returns:
        - equity: Series met equity curve
        - trades: DataFrame met trade details
        - metrics: Dict met performance metrics
    """
    # Validate input
    req = {"open", "high", "low", "close"}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not req.issubset(df.columns):
        miss = req.difference(df.columns)
        raise KeyError(f"DataFrame missing columns: {sorted(miss)}")

    # Prepare data
    df = df.sort_index().copy()
    spec = cfg.get_spec(symbol)

    # Generate MTF signals
    if verbose:
        print(f"Generating MTF signals for {symbol}...")

    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_start, session_end)

    if max_trades_per_day == 1:
        signals = mtf.filter_best_daily_signal(signals)

    # Count potential signals
    n_long_signals = signals['long_entry'].sum()
    n_short_signals = signals['short_entry'].sum()

    if verbose:
        print(f"Found {n_long_signals} long and {n_short_signals} short signals")

    # Initialize tracking
    eq = pd.Series(index=df.index, dtype="float64")
    equity = float(cfg.equity0)
    trades = []

    # Position state
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
            signal_side = None

            if signals.loc[ts, 'long_entry']:
                signal_side = "long"
                effective_entry = c + cfg.entry_slip_pts
                sl_raw = signals.loc[ts, 'long_sl']
                tp_raw = signals.loc[ts, 'long_tp']
                entry_score = signals.loc[ts, 'long_score']
                entry_ev = signals.loc[ts, 'long_ev']

            elif signals.loc[ts, 'short_entry']:
                signal_side = "short"
                effective_entry = c - cfg.entry_slip_pts
                sl_raw = signals.loc[ts, 'short_sl']
                tp_raw = signals.loc[ts, 'short_tp']
                entry_score = signals.loc[ts, 'short_score']
                entry_ev = signals.loc[ts, 'short_ev']

            if signal_side is None:
                eq.loc[ts] = equity
                continue

            # Validate SL/TP levels
            if signal_side == "long":
                if sl_raw >= effective_entry:
                    eq.loc[ts] = equity
                    continue
                if tp_raw <= effective_entry:
                    eq.loc[ts] = equity
                    continue
            else:  # short
                if sl_raw <= effective_entry:
                    eq.loc[ts] = equity
                    continue
                if tp_raw >= effective_entry:
                    eq.loc[ts] = equity
                    continue

            # Calculate position size
            this_size = _size(equity, effective_entry, sl_raw, spec.point_value,
                              cfg.risk_frac)

            # Skip if size too small
            if this_size < spec.min_step:
                eq.loc[ts] = equity
                continue

            # Round to min_step
            this_size = round(this_size / spec.min_step) * spec.min_step

            # Enter position
            side = signal_side
            entry_px = effective_entry
            sl_px = sl_raw
            tp_px = tp_raw
            size = this_size
            entry_time = ts
            in_pos = True

            # Calculate entry fees
            notional = abs(entry_px * size * spec.point_value)
            entry_leg_fee = notional * cfg.fee_rate

            eq.loc[ts] = equity
            continue

        # In position - check exits
        exit_reason = None
        exit_px = None

        if side == "long":
            # Check stop loss
            if l <= sl_px:
                exit_reason = "SL"
                exit_px = max(l, sl_px - cfg.sl_slip_pts)  # Realistic fill
            # Check take profit
            elif h >= tp_px:
                exit_reason = "TP"
                exit_px = min(h, tp_px + cfg.tp_slip_pts)  # Realistic fill

        else:  # short
            # Check stop loss
            if h >= sl_px:
                exit_reason = "SL"
                exit_px = min(h, sl_px + cfg.sl_slip_pts)  # Realistic fill
            # Check take profit
            elif l <= tp_px:
                exit_reason = "TP"
                exit_px = max(l, tp_px - cfg.tp_slip_pts)  # Realistic fill

        if exit_reason is None:
            eq.loc[ts] = equity
            continue

        # Calculate P&L
        notional_exit = abs(exit_px * size * spec.point_value)

        if side == "long":
            pnl_per_contract = (exit_px - entry_px) * spec.point_value
        else:
            pnl_per_contract = (entry_px - exit_px) * spec.point_value

        gross = pnl_per_contract * size
        exit_fee = notional_exit * cfg.fee_rate
        net = gross - entry_leg_fee - exit_fee

        # Update equity
        equity += net

        # Update win rate history for adaptive EV
        is_win = net > 0
        mtf.win_rate_history.append(float(is_win))
        if len(mtf.win_rate_history) > 100:  # Keep last 100 trades
            mtf.win_rate_history.pop(0)

        # Record trade
        trades.append({"symbol": symbol, "side": side, "entry_time": entry_time,
            "entry_px": entry_px, "exit_time": ts, "exit_px": exit_px,
            "reason": exit_reason, "size": size, "sl_px": sl_px, "tp_px": tp_px,
            "sl_pts": abs(entry_px - sl_px), "tp_pts": abs(tp_px - entry_px),
            "pnl": net, "pnl_cash": net,  # For metrics compatibility
            "gross": gross, "fees": entry_leg_fee + exit_fee, "score": entry_score,
            "ev": entry_ev, })

        # Reset position
        in_pos = False
        side = None
        entry_px = sl_px = tp_px = np.nan
        size = 0.0
        entry_time = None
        entry_score = 0.0
        entry_ev = 0.0
        entry_leg_fee = 0.0

        eq.loc[ts] = equity

    # Fill forward equity
    eq = eq.ffill().fillna(cfg.equity0).rename("equity")

    # Create trades DataFrame
    if trades:
        trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(
            drop=True)
    else:
        # Empty trades DataFrame with correct columns
        trades_df = pd.DataFrame(
            columns=["symbol", "side", "entry_time", "entry_px", "exit_time", "exit_px",
                "reason", "size", "sl_px", "tp_px", "sl_pts", "tp_pts", "pnl",
                "pnl_cash", "gross", "fees", "score", "ev"])

    # Calculate metrics
    metrics = summarize_equity_metrics(eq.to_frame("equity"), trades_df)

    # Add strategy-specific metrics
    if len(trades_df) > 0:
        metrics["avg_score"] = float(trades_df["score"].mean())
        metrics["avg_ev"] = float(trades_df["ev"].mean())
        metrics["expectancy"] = float(trades_df["pnl"].mean())
        metrics["expectancy_r"] = float(
            trades_df["pnl"].mean() / (cfg.equity0 * cfg.risk_frac))

        # Win/loss analysis
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        if len(wins) > 0:
            metrics["avg_win"] = float(wins["pnl"].mean())
            metrics["max_win"] = float(wins["pnl"].max())
        else:
            metrics["avg_win"] = 0.0
            metrics["max_win"] = 0.0

        if len(losses) > 0:
            metrics["avg_loss"] = float(losses["pnl"].mean())
            metrics["max_loss"] = float(losses["pnl"].min())
        else:
            metrics["avg_loss"] = 0.0
            metrics["max_loss"] = 0.0

        # Profit factor
        total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
        total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
        metrics["profit_factor"] = float(
            total_wins / total_losses) if total_losses > 0 else 0.0

    if verbose:
        print(f"Backtest complete: {len(trades_df)} trades, final equity: {equity:.2f}")

    return eq, trades_df, metrics