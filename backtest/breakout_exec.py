from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from strategies.breakout_signals import BreakoutParams, SymbolSpec, DEFAULT_SPECS, daily_levels, confirm_pass
from risk.ftmo_guard import FtmoGuard, FtmoRules
from utils.metrics import summarize_equity_metrics  # Gebruik gecentraliseerde metrics


@dataclass
class BTExecCfg:
    equity0: float = 20_000.0
    fee_rate: float = 0.0002          # per side
    slippage_pts: float = 0.5
    specs: Dict[str, SymbolSpec] = field(default_factory=lambda: DEFAULT_SPECS.copy())
    risk_frac: float = 0.01           # 1% per trade


def _tz_df(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    return (df.tz_localize(tz) if df.index.tz is None else df.tz_convert(tz)).sort_index()

def _size(equity: float, entry: float, stop: float, vpp: float, risk_frac: float) -> float:
    risk_cash = equity * risk_frac
    pts = abs(entry - stop)
    if pts <= 0:
        return 0.0
    return float(risk_cash / (pts * vpp))

def _fees(notional: float, rate: float) -> float:
    return float(abs(notional) * rate)

def backtest_breakout(
    df: pd.DataFrame,
    symbol: str,
    p: BreakoutParams,
    x: BTExecCfg = BTExecCfg(),
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float]]:
    """
    Mode 'close_confirm':
      - wacht tot na sessie-einde
      - als close door level gaat, wacht N closes; entry op de N-de close (geen look-ahead)
    Mode 'pending_stop':
      - zet stops direct na sessie; entry op wick-touch
      - confirm/volume gelden als pre-plaatsingsfilter (op data t/m sessie-einde)
    Beide: FTMO pre-trade check (worst-case SL), fees, slippage, optionele spread.
    Één trade max per dag.
    """
    df = _tz_df(df, p.broker_tz)
    vpp = x.specs[symbol].value_per_point

    levels = daily_levels(df, symbol, p)

    equity = float(x.equity0)
    guard = FtmoGuard(equity, FtmoRules())
    peak = equity

    trades: List[Dict] = []
    daily_eq: Dict[pd.Timestamp, float] = {}

    for od in levels:
        d = od["day"]
        dts = pd.Timestamp(d, tz=df.index.tz)
        day_df = df.loc[df.index.date == d]
        if day_df.empty:
            continue

        guard.new_day(dts.date(), equity)

        w_end = od["w_end"]
        post = day_df.loc[day_df.index > w_end]
        if post.empty or not od["vol_ok"]:
            daily_eq[dts] = equity
            continue

        buy_lvl = float(od["buy_lvl"])
        sell_lvl = float(od["sell_lvl"])
        sl_pts  = float(od["sl_pts"])
        tp_pts  = float(od["tp_pts"])
        cancel_at = od["cancel_at"]

        filled = None
        entry_px = entry_t = None
        sl = tp = None
        size = 0.0

        # --- ENTRY ---
        if p.mode == "close_confirm":
            cross_buy = post.index[(post["close"] > buy_lvl)]
            cross_sell = post.index[(post["close"] < sell_lvl)]
            t_trigger = None; side = None
            if cross_buy.any() and cross_sell.any():
                t_trigger = min(cross_buy[0], cross_sell[0])
                side = "BUY" if t_trigger == cross_buy[0] else "SELL"
            elif cross_buy.any():
                t_trigger, side = cross_buy[0], "BUY"
            elif cross_sell.any():
                t_trigger, side = cross_sell[0], "SELL"

            if side is not None:
                after = post.loc[post.index > t_trigger]
                if confirm_pass(after, buy_lvl if side=="BUY" else sell_lvl, side, p.confirm_bars):
                    t_entry = after.index[p.confirm_bars-1] if p.confirm_bars>0 else after.index[0]
                    if t_entry < cancel_at and guard.allowed_now():
                        px = float(df.loc[t_entry, "close"])
                        if side == "BUY":
                            sl = px - sl_pts; tp = px + tp_pts
                        else:
                            sl = px + sl_pts; tp = px - tp_pts
                        worst_loss = (abs(px - sl) * vpp) * _size(equity, px, sl, vpp, x.risk_frac)
                        if guard.pretrade_ok(worst_loss):
                            entry_px = px + (p.spread_pts if side=="BUY" else -p.spread_pts)
                            size = _size(equity, entry_px, sl, vpp, x.risk_frac)
                            equity -= _fees(entry_px * size * vpp, x.fee_rate)
                            entry_t = t_entry
                            filled = side

        else:  # pending_stop
            for ts, row in post.iterrows():
                if ts >= cancel_at or not guard.allowed_now():
                    break
                hi, lo = float(row["high"]), float(row["low"])
                if hi >= buy_lvl:
                    px = buy_lvl + x.slippage_pts + p.spread_pts
                    sl = px - sl_pts; tp = px + tp_pts
                    worst_loss = (abs(px - sl) * vpp) * _size(equity, px, sl, vpp, x.risk_frac)
                    if guard.pretrade_ok(worst_loss):
                        size = _size(equity, px, sl, vpp, x.risk_frac)
                        equity -= _fees(px * size * vpp, x.fee_rate)
                        entry_px, entry_t, filled = px, ts, "BUY"
                        break
                if lo <= sell_lvl:
                    px = sell_lvl - x.slippage_pts - p.spread_pts
                    sl = px + sl_pts; tp = px - tp_pts
                    worst_loss = (abs(px - sl) * vpp) * _size(equity, px, sl, vpp, x.risk_frac)
                    if guard.pretrade_ok(worst_loss):
                        size = _size(equity, px, sl, vpp, x.risk_frac)
                        equity -= _fees(px * size * vpp, x.fee_rate)
                        entry_px, entry_t, filled = px, ts, "SELL"
                        break

        # --- MANAGE / EXIT ---
        if filled is not None:
            for ts, row in post.loc[post.index >= entry_t].iterrows():
                if ts >= cancel_at:
                    break
                hi, lo = float(row["high"]), float(row["low"])
                hit_sl = (lo <= sl) if filled == "BUY" else (hi >= sl)
                hit_tp = (hi >= tp) if filled == "BUY" else (lo <= tp)
                if hit_sl or hit_tp:
                    exit_px = float(sl if hit_sl else tp)
                    pnl = (exit_px - entry_px) * size * vpp if filled == "BUY" else (entry_px - exit_px) * size * vpp
                    equity += pnl
                    equity -= _fees(exit_px * size * vpp, x.fee_rate)
                    trades.append({
                        "day": str(d), "symbol": symbol, "side": filled,
                        "entry_time": entry_t, "entry_price": float(entry_px),
                        "exit_time": ts,   "exit_price": float(exit_px),
                        "sl": float(sl), "tp": float(tp),
                        "sl_pts": float(sl_pts), "tp_pts": float(tp_pts),
                        "size": float(size), "pnl_cash": float(pnl),
                    })
                    break

        if equity > peak:
            peak = equity
        guard.update_equity(equity)
        daily_eq[dts] = equity

    eq_df = pd.Series(daily_eq).sort_index().to_frame("equity")
    trades_df = pd.DataFrame(trades)
    metrics = summarize_equity_metrics(eq_df, trades_df)  # Gebruik gecentraliseerde metrics
    return eq_df["equity"], trades_df, metrics