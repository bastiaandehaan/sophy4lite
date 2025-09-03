# backtest/mtf_exec_fast.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd

from strategies.mtf_confluence import MTFSignals, MTFParams
from utils.specs import get_spec, SymbolSpec


log = logging.getLogger(__name__)
if not log.handlers:
    # Light default handler; project-level logging can override this.
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)


class EnterOn(str, Enum):
    CLOSE = "close"
    NEXT_OPEN = "next_open"


@dataclass(frozen=True)
class MTFExecCfg:
    """
    Execution configuration for the MTF Confluence backtest engine.

    Notes:
    - risk_frac is the fraction of current equity risked per trade.
    - fee_rate is applied on notional per leg (entry + exit).
    - min_risk_pts floors the risk distance to avoid infinite position sizes.
    - enter_on governs entry timing: 'next_open' avoids look-ahead.
    - allow_concurrent=False prevents overlapping trades (safer accounting).
    """
    equity0: float = 20_000.0
    fee_rate: float = 0.0002          # 2 bps per leg on notional
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    risk_frac: float = 0.01           # 1% risk per trade
    min_risk_pts: float = 1e-3        # floors risk distance
    mark_to_market: bool = False      # value open positions at the end
    enter_on: EnterOn = EnterOn.NEXT_OPEN
    allow_concurrent: bool = False
    prefer_metric: str = "ev"         # "ev" | "score"


# ---- Internal helpers -------------------------------------------------------


_REQUIRED_OHLC = ("open", "high", "low", "close")
# Entries and protective levels expected from strategy
_REQUIRED_SIGNAL_COLS = (
    "long_entry", "short_entry",
    "long_sl", "long_tp", "short_sl", "short_tp",
)
# Optional quality metrics
_OPTIONAL_SIGNAL_COLS = ("long_ev", "short_ev", "long_score", "short_score")


def _validate_inputs(df: pd.DataFrame, signals: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex")

    missing = set(_REQUIRED_OHLC).difference(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing OHLC columns: {sorted(missing)}")

    missing_sig = set(_REQUIRED_SIGNAL_COLS).difference(signals.columns)
    if missing_sig:
        raise KeyError(f"Signals missing required columns: {sorted(missing_sig)}")


def _date_key(idx: pd.DatetimeIndex) -> np.ndarray:
    """Return date (YYYY-MM-DD) as np.array for grouping, TZ-safe."""
    if idx.tz is not None:
        # Keep local calendar date; do not convert to UTC to avoid day shift.
        return idx.tz_localize(None).date
    return idx.date


def _choose_metric_frame(data: pd.DataFrame, prefer: str) -> pd.Series:
    """
    Produce a single 'choose' metric column per row, using preferred signal metric.
    prefer: "ev" or "score".
    """
    def _choose_row(side: str, row: pd.Series) -> float:
        if side == "long":
            ev = row.get("long_ev")
            sc = row.get("long_score")
        else:
            ev = row.get("short_ev")
            sc = row.get("short_score")
        if prefer == "ev" and pd.notna(ev):
            return float(ev)
        if pd.notna(sc):
            return float(sc)
        # Fallback if nothing present
        return -np.inf

    return data.apply(lambda r: _choose_row(r["side"], r), axis=1)


# ---- Core engine ------------------------------------------------------------


def backtest_mtf_confluence_fast(
    df: pd.DataFrame,
    symbol: str,
    params: MTFParams = MTFParams(),
    cfg: MTFExecCfg = MTFExecCfg(),
    session_start: str = "09:00",
    session_end: str = "17:00",
    max_trades_per_day: Optional[int] = 1,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """
    FAST backtest for MTF Confluence with robust execution rules.

    Key properties:
    - Default entry at NEXT bar's OPEN (no look-ahead).
    - Single side per timestamp (choose best EV/score).
    - Enforces per-day trade limit for any integer max_trades_per_day >= 1.
    - Optional no-overlap mode (allow_concurrent=False).
    """

    # ---- Validate input & build signals
    df = df.sort_index()
    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_start, session_end)
    _validate_inputs(df, signals)

    spec: SymbolSpec = get_spec(symbol)

    # Merge base OHLC with signals of interest only
    cols_needed = list(_REQUIRED_OHLC) + list(_REQUIRED_SIGNAL_COLS) + [
        c for c in _OPTIONAL_SIGNAL_COLS if c in signals.columns
    ]
    data = pd.concat([df[list(_REQUIRED_OHLC)], signals[cols_needed[len(_REQUIRED_OHLC):]]], axis=1)

    # Build candidate entries
    long_entries = data.loc[data["long_entry"]].copy()
    short_entries = data.loc[data["short_entry"]].copy()
    if long_entries.empty and short_entries.empty:
        log.warning("No entry signals found after session filtering.")
        eq = pd.Series(cfg.equity0, index=df.index, dtype=float)
        return eq, pd.DataFrame(), _empty_metrics(cfg.equity0)

    long_entries["side"] = "long"
    short_entries["side"] = "short"
    candidates = pd.concat([long_entries, short_entries], axis=0).sort_index()

    # Per-timestamp side exclusivity (pick best by EV or score)
    # Compose a 'choose' metric and pick best per timestamp
    candidates["choose"] = _choose_metric_frame(candidates, cfg.prefer_metric)
    per_ts_best = (
        candidates.groupby(level=0, group_keys=False)
        .apply(lambda g: g.sort_values("choose", ascending=False).head(1))
    )

    # Per-day trade limit (pick best N by 'choose' inside each day)
    if max_trades_per_day is not None and max_trades_per_day >= 1:
        day_key = _date_key(per_ts_best.index)
        per_ts_best = (
            per_ts_best.assign(_day=day_key)
            .groupby("_day", group_keys=False)
            .apply(lambda g: g.sort_values(["choose", g.index.name], ascending=[False, True]).head(max_trades_per_day))
            .drop(columns="_day")
            .sort_index()
        )
    else:
        per_ts_best = per_ts_best.sort_index()

    # ---- Prepare arrays for fast index mapping
    times = df.index
    n_bars = len(times)
    indexer = pd.Index(times)

    # Execution bookkeeping
    trades: List[Dict[str, Any]] = []
    equity: float = cfg.equity0
    last_exit_pos: int = -1  # index position of last exit (enforce no overlap)
    skip_counts = {"bad_risk": 0, "zero_size": 0, "no_exit": 0, "late_signal": 0, "overlap": 0}

    for entry_time, row in per_ts_best.iterrows():
        # Determine actual execution bar (enter at next open or at current close)
        entry_pos = indexer.get_indexer([entry_time])[0]
        if entry_pos == -1:
            # Shouldn't happen, but guard anyway
            skip_counts["late_signal"] += 1
            continue

        if cfg.enter_on == EnterOn.NEXT_OPEN:
            exec_pos = entry_pos + 1
            if exec_pos >= n_bars:
                skip_counts["late_signal"] += 1
                continue
            exec_time = times[exec_pos]
            entry_px_base = df["open"].iloc[exec_pos]
        else:
            # Enter on same-bar close (can introduce look-ahead if signals use close!)
            exec_pos = entry_pos
            exec_time = entry_time
            entry_px_base = df["close"].iloc[exec_pos]

        # Enforce no overlapping trades (if disabled concurrency)
        if not cfg.allow_concurrent and last_exit_pos >= 0 and exec_pos <= last_exit_pos:
            skip_counts["overlap"] += 1
            continue

        side = str(row["side"])
        if side == "long":
            entry_px = float(entry_px_base + cfg.entry_slip_pts)
            sl_px = float(row["long_sl"])
            tp_px = float(row["long_tp"])
            score = float(row["long_score"]) if "long_score" in row and pd.notna(row["long_score"]) else np.nan
            ev = float(row["long_ev"]) if "long_ev" in row and pd.notna(row["long_ev"]) else np.nan
        else:
            entry_px = float(entry_px_base - cfg.entry_slip_pts)
            sl_px = float(row["short_sl"])
            tp_px = float(row["short_tp"])
            score = float(row["short_score"]) if "short_score" in row and pd.notna(row["short_score"]) else np.nan
            ev = float(row["short_ev"]) if "short_ev" in row and pd.notna(row["short_ev"]) else np.nan

        # Validate SL/TP
        if not np.isfinite(sl_px) or not np.isfinite(tp_px):
            skip_counts["bad_risk"] += 1
            continue

        risk_pts = abs(entry_px - sl_px)
        if not np.isfinite(risk_pts) or risk_pts <= 0:
            skip_counts["bad_risk"] += 1
            continue

        risk_pts = max(risk_pts, cfg.min_risk_pts)

        # Position sizing
        size = (equity * cfg.risk_frac) / (risk_pts * spec.point_value)
        if spec.lot_step > 0:
            size = np.floor(size / spec.lot_step) * spec.lot_step
        if size <= 0 or not np.isfinite(size):
            skip_counts["zero_size"] += 1
            continue

        # Exit search (from execution bar forward; include same bar)
        fut = df.iloc[exec_pos + 0 :]  # include exec bar
        if side == "long":
            sl_hit = fut.index[fut["low"] <= sl_px]
            tp_hit = fut.index[fut["high"] >= tp_px]
        else:
            sl_hit = fut.index[fut["high"] >= sl_px]
            tp_hit = fut.index[fut["low"] <= tp_px]

        exit_time: Optional[pd.Timestamp] = None
        exit_reason: Optional[str] = None
        exit_px: Optional[float] = None

        # Conservative tie-breaking: SL before TP if same timestamp
        if len(sl_hit) > 0 and len(tp_hit) > 0:
            if sl_hit[0] <= tp_hit[0]:
                exit_time = sl_hit[0]
                exit_reason = "SL"
                exit_px = float(sl_px - cfg.sl_slip_pts if side == "long" else sl_px + cfg.sl_slip_pts)
            else:
                exit_time = tp_hit[0]
                exit_reason = "TP"
                exit_px = float(tp_px + cfg.tp_slip_pts if side == "long" else tp_px - cfg.tp_slip_pts)
        elif len(sl_hit) > 0:
            exit_time = sl_hit[0]
            exit_reason = "SL"
            exit_px = float(sl_px - cfg.sl_slip_pts if side == "long" else sl_px + cfg.sl_slip_pts)
        elif len(tp_hit) > 0:
            exit_time = tp_hit[0]
            exit_reason = "TP"
            exit_px = float(tp_px + cfg.tp_slip_pts if side == "long" else tp_px - cfg.tp_slip_pts)

        # Optional mark-to-market
        if exit_time is None and cfg.mark_to_market and len(fut) > 0:
            exit_time = fut.index[-1]
            exit_reason = "M2M"
            exit_px = float(fut["close"].iloc[-1])

        if exit_time is None or exit_px is None:
            skip_counts["no_exit"] += 1
            continue

        exit_pos = indexer.get_indexer([exit_time])[0]
        if exit_pos == -1:
            # Should not happen; guard for robustness
            skip_counts["no_exit"] += 1
            continue

        # PnL & fees
        if side == "long":
            gross = (exit_px - entry_px) * size * spec.point_value
        else:
            gross = (entry_px - exit_px) * size * spec.point_value

        entry_fee = abs(entry_px * size * spec.point_value) * cfg.fee_rate
        exit_fee = abs(exit_px * size * spec.point_value) * cfg.fee_rate
        net = float(gross - entry_fee - exit_fee)

        equity = float(equity + net)
        last_exit_pos = max(last_exit_pos, exit_pos)

        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": exec_time,         # actual execution time
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
                "score": float(score) if np.isfinite(score) else None,
                "ev_R": float(ev) if np.isfinite(ev) else None,
            }
        )

    if any(skip_counts.values()):
        log.info("Entry skips: %s", skip_counts)

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values("exit_time") if not trades_df.empty else trades_df

    # Equity curve — cumulative PnL at exits (fast + correct)
    eq = pd.Series(cfg.equity0, index=df.index, dtype=float)
    if not trades_df.empty:
        pnl_at_exit = trades_df.groupby("exit_time")["pnl"].sum()
        pnl_series = pnl_at_exit.reindex(df.index, fill_value=0.0)
        eq = cfg.equity0 + pnl_series.cumsum()

    metrics = _compute_metrics(trades_df, eq, cfg.equity0, spec)
    return eq, trades_df, metrics


# ---- Metrics ----------------------------------------------------------------


def _empty_metrics(equity0: float) -> dict:
    return {
        "trades": 0,
        "winrate_pct": 0.0,
        "profit_factor": None,
        "pnl": 0.0,
        "equity_end": float(equity0),
        "return_pct": 0.0,
        "expectancy_R": 0.0,
        "actual_expectancy_R": 0.0,
        "trades_with_zero_r": 0,
        "sharpe": 0.0,
        "max_dd_pct": 0.0,
    }


def _compute_metrics(trades_df: pd.DataFrame, eq: pd.Series, equity0: float, spec: SymbolSpec) -> dict:
    if trades_df.empty:
        return _empty_metrics(equity0)

    n = len(trades_df)
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    # R-multiples
    risk_notional = trades_df["sl_pts"] * trades_df["size"] * spec.point_value
    r_mult = (trades_df["pnl"] / risk_notional.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    zero_r_trades = int(r_mult.isna().sum())
    valid_r = r_mult.dropna()
    expectancy_r = float(valid_r.mean()) if len(valid_r) > 0 else 0.0

    # Sharpe using per-trade returns vs initial equity (approx)
    trade_ret = trades_df["pnl"] / equity0
    sharpe = float((trade_ret.mean() / trade_ret.std() * np.sqrt(252))) if trade_ret.std() > 0 else 0.0

    # Max drawdown from equity curve
    cum = eq / float(equity0)
    runmax = cum.cummax()
    dd = (cum - runmax) / runmax
    max_dd = float(abs(dd.min()) * 100)

    winrate = float((trades_df["pnl"] > 0).mean() * 100.0)

    return {
        "trades": int(n),
        "winrate_pct": round(winrate, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "pnl": round(float(trades_df["pnl"].sum()), 2),
        "equity_end": round(float(eq.iloc[-1]), 2),
        "return_pct": round((float(eq.iloc[-1]) / float(equity0) - 1) * 100.0, 2),
        "expectancy_R": round(float(r_mult.mean()), 3) if np.isfinite(r_mult.mean()) else 0.0,
        "actual_expectancy_R": round(expectancy_r, 3),
        "trades_with_zero_r": zero_r_trades,
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
    }
