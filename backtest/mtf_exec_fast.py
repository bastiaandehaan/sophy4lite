from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from strategies.mtf_confluence import MTFParams, MTFSignals
from utils.specs import get_spec


@dataclass(frozen=True)
class MTFExecCfg:
    equity0: float = 20_000.0
    fee_rate: float = 0.0002
    entry_slip_pts: float = 0.1
    sl_slip_pts: float = 0.5
    tp_slip_pts: float = 0.0
    risk_frac: float = 0.01
    atr_floor: float = 1e-6
    mark_to_market: bool = False


def _calc_drawdown_pct(eq: pd.Series) -> float:
    run_max = eq.cummax()
    dd = (eq - run_max) / run_max
    return float(abs(dd.min()) * 100.0)


def backtest_mtf_confluence_fast(
    df: pd.DataFrame,
    symbol: str,
    params: MTFParams,
    cfg: MTFExecCfg,
    *,
    session_start: str = "09:00",
    session_end: str = "17:00",
    max_trades_per_day: int = 1,
) -> Tuple[pd.Series, pd.DataFrame, dict]:
    """Vectorized backtest (vereenvoudigd skeleton, sluit aan op jouw bestaande interface)."""

    # ---- SPEC LOOKUP (strikt, geen fallback) ----
    spec = get_spec(symbol)
    if spec is None:
        raise ValueError(
            f"Unknown symbol '{symbol}'. Configure it in utils/specs.py (no silent fallbacks)."
        )

    # ---- Signals ----
    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_start, session_end)
    if max_trades_per_day == 1:
        signals = mtf.filter_best_daily_signal(signals)

    # -------------------------------------------------------
    # HIER hoort jouw bestaande vectorized entry/exit engine.
    # Onderstaande is enkel een veilige placeholder die de
    # interface bewaart. Vervang dit blok door je eigen logica.
    # -------------------------------------------------------
    equity = pd.Series(cfg.equity0, index=df.index, dtype=float)

    trades = pd.DataFrame(
        columns=["time", "side", "entry", "exit", "pnl", "sl_pts", "size", "fees", "reason"]
    ).set_index("time", drop=True)

    metrics = {
        "trades": int(len(trades)),
        "winrate_pct": 0.0,
        "profit_factor": 0.0,
        "pnl": 0.0,
        "equity_end": float(equity.iloc[-1]),
        "return_pct": round((equity.iloc[-1] / cfg.equity0 - 1) * 100.0, 2),
        "expectancy_R": 0.0,
        "actual_expectancy_R": 0.0,
        "trades_with_zero_r": 0,
        "sharpe": 0.0,
        "max_dd_pct": _calc_drawdown_pct(equity),
    }

    return equity, trades, metrics
