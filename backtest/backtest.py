# backtest/backtest.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import logger, INITIAL_CAPITAL, FEES

def run_backtest(
    df: pd.DataFrame,
    entries: pd.Series,
    sl_stop: pd.Series,
    tp_stop: pd.Series,
    freq: str = "1H",
    slippage: float = 0.0002,  # = 2 basispunten (0.02%)
) -> vbt.Portfolio:

    if df.empty:
        raise ValueError("Empty dataframe")
    for k in ["open", "high", "low", "close"]:
        if k not in df.columns:
            raise ValueError("df must contain OHLC columns")

    entries = entries.astype(bool).reindex(df.index).fillna(False)
    sl_stop = sl_stop.reindex(df.index).fillna(0.0)
    tp_stop = tp_stop.reindex(df.index).fillna(0.0)

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        init_cash=INITIAL_CAPITAL,
        fees=FEES,
        slippage=slippage,
        freq=freq,
        call_seq="auto",
    )
    return pf

def metrics(pf: vbt.Portfolio) -> Dict[str, Any]:
    eq = pf.value()
    # Drawdown (negatief getal), rapporteren als positief percentage
    cummax = eq.cummax()
    dd_series = eq / cummax - 1.0
    max_dd = float(dd_series.min()) if len(dd_series) else 0.0

    # Sharpe op dagbasis
    # NB: pf.returns() kan timeframe-afhankelijk zijn; we nemen equity en resamplen
    daily_ret = eq.resample("1D").last().pct_change(fill_method=None).dropna()

    if len(daily_ret) > 2:
        ann_factor = np.sqrt(252)
        sharpe = float(daily_ret.mean() / (daily_ret.std() + 1e-12) * ann_factor)
    else:
        sharpe = 0.0

    trades = len(pf.trades.records) if hasattr(pf.trades, "records") else 0
    win_rate = float((pf.trades.pnl > 0).sum() / trades) if trades else 0.0

    return {
        "total_return": float(pf.total_return() or 0.0),
        "sharpe_ratio": sharpe,
        "max_drawdown": abs(max_dd),  # als positief percentage rapporteren
        "trades_count": trades,
        "win_rate": win_rate,
        "final_value": float(eq.iloc[-1]) if len(eq) else 0.0,
    }
