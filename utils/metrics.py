from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_sharpe(df_eq: pd.DataFrame) -> float:
    ret = df_eq["equity"].pct_change().dropna()
    if ret.std(ddof=0) == 0 or len(ret) < 2:
        return 0.0
    return float(ret.mean() / ret.std(ddof=0) * np.sqrt(TRADING_DAYS))


def equity_max_dd_and_duration(df_eq: pd.DataFrame) -> tuple[float, int]:
    eq = df_eq["equity"].to_numpy()
    roll_max = np.maximum.accumulate(eq)
    dd = eq / roll_max - 1.0
    max_dd = dd.min() if len(dd) else 0.0
    # duration in bars of the longest drawdown spell
    dur = 0
    max_dur = 0
    for x in dd:
        if x < 0:
            dur += 1
            if dur > max_dur:
                max_dur = dur
        else:
            dur = 0
    return abs(float(max_dd)), int(max_dur)


def summarize_equity_metrics(df_eq: pd.DataFrame, trades: pd.DataFrame) -> dict:
    sharpe = equity_sharpe(df_eq)
    max_dd, dd_dur = equity_max_dd_and_duration(df_eq)
    total_ret = float(df_eq["equity"].iloc[-1] / df_eq["equity"].iloc[0] - 1.0)
    n_trades = int(len(trades)) if trades is not None else 0
    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "dd_duration": dd_dur,
        "total_return": total_ret,
        "n_trades": n_trades,
    }