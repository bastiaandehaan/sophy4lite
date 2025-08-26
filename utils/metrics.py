from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _equity_series(df_eq: pd.DataFrame) -> pd.Series:
    if "equity" not in df_eq.columns:
        raise KeyError("Column 'equity' ontbreekt in equity DataFrame")
    s = pd.Series(df_eq["equity"], dtype="float64").dropna()
    if s.empty:
        raise ValueError("Lege equity-reeks ontvangen")
    return s


def equity_sharpe(
    df_eq: pd.DataFrame,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    s = _equity_series(df_eq)
    rets = s.pct_change().dropna()
    if rets.size < 2:
        return 0.0
    ex = rets - (risk_free / periods_per_year)
    std = ex.std(ddof=1)
    if std <= 1e-12:
        return float("inf") if ex.mean() > 0 else 0.0
    return float(ex.mean() / std * np.sqrt(periods_per_year))


def equity_max_dd_and_duration(df_eq: pd.DataFrame) -> tuple[float, int]:
    s = _equity_series(df_eq)
    v = s.values.astype("float64")
    run_max = np.maximum.accumulate(v)
    dd = v / run_max - 1.0
    max_dd_signed = float(dd.min())

    duration = 0
    max_duration = 0
    peak = v[0]
    for x in v:
        if x >= peak:
            peak = x
            duration = 0
        else:
            duration += 1
            max_duration = max(max_duration, duration)

    return abs(max_dd_signed), int(max_duration)


def summarize_equity_metrics(df_eq: pd.DataFrame, trades: pd.DataFrame | None) -> dict:
    s = _equity_series(df_eq)
    sharpe = equity_sharpe(df_eq)
    max_dd, dd_dur = equity_max_dd_and_duration(df_eq)
    total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    n_trades = int(len(trades)) if trades is not None else 0
    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "dd_duration": dd_dur,
        "total_return": total_ret,
        "n_trades": n_trades,
    }
