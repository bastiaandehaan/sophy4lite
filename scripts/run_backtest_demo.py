# scripts/run_backtest_demo.py
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from strategies.breakout_signals import breakout_long

from backtest.data_loader import fetch_data
from config import logger
from utils.data_health import health_line
from utils.days import summarize_day_health
from utils.plot import save_equity_and_dd  # module heet 'plot'


# -------- metrics helpers (lite, zonder extra deps) --------
def _max_drawdown(series: pd.Series) -> Dict[str, Any]:
    """Return max drawdown pct en duur (bars)."""
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    min_dd = float(dd.min())  # negatief getal
    # duur: langste aaneengesloten periode onder vorige top
    under = dd < 0
    # tel aaneengesloten 'under' streaks
    max_dur = int(
        pd.Series(np.where(under, 1, 0), index=series.index)
        .groupby((~under).cumsum())  # blokken tussen toppen
        .sum()
        .max()
        or 0
    )
    return {"max_drawdown_pct": min_dd * 100.0, "dd_duration_bars": max_dur}

def _sharpe_from_bar_returns(bar_rets: pd.Series, bars_per_year: float) -> float:
    """0% rf, annualized Sharpe op basis van bar-returns."""
    if len(bar_rets) < 2:
        return 0.0
    mu = float(bar_rets.mean())
    sigma = float(bar_rets.std(ddof=1))
    if sigma == 0:
        return 0.0
    return (mu / sigma) * math.sqrt(bars_per_year)

def _cagr_from_equity(eq: pd.Series, trading_days: int) -> float:
    """CAGR op basis van trading-dagen (≈252/jaar)."""
    if len(eq) == 0 or eq.iloc[0] <= 0:
        return 0.0
    years = max(trading_days, 1) / 252.0
    return float(eq.iloc[-1]) ** (1.0 / years) - 1.0

# -----------------------------------------------------------

def run(csv: str, start: Optional[str] = None, end: Optional[str] = None, window: int = 20) -> Path:
    # 1) Data laden
    df = fetch_data(csv_path=csv, start=start, end=end)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Data index must be DatetimeIndex")
    df = df.sort_index()

    # 2) Health + dag-samenvatting (alleen loggen)
    logger.info(health_line(df, expected_freq="15min"))
    days, min_bars, mean_bars = summarize_day_health(df)
    logger.info(f"DAYS {{'count': {days}, 'min_bars': {min_bars}, 'mean_bars': {mean_bars:.1f}}}")

    # 3) Simpel breakout-signaal (gap-safe, bar-based)  -> boolean Series
    sig = breakout_long(df["close"], df["high"], window=window).astype("boolean")

    # 4) Naïeve equity (1x notional; geen fees/slippage)
    pos = sig.fillna(False).astype(int)                 # 0/1
    bar_rets = df["close"].pct_change().fillna(0.0) * pos  # bar-PnL
    eq = (1.0 + bar_rets).cumprod().rename("Equity")

    # 5) Metrics (lite maar nuttig)
    bars_per_year = max(mean_bars, 1.0) * 252.0 if days > 0 else 252.0 * 56  # 56≈bars/dag @15m voor +/− 14h sessie
    md = _max_drawdown(eq)
    metrics = {
        "entries": int((sig & (~sig.shift(1).fillna(False))).sum()),
        "bars": int(len(df)),
        "final_equity": float(eq.iloc[-1]) if len(eq) else 1.0,
        "return_total_pct": (float(eq.iloc[-1]) - 1.0) * 100.0 if len(eq) else 0.0,
        "cagr_pct": _cagr_from_equity(eq, days) * 100.0 if days else 0.0,
        "sharpe": _sharpe_from_bar_returns(bar_rets.replace([np.inf, -np.inf], 0.0), bars_per_year),
        **md,
    }

    # 6) Output pad + visuals
    outdir = Path("output/plots") / pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    if os.getenv("SOPHY_VIZ") == "1":
        save_equity_and_dd(eq, outdir)

    # 7) Bewaar equity + metrics
    eq.to_csv(outdir / "equity.csv", index=True)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 8) Log human-friendly samenvatting
    logger.info(
        "METRICS "
        f"entries={metrics['entries']} total_return={metrics['return_total_pct']:.2f}% "
        f"CAGR={metrics['cagr_pct']:.2f}% sharpe={metrics['sharpe']:.2f} "
        f"maxDD={metrics['max_drawdown_pct']:.2f}% dur={metrics['dd_duration_bars']} bars"
    )

    return outdir

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sophy4Lite breakout demo (data-health + viz + metrics)")
    p.add_argument("--csv", required=True, help="Path to CSV (time,open,high,low,close[,volume])")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()
    run(args.csv, args.start, args.end, window=args.window)
