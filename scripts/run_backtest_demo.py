# scripts/run_backtest_demo.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import logger
from backtest.data_loader import fetch_data
from strategies.breakout_signals import breakout_long
from utils.data_health import health_line
from utils.days import summarize_day_health
from utils.plot import save_equity_and_dd  # let op: module heet 'plot', niet 'plots'

def run(csv: str, start: Optional[str] = None, end: Optional[str] = None, window: int = 20) -> Path:
    # 1) Data laden
    df = fetch_data(csv_path=csv, start=start, end=end)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Data index must be DatetimeIndex")
    df = df.sort_index()

    # 2) Data-health + dag-samenvatting (alleen loggen)
    logger.info(health_line(df, expected_freq="15T"))
    days, min_bars, mean_bars = summarize_day_health(df)
    logger.info(f"DAYS {{'count': {days}, 'min_bars': {min_bars}, 'mean_bars': {mean_bars:.1f}}}")

    # 3) Eenvoudig signaal (gap-safe, bar-based)
    sig = breakout_long(df["close"], df["high"], window=window)

    # 4) Naïeve equity (1x notional; geen fees/slippage/SL/TP)
    pos = sig.astype(int)
    rets = df["close"].pct_change().fillna(0.0)
    eq = (1.0 + rets * pos).cumprod().rename("Equity")

    # 5) Optionele visuals via env var SOPHY_VIZ=1
    outdir = Path("output/plots") / pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    if os.getenv("SOPHY_VIZ") == "1":
        save_equity_and_dd(eq, outdir)
        logger.info(f"Viz saved to {outdir}")

    # 6) Korte run-samenvatting
    n_entries = int((pos & ~pos.shift(1).fillna(False)).sum())
    logger.info(f"ENTRIES {n_entries} | FINAL_EQ {eq.iloc[-1]:.4f} | BARS {len(df)}")

    # 7) Bewaar equity (klein, handig om snel te openen)
    outdir.mkdir(parents=True, exist_ok=True)
    eq.to_csv(outdir / "equity.csv", index=True)
    return outdir

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sophy4Lite breakout demo (data-health + viz opt-in)")
    p.add_argument("--csv", required=True, help="Path to CSV (time,open,high,low,close[,volume])")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args()
    run(args.csv, args.start, args.end, window=args.window)
