#!/usr/bin/env python3
"""
Sophy4Lite - data diagnostics & quick visuals.

Usage examples:
  python -m scripts.diagnostics --csv data/GER40.cash_M15.csv --start 2023-01-01 --end 2023-03-31
  python -m scripts.diagnostics --csv data/XAUUSD_H1.csv

Outputs:
  - output/plots/df_overview.txt         (shape, date range, tz, columns, NaN stats)
  - output/plots/close.png               (Close price over time)
  - output/plots/volume.png              (Volume over time, if available)
  - output/plots/atr14.png               (Rolling ATR(14))
  - output/plots/returns_hist.png        (Histogram of 1-bar returns)
  - output/plots/missing_heatmap.png     (Simple missingness visualization)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.data_loader import fetch_data


# ---------- Helpers ----------

def ensure_outdir() -> Path:
    outdir = Path("output/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def save_txt(outpath: Path, text: str) -> None:
    outpath.write_text(text, encoding="utf-8")

def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int = 14) -> pd.Series:
    h, l, c = series_high, series_low, series_close
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def plot_series(y: pd.Series, title: str, filepath: Path) -> None:
    fig = plt.figure(figsize=(10, 4))
    y.plot()
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(y.name or "")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)

def plot_hist(values: pd.Series, bins: int, title: str, filepath: Path) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.hist(values.dropna().values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)

def plot_missing_heatmap(df: pd.DataFrame, filepath: Path) -> None:
    """Simple missingness plot: rows vs columns (no seaborn dependency)."""
    miss = df.isna().astype(int)
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(miss.T, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xticks([])
    plt.title("Missingness (1=NaN, 0=valid)")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)

# ---------- Main diagnostics ----------

def summarize_df(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"shape: {df.shape}")
    if isinstance(df.index, pd.DatetimeIndex):
        lines.append(f"index: DatetimeIndex tz={df.index.tz}")
        if len(df.index) > 0:
            lines.append(f"range: {df.index.min()} -> {df.index.max()}")
        # crude frequency guess
        if len(df.index) >= 3:
            deltas = df.index.to_series().diff().dropna().value_counts().head(3)
            lines.append("top time deltas:")
            for delta, cnt in deltas.items():
                lines.append(f"  {delta}  ({cnt} steps)")
    else:
        lines.append("index: (not DatetimeIndex)")
    lines.append(f"columns: {list(df.columns)}")
    miss = df.isna().mean() * 100.0
    for c, p in miss.items():
        lines.append(f"NaN % in {c:>6}: {p:5.2f}%")
    dups = df.index.duplicated().sum() if isinstance(df.index, pd.DatetimeIndex) else 0
    lines.append(f"duplicate index entries: {dups}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Sophy4Lite data diagnostics & visuals")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with time,open,high,low,close[,volume]")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Load
    df = fetch_data(csv_path=args.csv, start=args.start, end=args.end)

    # Ensure required columns
    cols = ["open", "high", "low", "close"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Required column missing: {c}")
    if "volume" not in df.columns:
        df["volume"] = np.nan

    outdir = ensure_outdir()

    # Text overview
    overview = summarize_df(df)
    save_txt(outdir / "df_overview.txt", overview)

    # Basic plots
    plot_series(df["close"], "Close", outdir / "close.png")
    if df["volume"].notna().any():
        plot_series(df["volume"], "Volume", outdir / "volume.png")

    # ATR(14)
    atr14 = atr(df["high"], df["low"], df["close"], period=14).rename("ATR(14)")
    plot_series(atr14, "ATR(14)", outdir / "atr14.png")

    # Returns histogram
    rets = df["close"].pct_change()
    plot_hist(rets, bins=50, title="1-bar Returns", filepath=outdir / "returns_hist.png")

    # Missingness
    plot_missing_heatmap(df[["open","high","low","close","volume"]], outdir / "missing_heatmap.png")

    print("[OK] Wrote diagnostics to", outdir.resolve())
    print(overview)

if __name__ == "__main__":
    main()
