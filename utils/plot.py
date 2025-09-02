# utils/plot.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_and_dd(equity: pd.Series, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Equity
    fig = plt.figure(figsize=(10, 4))
    equity.rename("Equity").plot()
    plt.title("Equity")
    plt.tight_layout()
    fig.savefig(outdir / "equity.png", dpi=120)
    plt.close(fig)
    # Drawdown
    dd = (equity / equity.cummax() - 1.0).rename("Drawdown")
    fig = plt.figure(figsize=(10, 3))
    dd.plot()
    plt.title("Drawdown")
    plt.tight_layout()
    fig.savefig(outdir / "drawdown.png", dpi=120)
    plt.close(fig)
