# cli/main.py
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cli")

from backtest.mtf_exec import backtest_mtf_confluence, MTFExecCfg
from strategies.mtf_confluence import MTFParams

app = typer.Typer(
    help="Sophy4Lite CLI for trading backtests",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def confluence(
    symbol: str,
    csv: Path,
    server_tz: str = "Europe/Athens",
    session_start: str = "09:00",
    session_end: str = "17:00",
):
    """Run Multi-Timeframe Confluence backtest on CSV (M1 OHLCV)."""
    import pandas as pd

    df = pd.read_csv(csv, parse_dates=["time"], index_col="time")

    # Make timestamps tz-aware. If the CSV is naive, we assume server TZ.
    if df.index.tz is None:
        df.index = df.index.tz_localize(server_tz)

    eq, trades, metrics = backtest_mtf_confluence(
        df, symbol, MTFParams(), MTFExecCfg(),
        session_start=session_start, session_end=session_end
    )

    console.print("[bold green]=== Confluence Backtest Metrics ===[/]")
    for k, v in metrics.items():
        console.print(f"{k:18}: {v}")

    if len(trades) > 0:
        console.print("[bold]Sample trades[/]")
        console.print(trades.head())


if __name__ == "__main__":
    app()
