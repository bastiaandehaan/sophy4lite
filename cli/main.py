# cli/main.py
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cli")

from strategies.mtf_confluence import MTFParams
from backtest.data_loader import fetch_data
from backtest.mtf_exec_fast import (
    backtest_mtf_confluence_fast as backtest_func,
    MTFExecCfg,
)

app = typer.Typer(help="Sophy4Lite CLI", no_args_is_help=True, add_completion=False)
console = Console()


def _save_outputs(eq, trades, metrics, outdir: Path, tag: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # equity series
    (outdir / f"equity_{tag}.csv").write_text(eq.to_csv())
    # trades
    trades.to_csv(outdir / f"trades_{tag}.csv", index=False)
    # metrics
    with open(outdir / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


@app.command()
def confluence(
    symbol: str = typer.Argument("GER40.cash"),
    csv: Path = typer.Option(..., help="Path to M1 CSV"),
    start: str | None = typer.Option(None, help="Start YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="End YYYY-MM-DD"),
    server_tz: str = typer.Option("Europe/Athens", help="Server timezone (e.g. Europe/Athens)"),
    session_start: str = typer.Option("09:00", help="Session start (server TZ)"),
    session_end: str = typer.Option("17:00", help="Session end (server TZ)"),
    risk_pct: float = typer.Option(1.0, min=0.01, max=10.0, help="Risk %% per trade"),
    min_score: float = typer.Option(0.65, help="Min confluence score"),
    max_daily: int = typer.Option(1, help="Max trades per day"),
    mark_to_market: bool = typer.Option(False, help="Mark-to-market open positions at dataset end"),
    output_json: bool = typer.Option(False, help="Print metrics as JSON"),
    outdir: Path | None = typer.Option(Path("output"), help="Directory to save results"),
):
    """
    Run Multi-Timeframe Confluence backtest on CSV (M1 OHLCV).
    Uses the fast vectorized engine.
    """
    # 1) Data inladen met consistente TZ (converteren gebeurt in fetch_data)
    df = fetch_data(csv_path=csv, server_tz=server_tz, start=start, end=end)

    # 2) Strategie-parameters
    params = MTFParams(min_confluence_score=min_score)

    # 3) Backtest-config
    cfg = MTFExecCfg(
        risk_frac=risk_pct / 100.0,
        mark_to_market=mark_to_market,
    )

    # 4) Run
    eq, trades, metrics = backtest_func(
        df,
        symbol,
        params,
        cfg,
        session_start=session_start,
        session_end=session_end,
        max_trades_per_day=max_daily,
    )

    # 5) Output
    tag = f"confluence_{symbol}"

    if output_json:
        payload = {
            "symbol": symbol,
            "period": f"{df.index[0]} to {df.index[-1]}",
            "metrics": metrics,
            "trades_summary": {
                "total": int(len(trades)),
                "longs": int((trades["side"] == "long").sum()) if "side" in trades else 0,
                "shorts": int((trades["side"] == "short").sum()) if "side" in trades else 0,
            },
        }
        console.print_json(data=payload)
    else:
        console.print("[bold green]=== Confluence Backtest Metrics ===[/]")
        for k, v in metrics.items():
            console.print(f"{k:20}: {v}")
        if len(trades) > 0:
            console.print("\n[bold]Sample trades[/]")
            console.print(trades.head())

    if outdir:
        _save_outputs(eq, trades, metrics, outdir, tag)


if __name__ == "__main__":
    app()
