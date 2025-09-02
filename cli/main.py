# cli/main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cli")

from backtest.runner import run_backtest
from backtest.runner_vbt import run_backtest_vbt  # blijft beschikbaar

app = typer.Typer(
    help="Sophy4Lite CLI for trading backtests",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def backtest(
    symbol: str = typer.Option("GER40.cash", help="Trading symbol, e.g. GER40.cash or US30.cash"),
    timeframe: str = typer.Option("M1", help="Timeframe label (CSV moet M1 zijn voor ORB)"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    csv: Optional[Path] = typer.Option(None, help="Path to CSV data file (UTC OHLC)"),
    engine: str = typer.Option("native", help="Backtest engine: native or vbt"),
    strategy: str = typer.Option("breakout", help="Strategy: breakout | premarket_orb | order_block"),
    # risk/fees/ATR
    atr_period: int = typer.Option(14, help="ATR period"),
    atr_sl: float = typer.Option(1.0, help="ATR SL multiplier"),
    atr_tp: float = typer.Option(1.8, help="ATR TP multiplier"),
    fee_bps: float = typer.Option(2.0, help="Fee per leg in bps"),
    risk_pct: float = typer.Option(1.0, help="Risk per trade in % of equity"),
    equity0: float = typer.Option(20_000.0, help="Starting equity"),
    entry_slip_pts: float = typer.Option(0.1, help="Entry slippage (pts)"),
    sl_slip_pts: float = typer.Option(0.5, help="SL slippage (pts)"),
    tp_slip_pts: float = typer.Option(0.0, help="TP slippage (pts)"),
    # opening-breakout specifics
    open_window_bars: int = typer.Option(4, help="Bars after open (breakout only)"),
    confirm: str = typer.Option("close", help="Confirmation: close|wick"),
    # ORB specifics
    session_open_local: Optional[str] = typer.Option(None, help="(ORB) cash-open, e.g. 09:00 or 09:30"),
    session_tz: Optional[str] = typer.Option(None, help="(ORB) timezone, e.g. Europe/Berlin"),
    premarket_minutes: int = typer.Option(60, help="(ORB) minutes for premarket range"),
    # output
    outdir: Optional[Path] = typer.Option(Path("output"), help="Output directory"),
):
    """Run a backtest with either the native or vectorbt engine."""
    logger.info(f"Starting backtest: strategy={strategy}, engine={engine}, symbol={symbol}, timeframe={timeframe}, csv={csv}")
    if engine == "native" and csv is None:
        console.print("[red]Error:[/red] CSV is required for native engine")
        raise typer.Exit(code=2)

    params = {
        "atr_period": atr_period,
        "atr_sl": atr_sl,
        "atr_tp": atr_tp,
        "fee_bps": fee_bps,
        "risk_pct": risk_pct,
        "equity0": equity0,
        "entry_slip_pts": entry_slip_pts,
        "sl_slip_pts": sl_slip_pts,
        "tp_slip_pts": tp_slip_pts,
        "open_window_bars": open_window_bars,
        "confirm": confirm,
        "premarket_minutes": premarket_minutes,
    }
    if session_open_local:
        params["session_open_local"] = session_open_local
    if session_tz:
        params["session_tz"] = session_tz

    try:
        if engine == "native":
            df_eq, trades, metrics = run_backtest(
                strategy_name=strategy,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                csv=str(csv) if csv else None,
                engine=engine,
                outdir=str(outdir) if outdir else None,
            )
        elif engine == "vbt":
            # behoud jouw bestaande VBT-flow (bv. order_block)
            if strategy == "breakout":
                raise ValueError("Breakout strategy not supported in VBT engine; use native engine.")
            df_eq, trades, metrics = run_backtest_vbt(
                strategy_name=strategy,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                csv_path=csv,
            )
        else:
            raise ValueError("engine must be 'native' or 'vbt'")

        # Tabel-summary (metrics die ontbreken worden 0/NA getoond)
        tbl = Table(title=f"Backtest Summary ({strategy}, {engine})")
        tbl.add_column("Metric"); tbl.add_column("Value", justify="right")
        order = ["final_equity","return_total_pct","n_trades","winrate_pct","avg_rr","sharpe","max_drawdown_pct"]
        for k in order:
            v = metrics.get(k, None)
            if isinstance(v, (int, float)):
                tbl.add_row(k, f"{v:,.4f}")
            else:
                tbl.add_row(k, "-" if v is None else str(v))
        console.print(tbl)

        # Opslaan
        out = outdir or Path("output")
        Path(out).mkdir(exist_ok=True)
        df_eq.to_csv(Path(out) / f"equity_{strategy}_{engine}.csv")
        trades.to_csv(Path(out) / f"trades_{strategy}_{engine}.csv", index=False)
        console.print(f"[green]Saved[/green] {Path(out) / f'equity_{strategy}_{engine}.csv'} and {Path(out) / f'trades_{strategy}_{engine}.csv'}")
        logger.info("Backtest completed successfully")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
