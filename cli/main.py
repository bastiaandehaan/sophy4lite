from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional, Annotated
from rich.table import Table
from rich.console import Console
import pandas as pd
from backtest.runner import run_backtest
from backtest.runner_vbt import run_backtest_vbt
from backtest.breakout_exec import backtest_breakout, BTExecCfg
from strategies.breakout_signals import BreakoutParams, DEFAULT_SPECS, SymbolSpec
from config import logger

app = typer.Typer(
    help="Sophy4Lite CLI for trading backtests",
    no_args_is_help=True,
    add_completion=False
)
console = Console()

@app.command()
def backtest(
    symbol: str = typer.Option("GER40.cash", help="Trading symbol, e.g. GER40.cash"),
    timeframe: str = typer.Option("M15", help="Timeframe, e.g. M15"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    csv: Optional[Path] = typer.Option("data/GER40.cash_M15.csv", help="Path to CSV data file"),
    engine: str = typer.Option("native", help="Backtest engine: native or vbt"),
    strategy: str = typer.Option("breakout", help="Strategy: breakout or order_block"),
):
    """Run a backtest with either the native or vectorbt engine."""
    logger.info(f"Starting backtest: strategy={strategy}, engine={engine}, symbol={symbol}, timeframe={timeframe}, csv={csv}")

    # Standaard parameters voor breakout
    params = {
        "min_range": 10.0,
        "vol_pctl": 40.0,
        "vol_lookback": 60,
        "confirm": 2,
        "mode": "close_confirm",
        "atr_period": 14,
        "atr_sl": 1.0,
        "atr_tp": 1.8,
        "fee_bps": 2.0,
        "risk_pct": 1.0,
        "lookback_bos": 3,  # Voor order_block
    }

    try:
        if engine == "native":
            df_eq, trades, metrics = run_backtest(
                strategy_name=strategy,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                csv=csv,
            )
        elif engine == "vbt":
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

        # Toon resultaten
        tbl = Table(title=f"Backtest Summary ({strategy}, {engine})")
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        for k in ["final_equity", "return_total_pct", "n_trades", "winrate_pct", "avg_rr", "sharpe", "max_drawdown_pct"]:
            v = metrics.get(k, 0.0)
            tbl.add_row(k, f"{v:,.4f}" if isinstance(v, (int, float)) else str(v))
        console.print(tbl)

        # Opslaan outputs
        out = Path("output")
        out.mkdir(exist_ok=True)
        df_eq.to_csv(out / f"equity_{strategy}_{engine}.csv")
        trades.to_csv(out / f"trades_{strategy}_{engine}.csv", index=False)
        console.print(f"[green]Saved[/green] {out / f'equity_{strategy}_{engine}.csv'} and {out / f'trades_{strategy}_{engine}.csv'}")
        logger.info("Backtest completed successfully")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

@app.command()
def breakout(
    csv: Annotated[Path, typer.Option("--csv", exists=True, help="CSV with time,open,high,low,close[,volume]")] = Path("data/GER40.cash_M15.csv"),
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "GER40.cash",
    broker_tz: Annotated[str, typer.Option("--broker-tz")] = "UTC",
    start: Annotated[str | None, typer.Option("--start")] = None,
    end: Annotated[str | None, typer.Option("--end")] = None,
    session_start: Annotated[int, typer.Option("--session-start")] = 7,
    session_end: Annotated[int, typer.Option("--session-end")] = 8,
    cancel_at: Annotated[int, typer.Option("--cancel-at")] = 17,
    min_range: Annotated[float, typer.Option("--min-range")] = 10.0,
    mode: Annotated[str, typer.Option("--mode", help="'close_confirm' or 'pending_stop'")] = "close_confirm",
    confirm: Annotated[int, typer.Option("--confirm", help="#closed bars beyond level (close_confirm only)")] = 2,
    vol_pctl: Annotated[float, typer.Option("--vol-pctl", help="0 disables volume filter")] = 40.0,
    vol_lookback: Annotated[int, typer.Option("--vol-lookback")] = 60,
    atr_period: Annotated[int, typer.Option("--atr-period")] = 14,
    atr_sl: Annotated[float, typer.Option("--atr-sl")] = 1.0,
    atr_tp: Annotated[float, typer.Option("--atr-tp")] = 1.8,
    offset: Annotated[float, typer.Option("--offset")] = 0.0,
    spread: Annotated[float, typer.Option("--spread")] = 0.0,
    fee_bps: Annotated[float, typer.Option("--fee-bps", help="per side, e.g. 2 = 0.02%")] = 2.0,
    risk_pct: Annotated[float, typer.Option("--risk-pct")] = 1.0,
    vpp_override: Annotated[float | None, typer.Option("--vpp", help="value per point override")] = None,
):
    """Run ATR-based breakout backtest with FTMO guards for GER40.cash."""
    logger.info(f"Starting breakout backtest: symbol={symbol}, mode={mode}, csv={csv}")

    # Input validatie
    if risk_pct <= 0:
        console.print("[red]Error:[/red] --risk-pct must be > 0")
        logger.error("Invalid risk_pct")
        raise typer.Exit(code=1)
    if fee_bps < 0:
        console.print("[red]Error:[/red] --fee-bps cannot be negative")
        logger.error("Invalid fee_bps")
        raise typer.Exit(code=1)
    if mode == "close_confirm" and confirm <= 0:
        console.print("[red]Error:[/red] --confirm must be > 0 in close_confirm mode")
        logger.error("Invalid confirm_bars")
        raise typer.Exit(code=1)
    if mode not in ("close_confirm", "pending_stop"):
        console.print(f"[red]Error:[/red] --mode must be 'close_confirm' or 'pending_stop', got {mode}")
        logger.error(f"Invalid mode: {mode}")
        raise typer.Exit(code=1)

    try:
        # Laad en prepareer data
        df = pd.read_csv(csv, parse_dates=["time"])
        df.columns = [c.lower() for c in df.columns]
        if "time" not in df.columns:
            console.print("[red]Error:[/red] CSV must contain a 'time' column.")
            logger.error("CSV missing 'time' column")
            raise typer.Exit(code=1)
        df = df.set_index("time").sort_index()
        if start:
            df = df.loc[pd.Timestamp(start):]
        if end:
            df = df.loc[:pd.Timestamp(end)]
        if df.empty:
            console.print("[red]Error:[/red] No data in selected range.")
            logger.error("Empty data after filtering")
            raise typer.Exit(code=1)

        # Stel BreakoutParams in
        p = BreakoutParams(
            broker_tz=broker_tz,
            session_start_h=session_start,
            session_end_h=session_end,
            cancel_at_h=cancel_at,
            min_range_pts=min_range,
            vol_percentile=vol_pctl,
            vol_lookback=vol_lookback,
            confirm_bars=confirm,
            mode=mode,
            atr_period=atr_period,
            atr_sl_mult=atr_sl,
            atr_tp_mult=atr_tp,
            offset_pts=offset,
            spread_pts=spread,
        )

        # Stel BTExecCfg in met symbol-specifieke waarde
        specs = DEFAULT_SPECS.copy()
        base_symbol = symbol.split(".")[0]  # Gebruik GER40 als fallback
        if base_symbol not in specs and symbol not in specs:
            specs[symbol] = SymbolSpec(value_per_point=1.0, min_tick=0.1)  # Default voor GER40.cash
        if vpp_override is not None:
            base = specs.get(symbol, specs.get(base_symbol, SymbolSpec(value_per_point=1.0, min_tick=0.1)))
            specs[symbol] = SymbolSpec(value_per_point=float(vpp_override), min_tick=base.min_tick)

        cfg = BTExecCfg(
            equity0=20_000.0,
            fee_rate=float(fee_bps) / 10000.0,
            slippage_pts=0.5,
            specs=specs,
            risk_frac=float(risk_pct) / 100.0,
        )

        # Run breakout backtest
        eq, trades, met = backtest_breakout(df, symbol, p, cfg)

        # Toon resultaten
        tbl = Table(title="Breakout Backtest (ATR RR, FTMO guards)")
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        for k in ["final_equity", "return_total_pct", "n_trades", "winrate_pct", "avg_rr", "sharpe", "max_drawdown_pct"]:
            v = met.get(k, 0.0)
            tbl.add_row(k, f"{v:,.4f}" if isinstance(v, (int, float)) else str(v))
        console.print(tbl)

        # Opslaan outputs
        out = Path("output")
        out.mkdir(exist_ok=True)
        eq.to_csv(out / "equity_breakout.csv")
        trades.to_csv(out / "trades_breakout.csv", index=False)
        console.print(f"[green]Saved[/green] {out / 'equity_breakout.csv'} and {out / 'trades_breakout.csv'}")
        logger.info("Breakout backtest completed successfully")

    except Exception as e:
        logger.error(f"Breakout backtest failed: {str(e)}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
