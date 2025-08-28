from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional, Annotated
from rich.table import Table
from rich.console import Console

import pandas as pd

from backtest.runner import run_backtest
from backtest.runner_vbt import run_backtest_vbt
from utils.metrics import summarize_equity_metrics

from strategies.breakout_signals import BreakoutParams, DEFAULT_SPECS, SymbolSpec
from backtest.breakout_exec import backtest_breakout, BTExecCfg

app = typer.Typer(help="Sophy4Lite CLI", no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def backtest(
    symbol: str = typer.Option(..., help="Trading symbol, e.g. XAUUSD"),
    timeframe: str = typer.Option(..., help="Timeframe, e.g. H1"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    csv: Optional[Path] = typer.Option(None, help="Path to CSV data file"),
    engine: str = typer.Option("native", help="Backtest engine: native or vbt"),
):
    """Run a backtest with either the native or vectorbt engine."""
    if engine == "native":
        df_eq, trades = run_backtest(
            strategy_name="order_block_simple",
            params={"lookback_bos": 20, "min_body_pct": 0.55},
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            csv=csv,
        )
        metrics = summarize_equity_metrics(df_eq, trades)

    elif engine == "vbt":
        df_eq, trades, metrics = run_backtest_vbt(
            strategy_name="order_block_simple",
            params={"fast": 10, "slow": 20},
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            csv=csv,
        )
    else:
        raise ValueError("engine must be 'native' or 'vbt'")

    table = Table(title=f"Backtest Summary ({engine})")
    for col in ["sharpe", "max_dd", "dd_duration", "total_return", "n_trades"]:
        table.add_column(col)
    table.add_row(
        f"{metrics['sharpe']:.2f}",
        f"{metrics['max_dd']:.2%}",
        f"{metrics['dd_duration']}",
        f"{metrics['total_return']:.2%}",
        f"{metrics['n_trades']}",
    )
    console.print(table)


@app.command(help="Breakout backtest (close-confirm or pending-stop) with ATR RR, FTMO guards, and proper TZ.")
def breakout(
    csv: Annotated[Path, typer.Option("--csv", exists=True, help="CSV with time,open,high,low,close[,volume]")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "DE40",
    broker_tz: Annotated[str, typer.Option("--broker-tz")] = "UTC",
    start: Annotated[str | None, typer.Option("--start")] = None,
    end: Annotated[str | None, typer.Option("--end")] = None,
    session_start: Annotated[int, typer.Option("--session-start")] = 7,
    session_end: Annotated[int, typer.Option("--session-end")] = 8,
    cancel_at: Annotated[int, typer.Option("--cancel-at")] = 17,
    min_range: Annotated[float, typer.Option("--min-range")] = 10.0,
    mode: Annotated[str, typer.Option("--mode", help="'close_confirm' or 'pending_stop'")] = "close_confirm",
    confirm: Annotated[int, typer.Option("--confirm", help="#closed bars beyond level (close_confirm only)")] = 1,
    vol_pctl: Annotated[float, typer.Option("--vol-pctl", help="0 disables volume filter")] = 0.0,
    vol_lookback: Annotated[int, typer.Option("--vol-lookback")] = 50,
    atr_period: Annotated[int, typer.Option("--atr-period")] = 14,
    atr_sl: Annotated[float, typer.Option("--atr-sl")] = 1.0,
    atr_tp: Annotated[float, typer.Option("--atr-tp")] = 1.5,
    offset: Annotated[float, typer.Option("--offset")] = 0.0,
    spread: Annotated[float, typer.Option("--spread")] = 0.0,
    fee_bps: Annotated[float, typer.Option("--fee-bps", help="per side, e.g. 2 = 0.02%")] = 2.0,
    risk_pct: Annotated[float, typer.Option("--risk-pct")] = 1.0,
    vpp_override: Annotated[float | None, typer.Option("--vpp", help="value per point override")] = None,
):
    # --- input checks ---
    if risk_pct <= 0:
        typer.echo("[ERROR] --risk-pct must be > 0"); raise typer.Exit(code=1)
    if fee_bps < 0:
        typer.echo("[ERROR] --fee-bps cannot be negative"); raise typer.Exit(code=1)
    if mode == "close_confirm" and confirm <= 0:
        typer.echo("[ERROR] --confirm must be > 0 in close_confirm mode"); raise typer.Exit(code=1)

    df = pd.read_csv(csv, parse_dates=["time"])
    df.columns = [c.lower() for c in df.columns]
    if "time" not in df.columns:
        typer.echo("[ERROR] CSV must contain a 'time' column."); raise typer.Exit(code=1)
    df = df.set_index("time").sort_index()
    if start: df = df.loc[pd.Timestamp(start):]
    if end:   df = df.loc[:pd.Timestamp(end)]
    if df.empty:
        typer.echo("[ERROR] No data in selected range."); raise typer.Exit(code=1)

    p = BreakoutParams(
        broker_tz=broker_tz,
        session_start_h=session_start, session_end_h=session_end, cancel_at_h=cancel_at,
        min_range_pts=min_range, vol_percentile=vol_pctl, vol_lookback=vol_lookback,
        confirm_bars=confirm, mode=mode if mode in ("close_confirm","pending_stop") else "close_confirm",
        atr_period=atr_period, atr_sl_mult=atr_sl, atr_tp_mult=atr_tp,
        offset_pts=offset, spread_pts=spread
    )

    specs = DEFAULT_SPECS.copy()
    if vpp_override is not None:
        base = specs.get(symbol, SymbolSpec(value_per_point=1.0, min_tick=0.1))
        specs[symbol] = SymbolSpec(value_per_point=float(vpp_override), min_tick=base.min_tick)

    eq, trades, met = backtest_breakout(
        df, symbol, p,
        BTExecCfg(
            equity0=20_000.0,
            fee_rate=float(fee_bps) / 10000.0,
            slippage_pts=0.5,
            specs=specs,
            risk_frac=float(risk_pct) / 100.0
        )
    )

    tbl = Table(title="Breakout Backtest (ATR RR, FTMO guards)")
    tbl.add_column("Metric"); tbl.add_column("Value", justify="right")
    for k in ["final_equity","return_total_pct","n_trades","winrate_pct","avg_rr","sharpe","max_drawdown_pct"]:
        v = met[k]; tbl.add_row(k, f"{v:,.4f}" if isinstance(v,(int,float)) else str(v))
    console.print(tbl)

    out = Path("outputs"); out.mkdir(exist_ok=True)
    eq.to_csv(out/"equity_breakout.csv")
    trades.to_csv(out/"trades_breakout.csv", index=False)
    console.print(f"[green]Saved[/green] {out/'equity_breakout.csv'} and {out/'trades_breakout.csv'}")


if __name__ == "__main__":
    app()