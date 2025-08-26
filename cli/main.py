from __future__ import annotations
import typer
from pathlib import Path
from typing import Optional
from rich import print
from rich.table import Table

from optimizer.optimizer import Optimizer
from utils.metrics import summarize_equity_metrics

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def backtest(
    symbol: str = typer.Option(..., help="Instrument symbol, e.g., XAUUSD"),
    timeframe: str = typer.Option(..., help="Timeframe, e.g., H1"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    csv: Optional[Path] = typer.Option(None, help="Path to CSV with OHLCV"),
    engine: str = typer.Option("native", help="Backtest engine: native or vbt"),
):
    if engine == "native":
        from backtest.runner import run_backtest
        df_eq, trades = run_backtest(
            strategy_name="order_block_simple",
            params={"lookback_bos": 20, "min_body_pct": 0.55},
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            csv_path=csv,
        )
        metrics = summarize_equity_metrics(df_eq, trades)

    elif engine == "vbt":
        from backtest.runner_vbt import run_backtest_vbt
        df_eq, trades, metrics = run_backtest_vbt(
            strategy_name="order_block_simple",
            params={"fast": 10, "slow": 20},
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            csv_path=csv,
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
    print(table)


@app.command()
def optimize(
    symbol: str = typer.Option(...),
    timeframe: str = typer.Option(...),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    csv: Optional[Path] = typer.Option(None),
):
    param_grid = {
        "lookback_bos": [15, 20, 30],
        "min_body_pct": [0.5, 0.6, 0.7],
        "rr": [1.2, 1.5, 2.0],
        "stop_buffer_pct": [0.0003, 0.0005, 0.0008],
        "max_concurrent": [1],
    }

    opt = Optimizer(
        strategy_name="order_block_simple",
        param_grid=param_grid,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        csv_path=csv,
    )
    df = opt.run()
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "optimizer_results.csv", index=False)

    best = opt.select_best(min_trades=50)
    print(f"[bold green]Best config[/]: {best['params']}")

    profiles = Path("profiles")
    profiles.mkdir(exist_ok=True)
    opt.export_best_config(profiles)
    print("Wrote profiles/live.yaml")


if __name__ == "__main__":
    app()
