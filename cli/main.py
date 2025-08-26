import typer
from pathlib import Path
from typing import Optional
from rich.table import Table

from backtest.runner import run_backtest
from backtest.runner_vbt import run_backtest_vbt
from utils.metrics import summarize_equity_metrics

app = typer.Typer(help="Sophy4Lite CLI")


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
    print(table)


if __name__ == "__main__":
    app()
