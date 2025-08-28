import typer
from pathlib import Path

from backtest.runner import run_backtest
from backtest.runner_vbt import run_backtest_vbt
from utils.metrics import print_summary

app = typer.Typer()

@app.command()
def backtest(
    symbol: str = typer.Option(..., help="Symbool (bv. GER40.cash)"),
    timeframe: str = typer.Option(..., help="Timeframe (bv. M15)"),
    start: str = typer.Option(..., help="Startdatum (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Einddatum (YYYY-MM-DD)"),
    csv: Path = typer.Option(..., help="Pad naar CSV-file"),
    engine: str = typer.Option("native", help="native | vbt"),
    strategy: str = typer.Option("breakout", help="breakout | order_block"),
):
    """Run a backtest with either the native or vectorbt engine."""

    params = {
        "lookback_bos": 20,      # Voor order_block (aanpassen naar wens)
        # Andere params kun je later via CLI optioneel toevoegen
    }

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
        print_summary(metrics, engine)
        trades.to_csv("outputs/trades.csv", index=False)
        df_eq.to_csv("outputs/equity.csv")
    elif engine == "vbt":
        df_eq, trades, metrics = run_backtest_vbt(
            strategy_name=strategy,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            csv_path=csv,
        )
        print_summary(metrics, engine)
        trades.to_csv("outputs/trades_vbt.csv", index=False)
        df_eq.to_csv("outputs/equity_vbt.csv")
    else:
        typer.echo(f"[ERROR] Ongeldige engine: {engine}")

if __name__ == "__main__":
    app()
