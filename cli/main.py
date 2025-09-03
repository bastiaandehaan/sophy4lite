from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

# Maak één Typer-app
app = typer.Typer(add_completion=False)

# --------- kleine helpers (niet afhankelijk van Rich) ---------
def display_metrics_table(metrics: dict) -> None:
    # Eenvoudige, robuuste print zonder externe deps
    print("\n=== Metrics ===")
    for k in sorted(metrics.keys()):
        print(f"{k:>18}: {metrics[k]}")
    print("===============")


def save_results(eq, trades, metrics: dict, outdir: Path, symbol: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    eq_path = outdir / f"equity_{symbol}.csv"
    tr_path = outdir / f"trades_{symbol}.csv"
    mt_path = outdir / f"metrics_{symbol}.json"

    try:
        # eq is doorgaans pd.Series
        eq.to_csv(eq_path)
    except Exception:
        pass

    try:
        # trades is doorgaans pd.DataFrame
        trades.to_csv(tr_path, index=True)
    except Exception:
        pass

    try:
        with open(mt_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
    except Exception:
        pass


@app.command(help="Run Multi-Timeframe Confluence backtest on CSV (M1 OHLCV). Uses the fast vectorized engine.")
def confluence(
    # BELANGRIJK: symbol is VERPLICHT → geen verwarring meer (zoals 'confluence' als symbol)
    symbol: str = typer.Argument(..., help="Instrument, bv. GER40.cash"),
    csv: Path = typer.Option(..., help="Path to M1 CSV"),
    start: Optional[str] = typer.Option(None, help="Start YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End YYYY-MM-DD"),
    server_tz: str = typer.Option("Europe/Athens", help="Server timezone (e.g. Europe/Athens)"),
    session_start: str = typer.Option("09:00", help="Session start (server TZ)"),
    session_end: str = typer.Option("17:00", help="Session end (server TZ)"),
    risk_pct: float = typer.Option(1.0, min=0.01, max=10.0, help="Risk % per trade"),
    min_score: float = typer.Option(0.65, help="Min confluence score"),
    max_daily: int = typer.Option(1, help="Max trades per day"),
    mark_to_market: bool = typer.Option(False, help="Mark-to-market open positions at dataset end"),
    output_json: bool = typer.Option(False, help="Print metrics as JSON"),
    outdir: Optional[Path] = typer.Option(Path("output"), help="Directory to save results"),
):
    # 1) Load data (TZ-consistent)
    from backtest.data_loader import fetch_data
    df = fetch_data(csv_path=csv, server_tz=server_tz, start=start, end=end)

    # 2) Strategy params
    from strategies.mtf_confluence import MTFParams
    params = MTFParams(min_confluence_score=min_score)

    # 3) Engine (fast)
    from backtest.mtf_exec_fast import backtest_mtf_confluence_fast, MTFExecCfg

    cfg = MTFExecCfg(
        risk_frac=risk_pct / 100.0,
        mark_to_market=mark_to_market,
    )

    # 4) Run
    eq, trades, metrics = backtest_mtf_confluence_fast(
        df,
        symbol,
        params,
        cfg,
        session_start=session_start,
        session_end=session_end,
        max_trades_per_day=max_daily,
    )

    # 5) Output
    if output_json:
        output = {
            "symbol": symbol,
            "period": f"{df.index[0]} to {df.index[-1]}",
            "metrics": metrics,
            "trades_summary": {
                "total": int(len(trades)),
                "longs": int((trades["side"] == "long").sum()) if "side" in trades else 0,
                "shorts": int((trades["side"] == "short").sum()) if "side" in trades else 0,
            },
        }
        print(json.dumps(output, default=float, indent=2))
    else:
        display_metrics_table(metrics)

    if outdir:
        save_results(eq, trades, metrics, outdir, symbol)


if __name__ == "__main__":
    # Sta direct script-run toe
    app()
