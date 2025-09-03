from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


# -------- Helpers (zonder externe deps) --------
def display_metrics_table(metrics: dict) -> None:
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
        eq.to_csv(eq_path)
    except Exception:
        pass

    try:
        trades.to_csv(tr_path, index=True)
    except Exception:
        pass

    try:
        with open(mt_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
    except Exception:
        pass


def _run_confluence(
    *,
    symbol: str,
    csv: Path,
    start: Optional[str],
    end: Optional[str],
    server_tz: str,
    session_start: str,
    session_end: str,
    risk_pct: float,
    min_score: float,
    max_daily: int,
    mark_to_market: bool,
    output_json: bool,
    outdir: Optional[Path],
) -> None:
    # 1) Load
    from backtest.data_loader import fetch_data
    df = fetch_data(csv_path=csv, server_tz=server_tz, start=start, end=end)

    # 2) Params
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


# ---- Subcommand: confluence ----
@app.command(help="Run Multi-Timeframe Confluence backtest on CSV (M1 OHLCV). Uses the fast vectorized engine.")
def confluence(
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
    _run_confluence(
        symbol=symbol,
        csv=csv,
        start=start,
        end=end,
        server_tz=server_tz,
        session_start=session_start,
        session_end=session_end,
        risk_pct=risk_pct,
        min_score=min_score,
        max_daily=max_daily,
        mark_to_market=mark_to_market,
        output_json=output_json,
        outdir=outdir,
    )


# ---- Callback to allow default command without subcommand ----
@app.callback(invoke_without_command=True)
def root_as_default(
    ctx: typer.Context,
    symbol: Optional[str] = typer.Argument(
        None, help="Instrument, bv. GER40.cash (je kunt ook het subcommand 'confluence' gebruiken)"
    ),
    csv: Optional[Path] = typer.Option(None, help="Path to M1 CSV"),
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
    # Als er een subcommand is aangeroepen, doe niks hier.
    if ctx.invoked_subcommand is not None:
        return

    # Geen subcommand → gebruik confluence als default.
    # Vereiste velden controleren:
    if symbol is None or csv is None:
        # Toon help als verplichte stukken ontbreken
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

    _run_confluence(
        symbol=symbol,
        csv=csv,
        start=start,
        end=end,
        server_tz=server_tz,
        session_start=session_start,
        session_end=session_end,
        risk_pct=risk_pct,
        min_score=min_score,
        max_daily=max_daily,
        mark_to_market=mark_to_market,
        output_json=output_json,
        outdir=outdir,
    )


def main():
    app()


if __name__ == "__main__":
    main()
