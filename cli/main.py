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


def _run_confluence(*, symbol: str, csv: Path, start: Optional[str], end: Optional[str],
        server_tz: str, session_start: str, session_end: str, risk_pct: float,
        min_score: float, max_daily: int, mark_to_market: bool, output_json: bool,
        outdir: Optional[Path], ) -> None:
    # 1) Load
    from backtest.data_loader import fetch_data
    df = fetch_data(csv_path=csv, server_tz=server_tz, start=start, end=end)

    # 2) Params
    from strategies.mtf_confluence import MTFParams
    params = MTFParams(min_confluence_score=min_score)

    # 3) Engine (fast)
    from backtest.mtf_exec_fast import backtest_mtf_confluence_fast, MTFExecCfg
    cfg = MTFExecCfg(risk_frac=risk_pct / 100.0, mark_to_market=mark_to_market, )

    # 4) Run
    eq, trades, metrics = backtest_mtf_confluence_fast(df, symbol, params, cfg,
        session_start=session_start, session_end=session_end,
        max_trades_per_day=max_daily, )

    # 5) Output
    if output_json:
        output = {"symbol": symbol, "period": f"{df.index[0]} to {df.index[-1]}",
            "metrics": metrics, "trades_summary": {"total": int(len(trades)),
                "longs": int(
                    (trades["side"] == "long").sum()) if "side" in trades else 0,
                "shorts": int(
                    (trades["side"] == "short").sum()) if "side" in trades else 0, }, }
        print(json.dumps(output, default=float, indent=2))
    else:
        display_metrics_table(metrics)

    if outdir:
        save_results(eq, trades, metrics, outdir, symbol)


# ---- MAIN command (default) - expliciet zonder subcommands ----
@app.command(name="run", help="Run MTF Confluence backtest (default command)")
def run_backtest(
        csv: Path = typer.Option(..., "--csv", "-c", help="Path to M1 CSV file"),
        symbol: str = typer.Option("GER40.cash", "--symbol", "-s",
                                   help="Instrument symbol"),
        start: Optional[str] = typer.Option(None, "--start",
                                            help="Start date YYYY-MM-DD"),
        end: Optional[str] = typer.Option(None, "--end", help="End date YYYY-MM-DD"),
        server_tz: str = typer.Option("Europe/Athens", "--server-tz",
                                      help="Server timezone"),
        session_start: str = typer.Option("09:00", "--session-start",
                                          help="Session start time"),
        session_end: str = typer.Option("17:00", "--session-end",
                                        help="Session end time"),
        risk_pct: float = typer.Option(1.0, "--risk-pct", "-r", min=0.01, max=10.0,
                                       help="Risk % per trade"),
        min_score: float = typer.Option(0.65, "--min-score",
                                        help="Minimum confluence score"),
        max_daily: int = typer.Option(1, "--max-daily", help="Max trades per day"),
        mark_to_market: bool = typer.Option(False, "--mark-to-market",
                                            help="Mark open positions at end"),
        output_json: bool = typer.Option(False, "--output-json", "-j",
                                         help="Output as JSON"),
        outdir: Optional[Path] = typer.Option(Path("output"), "--outdir", "-o",
                                              help="Output directory"), ):
    """Main backtest command - gebruik dit voor normale runs."""
    _run_confluence(symbol=symbol, csv=csv, start=start, end=end, server_tz=server_tz,
        session_start=session_start, session_end=session_end, risk_pct=risk_pct,
        min_score=min_score, max_daily=max_daily, mark_to_market=mark_to_market,
        output_json=output_json, outdir=outdir, )


# ---- Legacy support: alias voor backward compatibility ----
@app.command(name="confluence", help="Alias for 'run' command (backward compatibility)")
def confluence(csv: Path = typer.Option(..., "--csv", "-c", help="Path to M1 CSV file"),
        symbol: str = typer.Option("GER40.cash", "--symbol", "-s",
                                   help="Instrument symbol"),
        start: Optional[str] = typer.Option(None, "--start",
                                            help="Start date YYYY-MM-DD"),
        end: Optional[str] = typer.Option(None, "--end", help="End date YYYY-MM-DD"),
        server_tz: str = typer.Option("Europe/Athens", "--server-tz",
                                      help="Server timezone"),
        session_start: str = typer.Option("09:00", "--session-start",
                                          help="Session start time"),
        session_end: str = typer.Option("17:00", "--session-end",
                                        help="Session end time"),
        risk_pct: float = typer.Option(1.0, "--risk-pct", "-r", min=0.01, max=10.0,
                                       help="Risk % per trade"),
        min_score: float = typer.Option(0.65, "--min-score",
                                        help="Minimum confluence score"),
        max_daily: int = typer.Option(1, "--max-daily", help="Max trades per day"),
        mark_to_market: bool = typer.Option(False, "--mark-to-market",
                                            help="Mark open positions at end"),
        output_json: bool = typer.Option(False, "--output-json", "-j",
                                         help="Output as JSON"),
        outdir: Optional[Path] = typer.Option(Path("output"), "--outdir", "-o",
                                              help="Output directory"), ):
    """Backward compatibility - roept intern 'run' aan."""
    _run_confluence(symbol=symbol, csv=csv, start=start, end=end, server_tz=server_tz,
        session_start=session_start, session_end=session_end, risk_pct=risk_pct,
        min_score=min_score, max_daily=max_daily, mark_to_market=mark_to_market,
        output_json=output_json, outdir=outdir, )


# ---- Geen callback meer - alleen expliciete commands ----
def main():
    # Check if no arguments given - show help
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    # Als alleen een CSV pad gegeven wordt, voeg 'run' command toe
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        if sys.argv[1].endswith(".csv"):
            # User gaf direct een CSV - help ze door 'run --csv' te maken
            sys.argv.insert(1, "run")
            sys.argv.insert(2, "--csv")

    app()


if __name__ == "__main__":
    main()