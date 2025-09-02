from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import typer


def export(
    symbol: str = typer.Option(..., help="Instrument, bv. GER40.cash of XAUUSD"),
    tf: str = typer.Option("M15", help="Timeframe: M1, M5, M15, H1, D1"),
    start: str = typer.Option(..., help="Startdatum (YYYY-MM-DD)"),
    end: str = typer.Option(..., help="Einddatum (YYYY-MM-DD)")
):
    """ Exporteer MT5 rates naar CSV in de project-root/data/ map """

    # Mapping van string → MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1,
    }
    if tf not in tf_map:
        raise ValueError(f"Unsupported timeframe {tf}. Kies uit {list(tf_map.keys())}")

    timeframe = tf_map[tf]
    start_ts = pd.Timestamp(start).to_pydatetime()
    end_ts = pd.Timestamp(end).to_pydatetime()

    # Init MT5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed, error code: {mt5.last_error()}")

    rates = mt5.copy_rates_range(symbol, timeframe, start_ts, end_ts)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Geen data ontvangen voor {symbol} {tf} {start}–{end}")

    df = pd.DataFrame(rates)
    df = df.rename(columns={
        "time": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Pad altijd relatief aan project-root
    ROOT = Path(__file__).resolve().parent.parent
    outpath = ROOT / "data" / f"{symbol}_{tf}.csv"
    outpath.parent.mkdir(exist_ok=True)
    df.to_csv(outpath, index=False)

    print(f"[OK] Wrote {len(df)} rows to {outpath}")

    mt5.shutdown()


if __name__ == "__main__":
    typer.run(export)
