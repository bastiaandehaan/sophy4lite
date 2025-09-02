# scripts/export_mt5_m1.py
from __future__ import annotations

import os
import time

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "GER40.cash"
BARS = 200_000             # target bars via copy_from_pos
SERVER_TZ = "Europe/Athens"
RANGE_DAYS_FALLBACK = 30   # fallback window if pos-call returns nothing
OUTDIR = "data"


def _ensure_symbol(symbol: str) -> None:
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"Symbol not found: {symbol}")
    if not si.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Cannot select symbol: {symbol}")


def _rates_to_df(rates) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
    df = pd.DataFrame(rates)
    if "time" not in df.columns:
        return pd.DataFrame()  # robust guard
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df


def _copy_pos(symbol: str, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
    return _rates_to_df(rates)


def _copy_range(symbol: str, days: int) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().to_pydatetime()
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).to_pydatetime()
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    return _rates_to_df(rates)


def main():
    if not mt5.initialize():
        err = mt5.last_error()
        raise RuntimeError(f"MT5 init failed: {err}")

    _ensure_symbol(SYMBOL)

    # Try 'from_pos' first
    df = _copy_pos(SYMBOL, BARS)
    if df.empty:
        # Let MT5 warm up its cache, then retry
        time.sleep(2)
        df = _copy_pos(SYMBOL, BARS)

    # Fallback to range if still empty
    if df.empty:
        df = _copy_range(SYMBOL, RANGE_DAYS_FALLBACK)

    if df.empty:
        raise RuntimeError(
            "No M1 data returned from MT5. Fixes:\n"
            "- Open a GER40.cash M1 chart and scroll back (Home) to force history download\n"
            "- Use History Center (F2) to download M1\n"
            "- Ensure the correct MT5 terminal/account is running and logged in"
        )

    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Server TZ as the single source of truth
    df = df.tz_convert(SERVER_TZ)

    os.makedirs(OUTDIR, exist_ok=True)
    out_parquet = os.path.join(OUTDIR, f"{SYMBOL}_M1.parquet")
    out_csv = os.path.join(OUTDIR, f"{SYMBOL}_M1.csv")

    df.to_parquet(out_parquet)
    # CSV without tz for simplicity; we localize on load
    df.tz_convert(None).to_csv(out_csv, index_label="time")

    print("Saved:")
    print(" -", out_parquet)
    print(" -", out_csv)
    print(df.tail())


if __name__ == "__main__":
    main()
