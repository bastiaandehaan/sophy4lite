# scripts/export_mt5_orb.py
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import MetaTrader5 as mt5

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1,
}

def _backfill_until(symbol: str, tf: int, start: datetime, end: datetime,
                    chunk: int = 100_000, max_iters: int = 200) -> pd.DataFrame:
    """
    Trek data achteruit vanaf 'end' in blokken totdat 'start' is bereikt of geen data meer komt.
    """
    # "wakker maken"
    mt5.symbol_select(symbol, True)
    mt5.copy_rates_from_pos(symbol, tf, 0, 1)

    # eerste blok rond 'end'
    r = mt5.copy_rates_from(symbol, tf, end, chunk)
    if r is None or len(r) == 0:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # achteruit blijven trekken
    for _ in range(max_iters):
        tmin = df["time"].min().to_pydatetime().replace(tzinfo=timezone.utc)
        if tmin <= start:  # genoeg historie
            break
        r = mt5.copy_rates_from(symbol, tf, tmin - timedelta(seconds=1), chunk)
        if r is None or len(r) == 0:
            # kleine nudge en nog één poging
            mt5.copy_rates_from_pos(symbol, tf, 0, 1)
            r = mt5.copy_rates_from(symbol, tf, tmin - timedelta(seconds=1), chunk)
            if r is None or len(r) == 0:
                break
        tmp = pd.DataFrame(r)
        tmp["time"] = pd.to_datetime(tmp["time"], unit="s", utc=True)
        df = pd.concat([df, tmp], ignore_index=True).drop_duplicates("time").sort_values("time")

    # trimmen op [start, end] + volume kiezen
    df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))].copy()
    if "real_volume" in df.columns and df["real_volume"].notna().any():
        vol = df["real_volume"]
    else:
        vol = df["tick_volume"] if "tick_volume" in df.columns else 0

    out = pd.DataFrame({
        "time": df["time"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": vol,
    }).drop_duplicates("time").sort_values("time")
    return out

def main():
    if len(sys.argv) != 6:
        print("Usage: python scripts/export_mt5_orb.py SYMBOL TF START END OUTCSV", file=sys.stderr)
        sys.exit(2)
    symbol, tf_label, start_s, end_s, out = sys.argv[1:]
    tf = TF_MAP.get(tf_label)
    if tf is None:
        sys.exit(f"Unknown TF: {tf_label}")

    start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)

    if not mt5.initialize():
        sys.exit("mt5.initialize() failed")
    try:
        df = _backfill_until(symbol, tf, start, end)
    finally:
        mt5.shutdown()

    if df.empty:
        sys.exit(f"No data received for {symbol} {tf_label}")

    outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"[OK] {symbol} {tf_label} -> {outp} rows={len(df)} range={df['time'].min()} .. {df['time'].max()}")

if __name__ == "__main__":
    main()
