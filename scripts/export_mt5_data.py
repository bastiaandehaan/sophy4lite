import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-01-01")

if not mt5.initialize():
    raise RuntimeError(f"MT5 init failed, error code: {mt5.last_error()}")

rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, START.to_pydatetime(), END.to_pydatetime())
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

outpath = Path("data") / f"{SYMBOL}_H1.csv"
outpath.parent.mkdir(exist_ok=True)
df.to_csv(outpath, index=False)

print(f"Wrote {len(df)} rows to {outpath}")

mt5.shutdown()
