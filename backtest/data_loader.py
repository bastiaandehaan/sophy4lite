from datetime import datetime
from typing import Optional
import pandas as pd
import MetaTrader5 as mt5


def fetch_data(symbol: str, timeframe: str, days: int = 365, start: Optional[str] = None) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed")

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }

    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if start:
        start_time = datetime.strptime(start, "%Y-%m-%d")
        rates = mt5.copy_rates_from(symbol, timeframe_map[timeframe], start_time, days * 24)
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, days * 24)

    if rates is None or len(rates) == 0:
        raise ValueError("Failed to fetch data from MT5")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df
