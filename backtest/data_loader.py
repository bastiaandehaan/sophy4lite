from __future__ import annotations
import pandas as pd
from pathlib import Path

class DataError(Exception):
    pass

def fetch_data(symbol: str, timeframe: str, start: str, end: str, csv_path: Path | None) -> pd.DataFrame:
    """Load OHLCV data; for now, from a CSV with columns: time, open, high, low, close, volume.
    Index is datetime (UTC). Filters by [start, end].
    """
    if csv_path is None:
        raise DataError("csv_path is required for now. Provide --csv to CLI.")

    df = pd.read_csv(csv_path)
    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise DataError(f"CSV missing columns: {required - set(df.columns)}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df.loc[start:end]
    return df
```

---

## strategies/base.py
```python
from __future__ import annotations
from typing import Protocol, Dict, Any
import pandas as pd

class Strategy(Protocol):
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Return a DataFrame with columns: entry, exit, sl, tp (floats or NaN)."""
        ...
```