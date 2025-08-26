from __future__ import annotations
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple

from utils.metrics import summarize_equity_metrics


def run_backtest_vbt(
    strategy_name: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run backtest using vectorbt.
    Returns (equity_df, trades_df, metrics_dict).
    """

    if csv_path is None:
        raise ValueError("csv_path is required for vectorbt runner")

    # Load CSV (expected columns: time, open, high, low, close, volume)
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    df = df.loc[start:end]

    close = df["close"]

    if strategy_name == "order_block_simple":
        # ⚠️ Voor nu placeholder: simpele MA cross om pipeline te testen
        fast = close.vbt.rolling_mean(window=params.get("fast", 10))
        slow = close.vbt.rolling_mean(window=params.get("slow", 20))
        entries = fast > slow
        exits = fast < slow
    else:
        raise ValueError(f"Strategy not implemented in VBT runner: {strategy_name}")

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=10_000,
        fees=0.0002,
        slippage=0.0001,
    )

    equity = pf.value()
    trades = pf.trades.records_readable
    metrics = summarize_equity_metrics(pd.DataFrame({"equity": equity}), trades)

    return pd.DataFrame({"equity": equity}), trades, metrics
