import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple
from backtest.data_loader import fetch_data
from strategies.order_block import order_block_signals
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
    df = fetch_data(csv_path=csv_path, symbol=symbol, timeframe=timeframe, start=start, end=end)

    if strategy_name == "order_block":
        signals = order_block_signals(df, swing_w=params.get("lookback_bos", 3))
        entries = signals["long"]
        exits = ~signals["long"]  # Simpele exit op ~long
    else:
        raise ValueError(f"Strategy not implemented in VBT runner: {strategy_name}")

    pf = vbt.Portfolio.from_signals(
        df["close"],
        entries,
        exits,
        init_cash=20_000,
        fees=0.0002,
        slippage=0.0001,
    )

    equity = pf.value().to_frame("equity")
    trades = pf.trades.records_readable
    # Voeg pnl_cash toe voor metrics compatibiliteit als nodig
    if not trades.empty:
        trades["pnl_cash"] = trades["PnL"]
    metrics = summarize_equity_metrics(equity, trades)

    return equity, trades, metrics