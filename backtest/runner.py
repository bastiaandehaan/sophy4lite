from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from backtest.data_loader import fetch_data
from strategies.order_block import order_block_signals
from risk.ftmo import pretrade_checks


def run_backtest(
    strategy_name: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv_path=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = fetch_data(symbol, timeframe, start, end, csv_path)

    if strategy_name == "order_block_simple":
        sig = order_block_signals(df, params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Simple SL/TP engine, 1 trade per signal timestamp
    equity = [1_0000.0]
    eq_index = []
    trades = []

    balance = equity[0]
    open_pos = None

    for ts, row in df.iterrows():
        price = row["close"]
        # new entry?
        if not np.isnan(sig.at[ts, "entry"]) and open_pos is None:
            entry = float(sig.at[ts, "entry"])
            sl = float(sig.at[ts, "sl"]) if not np.isnan(sig.at[ts, "sl"]) else None
            tp = float(sig.at[ts, "tp"]) if not np.isnan(sig.at[ts, "tp"]) else None

            # pre-trade FTMO/risk checks (simplified): max daily loss / total loss are not modeled fully
            if not pretrade_checks(balance):
                continue
            direction = 1.0 if entry <= price else -1.0
            risk_amt = 0.003 * balance  # 0.3% per trade default risk
            # position size = risk / (entry - sl) absolute
            if sl is not None and sl != entry:
                pos_size = abs(risk_amt / (entry - sl))
            else:
                pos_size = 0.0

            open_pos = {
                "ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "dir": direction,
                "size": pos_size,
            }

        # manage open position
        if open_pos is not None:
            hi, lo = row["high"], row["low"]
            hit_tp = open_pos["tp"] is not None and (
                (open_pos["dir"] > 0 and hi >= open_pos["tp"]) or (open_pos["dir"] < 0 and lo <= open_pos["tp"])
            )
            hit_sl = open_pos["sl"] is not None and (
                (open_pos["dir"] > 0 and lo <= open_pos["sl"]) or (open_pos["dir"] < 0 and hi >= open_pos["sl"])
            )

            exit_px = None
            outcome = None
            if hit_tp:
                exit_px = open_pos["tp"]
                outcome = "TP"
            elif hit_sl:
                exit_px = open_pos["sl"]
                outcome = "SL"

            if exit_px is not None:
                pnl = open_pos["dir"] * (exit_px - open_pos["entry"]) * open_pos["size"]
                balance += pnl
                trades.append({
                    "entry_ts": open_pos["ts"],
                    "exit_ts": ts,
                    "entry": open_pos["entry"],
                    "exit": exit_px,
                    "dir": open_pos["dir"],
                    "size": open_pos["size"],
                    "pnl": pnl,
                    "outcome": outcome,
                })
                open_pos = None

        equity.append(balance)
        eq_index.append(ts)

    df_eq = pd.DataFrame(index=pd.Index(eq_index, name="time"), data={"equity": equity[1:]})
    df_trades = pd.DataFrame(trades)
    return df_eq, df_trades
```

---

## risk/ftmo.py
```python
from __future__ import annotations

def pretrade_checks(balance: float) -> bool:
    # Placeholder: in backtest we don’t track intraday PnL yet. Always True for now.
    # Extend with daily loss / total loss simulation if you add timestamped PnL aggregation.
    return True