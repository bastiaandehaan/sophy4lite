from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from backtest.data_loader import fetch_data
from strategies.order_block import order_block_signals  # jouw signaal-functie
from risk.ftmo import pretrade_checks


def run_backtest(
    *,
    strategy_name: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Eenvoudige backtest-engine met SL/TP.

    Returns:
        df_eq: DataFrame(index=time, columns=['equity'])
        df_trades: DataFrame met trade-records (time, side, entry, sl, tp, exit, pnl)
    """
    # ----- data -----
    df = fetch_data(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        csv_path=csv_path,
    )

    # ----- signalen -----
    if strategy_name == "order_block_simple":
        entries, exits, sl_series, tp_series = order_block_signals(df, **params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # sanity
    if not isinstance(entries, pd.Series) or not isinstance(exits, pd.Series):
        raise TypeError("entries/exits must be pd.Series aligned to df.index")

    entries = entries.reindex(df.index, fill_value=False)
    exits = exits.reindex(df.index, fill_value=False)
    sl_series = sl_series.reindex(df.index)
    tp_series = tp_series.reindex(df.index)

    # ----- risk checks (placeholder) -----
    if not pretrade_checks(balance=10_000.0):
        # in deze simpele engine blokkeren we niets, maar hook is aanwezig
        pass

    # ----- backtest loop -----
    cash = 10_000.0
    position = 0  # 0/1 long-only voor nu
    entry_price = None
    sl = None
    tp = None

    equity = [cash]
    eq_index = [df.index[0]]
    trades = []

    for ts, row in df.iterrows():
        price = float(row["close"])

        # exit-voorwaarden: signaal, SL of TP
        if position == 1:
            # SL/TP check
            hit_sl = sl is not None and row["low"] <= sl
            hit_tp = tp is not None and row["high"] >= tp
            signal_exit = exits.loc[ts]

            if hit_sl or hit_tp or signal_exit:
                pnl = (price - entry_price) if not hit_sl and not hit_tp else \
                      ((sl - entry_price) if hit_sl else (tp - entry_price))
                cash += pnl
                trades.append(
                    {
                        "time": ts,
                        "side": "LONG",
                        "entry": entry_price,
                        "sl": sl,
                        "tp": tp,
                        "exit": (sl if hit_sl else (tp if hit_tp else price)),
                        "pnl": pnl,
                    }
                )
                position = 0
                entry_price = sl = tp = None

        # entry-voorwaarde
        if position == 0 and entries.loc[ts]:
            position = 1
            entry_price = price
            # per-candle SL/TP (kunnen NaN zijn als je dat wilt)
            sl = float(sl_series.loc[ts]) if pd.notna(sl_series.loc[ts]) else None
            tp = float(tp_series.loc[ts]) if pd.notna(tp_series.loc[ts]) else None

        equity.append(cash if position == 0 else cash + (price - entry_price))
        eq_index.append(ts)

    df_eq = pd.DataFrame(index=pd.Index(eq_index, name="time"), data={"equity": equity})
    df_trades = pd.DataFrame(trades)
    return df_eq, df_trades
