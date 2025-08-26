from __future__ import annotations
import pandas as pd
from typing import Optional

from backtest.data_loader import fetch_data
from strategies.order_block import order_block_signals


def run_backtest(
    *,
    strategy_name: str = "order_block_simple",
    params: Optional[dict] = None,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv: Optional[str] = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
):
    """
    Backtest runner die werkt met cli.main backtest.
    Laadt zelf data via data_loader.fetch_data of CSV.
    """

    # Data laden
    df = fetch_data(symbol=symbol, timeframe=timeframe, start=start, end=end, csv=csv)
    if df is None or "close" not in df.columns:
        raise ValueError("Kon geen geldige data laden, 'close' kolom ontbreekt")

    # Strategie-signalen
    if strategy_name == "order_block_simple":
        lookback = params.get("lookback_bos", 3) if params else 3
        signals = order_block_signals(df, swing_w=lookback)
    else:
        raise ValueError(f"Onbekende strategie: {strategy_name}")

    # --- Simpele backtest engine ---
    price = df["close"].astype(float)
    pos = signals["long"].astype(int).reindex(price.index).fillna(0)

    ptc = (fee_bps + slippage_bps) / 10_000.0
    cash = initial_cash
    units = 0.0
    trades = []
    equity = []

    delta = pos.diff().fillna(pos.iloc[0])
    entries = price.index[(delta == 1)]
    exits = price.index[(delta == -1)]

    if pos.iloc[0] == 1 and (len(entries) == 0 or entries[0] != price.index[0]):
        entries = entries.insert(0, price.index[0])
    if pos.iloc[-1] == 1 and (len(exits) == 0 or exits[-1] != price.index[-1]):
        exits = exits.append(pd.Index([price.index[-1]]))

    for ent, ex in zip(entries, exits):
        ent_px = price.loc[ent] * (1 + ptc)
        if units == 0.0:
            units = cash / ent_px
            cash -= units * ent_px
            trades.append({"entry_time": ent, "entry_px": float(ent_px)})

        for ts in price.loc[ent:ex].index:
            equity.append((ts, float(cash + units * price.loc[ts])))

        ex_px = price.loc[ex] * (1 - ptc)
        cash += units * ex_px
        trades[-1].update(
            {"exit_time": ex, "exit_px": float(ex_px), "pnl": float(cash - initial_cash)}
        )
        units = 0.0

    if not equity:
        for ts in price.index:
            equity.append((ts, float(cash)))
    else:
        seen = set(ts for ts, _ in equity)
        for ts in price.index:
            if ts not in seen:
                equity.append((ts, float(cash)))

    df_eq = (
        pd.DataFrame(equity, columns=["time", "equity"])
        .drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )
    trades_df = pd.DataFrame(trades)

    return df_eq, trades_df
