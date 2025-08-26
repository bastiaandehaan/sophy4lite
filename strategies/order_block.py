from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

# Simplified Order Block (OB) heuristic:
# - Identify BOS (break of structure) when close crosses above rolling max (bullish) or below rolling min (bearish)
# - Bullish OB = last bearish candle before BOS up. Bearish OB = last bullish candle before BOS down.
# - Entry on retest into OB body zone; SL at OB extreme +/- buffer; TP = RR * risk.


def _body_pct(row: pd.Series) -> float:
    rng = (row["high"] - row["low"]) or 1e-12
    return abs(row["close"] - row["open"]) / rng


def order_block_signals(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    lookback = int(params.get("lookback_bos", 20))
    min_body_pct = float(params.get("min_body_pct", 0.55))
    rr = float(params.get("rr", 1.5))
    stop_buf = float(params.get("stop_buffer_pct", 0.0005))
    max_conc = int(params.get("max_concurrent", 1))

    out = df.copy()
    out["bos_up"] = out["close"] > out["high"].rolling(lookback).max().shift(1)
    out["bos_dn"] = out["close"] < out["low"].rolling(lookback).min().shift(1)

    # find candidate OB candles
    body = out.apply(_body_pct, axis=1)
    is_bear = out["close"] < out["open"]
    is_bull = ~is_bear

    # last red before BOS up; last green before BOS down
    out["bull_ob"] = (is_bear & body.ge(min_body_pct)).astype(int)
    out["bear_ob"] = (is_bull & body.ge(min_body_pct)).astype(int)

    # hold last candidate index until BOS
    last_bear_idx = None
    last_bull_idx = None

    entries = []
    exits = []
    sl_list = []
    tp_list = []

    open_trades = 0

    for ts, row in out.iterrows():
        # track last candidates
        if row["bull_ob"]:
            last_bear_idx = ts
        if row["bear_ob"]:
            last_bull_idx = ts

        # BOS up → define bullish OB from last bearish candle
        if row["bos_up"] and last_bear_idx is not None and open_trades < max_conc:
            ob = out.loc[last_bear_idx]
            zone_low = min(ob["open"], ob["close"]) - stop_buf
            zone_high = max(ob["open"], ob["close"]) + stop_buf
            sl = ob["low"] - stop_buf
            # Entry on retest into [zone_low, zone_high]
            if row["low"] <= zone_high and row["high"] >= zone_low and row["open"] > zone_high:
                entry = zone_high  # conservative: top of body zone touch
                risk = entry - sl
                tp = entry + rr * risk
                entries.append((ts, entry))
                sl_list.append((ts, sl))
                tp_list.append((ts, tp))
                exits.append((np.nan, np.nan))  # placeholder until backtest computes exit
                open_trades += 1

        # BOS down → define bearish OB from last bullish candle
        if row["bos_dn"] and last_bull_idx is not None and open_trades < max_conc:
            ob = out.loc[last_bull_idx]
            zone_low = min(ob["open"], ob["close"]) - stop_buf
            zone_high = max(ob["open"], ob["close"]) + stop_buf
            sl = ob["high"] + stop_buf
            if row["low"] <= zone_high and row["high"] >= zone_low and row["open"] < zone_low:
                entry = zone_low  # conservative: bottom of body zone touch
                risk = sl - entry
                tp = entry - rr * risk
                entries.append((ts, entry))
                sl_list.append((ts, sl))
                tp_list.append((ts, tp))
                exits.append((np.nan, np.nan))
                open_trades += 1

    # Build signals frame aligned to index
    sig = pd.DataFrame(index=out.index, data={"entry": np.nan, "sl": np.nan, "tp": np.nan})
    for ts, price in entries:
        sig.at[ts, "entry"] = price
    for ts, price in sl_list:
        sig.at[ts, "sl"] = price
    for ts, price in tp_list:
        sig.at[ts, "tp"] = price

    sig["exit"] = np.nan  # actual exits computed in runner via SL/TP touch
    return sig