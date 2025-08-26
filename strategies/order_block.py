from __future__ import annotations
import numpy as np
import pandas as pd


def _body_pct(row: pd.Series) -> float:
    rng = (row["high"] - row["low"]) or 1e-12
    return abs(row["close"] - row["open"]) / rng


def order_block_signals(
    df: pd.DataFrame,
    lookback_bos: int = 20,
    min_body_pct: float = 0.55,
    rr: float = 1.5,
    stop_buffer_pct: float = 0.0005,
    max_concurrent: int = 1,
):
    """Simplified Order Block (OB) heuristic.

    Returns:
        entries: pd.Series(bool)
        exits: pd.Series(bool)
        sl: pd.Series(float or NaN)
        tp: pd.Series(float or NaN)
    """
    out = df.copy()
    out["bos_up"] = out["close"] > out["high"].rolling(lookback_bos).max().shift(1)
    out["bos_dn"] = out["close"] < out["low"].rolling(lookback_bos).min().shift(1)

    body = out.apply(_body_pct, axis=1)
    is_bear = out["close"] < out["open"]
    is_bull = ~is_bear

    out["bull_ob"] = (is_bear & body.ge(min_body_pct)).astype(int)
    out["bear_ob"] = (is_bull & body.ge(min_body_pct)).astype(int)

    last_bear_idx = None
    last_bull_idx = None

    entries = pd.Series(False, index=out.index)
    exits = pd.Series(False, index=out.index)
    sl = pd.Series(np.nan, index=out.index)
    tp = pd.Series(np.nan, index=out.index)

    open_trades = 0

    for ts, row in out.iterrows():
        if row["bull_ob"]:
            last_bear_idx = ts
        if row["bear_ob"]:
            last_bull_idx = ts

        # BOS up → bullish OB
        if row["bos_up"] and last_bear_idx is not None and open_trades < max_concurrent:
            ob = out.loc[last_bear_idx]
            zone_low = min(ob["open"], ob["close"]) - stop_buffer_pct
            zone_high = max(ob["open"], ob["close"]) + stop_buffer_pct
            stop = ob["low"] - stop_buffer_pct
            if row["low"] <= zone_high and row["high"] >= zone_low and row["open"] > zone_high:
                entries.at[ts] = True
                risk = zone_high - stop
                sl.at[ts] = stop
                tp.at[ts] = zone_high + rr * risk
                open_trades += 1

        # BOS down → bearish OB
        if row["bos_dn"] and last_bull_idx is not None and open_trades < max_concurrent:
            ob = out.loc[last_bull_idx]
            zone_low = min(ob["open"], ob["close"]) - stop_buffer_pct
            zone_high = max(ob["open"], ob["close"]) + stop_buffer_pct
            stop = ob["high"] + stop_buffer_pct
            if row["low"] <= zone_high and row["high"] >= zone_low and row["open"] < zone_low:
                entries.at[ts] = True
                risk = stop - zone_low
                sl.at[ts] = stop
                tp.at[ts] = zone_low - rr * risk
                open_trades += 1

    return entries, exits, sl, tp
