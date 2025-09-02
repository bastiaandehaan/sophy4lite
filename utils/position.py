# utils/position.py
from __future__ import annotations

def _size(
    equity: float,
    entry: float,
    stop: float,
    value_per_point: float,
    risk_frac: float,
    lot_step: float = 0.01,
) -> float:
    """
    Fixed-% risk position sizing with broker lot rounding.

    equity          : current equity (cash)
    entry, stop     : entry and stop price
    value_per_point : PnL per 1.0 price point per 1.00 lot
    risk_frac       : risk as fraction of equity (e.g., 0.01 for 1%)
    lot_step        : broker lot step (rounding down)
    """
    if equity <= 0 or value_per_point <= 0 or risk_frac <= 0:
        return 0.0
    pts = abs(entry - stop)
    if pts <= 0:
        return 0.0

    risk_cash = equity * risk_frac
    raw = risk_cash / (pts * value_per_point)

    if lot_step > 0:
        # round down to nearest lot_step to avoid order rejections
        rounded = (raw // lot_step) * lot_step
        return float(max(0.0, rounded))
    return float(max(0.0, raw))
