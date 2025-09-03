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
    Position sizing with validation and conservative rounding.
    Returns size in lots.
    """
    if equity <= 0:
        raise ValueError(f"Invalid equity: {equity}")
    if value_per_point <= 0:
        raise ValueError(f"Invalid value_per_point: {value_per_point}")
    if risk_frac <= 0 or risk_frac > 0.10:
        raise ValueError(f"Invalid risk_frac: {risk_frac} (0 < risk_frac <= 0.10)")

    pts = abs(entry - stop)
    if pts <= 0:
        return 0.0

    risk_cash = equity * risk_frac
    raw = risk_cash / (pts * value_per_point)

    if lot_step > 0:
        # round DOWN to broker step
        rounded = (raw // lot_step) * lot_step
        result = float(max(0.0, rounded))
    else:
        result = float(max(0.0, raw))

    # optional debug logging (uncomment if needed)
    # if result > 0:
    #     actual_risk = result * pts * value_per_point
    #     print(f"Sized {result:.2f} lots (risk ${actual_risk:.2f})")

    return result
