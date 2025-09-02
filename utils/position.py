"""
Position sizing and trade utilities for Sophy4Lite.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class Side(Enum):
    """Trade side enumeration."""
    BUY = 1
    SELL = -1


def side_factor(side: Side) -> float:
    """Return 1.0 for BUY, -1.0 for SELL."""
    return float(side.value)


def side_name(side: Side) -> str:
    """Return 'BUY' or 'SELL' string."""
    return side.name


class Position(NamedTuple):
    """Position information."""
    symbol: str
    side: Side
    size: float
    entry_price: float
    sl_price: float
    tp_price: float

    @property
    def risk_points(self) -> float:
        """Distance to stop loss in points."""
        return abs(self.entry_price - self.sl_price)

    @property
    def reward_points(self) -> float:
        """Distance to take profit in points."""
        return abs(self.tp_price - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        """R:R ratio of the position."""
        if self.risk_points == 0:
            return 0.0
        return self.reward_points / self.risk_points


def _size(equity: float, entry: float, stop: float, vpp: float,
          risk_frac: float) -> float:
    """
    Calculate position size based on fixed percentage risk.

    Args:
        equity: Current account equity
        entry: Entry price
        stop: Stop loss price
        vpp: Value per point
        risk_frac: Risk fraction (e.g., 0.01 for 1%)

    Returns:
        Position size in lots/contracts
    """
    risk_cash = equity * risk_frac
    pts = abs(entry - stop)
    if pts <= 0 or vpp <= 0:
        return 0.0
    return float(risk_cash / (pts * vpp))