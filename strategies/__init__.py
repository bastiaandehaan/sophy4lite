"""
Sophy4Lite strategies package.

Exports:
- Breakout (ATR-based R:R, sessie-range, FTMO-guards via backtester)
- Shared types/helpers for strategie-implementaties
"""

from .breakout_signals import (
    BreakoutParams,
    SymbolSpec,
    DEFAULT_SPECS,
    daily_levels,
    confirm_pass,
)

__all__ = [
    "BreakoutParams",
    "SymbolSpec",
    "DEFAULT_SPECS",
    "daily_levels",
    "confirm_pass",
]
