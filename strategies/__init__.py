"""
Sophy4Lite strategies package.

Exports:
- Breakout signals (long/short)
- Breakout parameters & helpers (ATR, levels, specs)
"""

from .breakout_signals import breakout_long, breakout_short
from .breakout_params import (
    BreakoutParams,
    SymbolSpec,
    DEFAULT_SPECS,
    daily_levels,
)

__all__ = [
    "breakout_long",
    "breakout_short",
    "BreakoutParams",
    "SymbolSpec",
    "DEFAULT_SPECS",
    "daily_levels",
]
