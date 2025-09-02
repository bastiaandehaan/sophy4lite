# strategies/__init__.py
"""
Sophy4Lite strategies package.

Exports:
- Breakout signals (long, opening breakout)
- Breakout parameters & helpers (ATR, levels, specs)
"""

from .breakout_params import (BreakoutParams, SymbolSpec, DEFAULT_SPECS, daily_levels, )
from .breakout_signals import (breakout_long, opening_breakout_long, )

__all__ = [
    "breakout_long",
    "opening_breakout_long",
    "BreakoutParams",
    "SymbolSpec",
    "DEFAULT_SPECS",
    "daily_levels",
]
