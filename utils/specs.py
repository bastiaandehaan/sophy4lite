# utils/specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import MetaTrader5 as mt5  # optional, only used for auto-probe
except Exception:
    mt5 = None


@dataclass(frozen=True)
class SymbolSpec:
    """
    Per-symbol trading specs.

    point_value : PnL per 1.0 price point per 1.00 lot
    min_step    : smallest price increment (tick size)
    lot_step    : broker volume (lot) step used for order size rounding
    """
    name: str
    point_value: float = 1.0
    min_step: float = 0.01
    lot_step: float = 0.01


# === Fill with YOUR broker values (FTMO demo output you posted) ===
# GER40.cash: tick_size=0.01, tick_value≈0.011638 => point_value≈1.16379 per indexpunt per 1 lot.
# XAUUSD    : tick_size=0.01, tick_value=1.0     => point_value=100.0    per $1 per 1 lot (contract 100).
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    "GER40.cash": SymbolSpec("GER40.cash", point_value=1.16379, min_step=0.01, lot_step=0.01),

}


def _auto_probe_mt5(symbol: str) -> Optional[SymbolSpec]:
    """
    Read symbol specification from a running MT5 terminal to synthesize a SymbolSpec.
    Returns None if MT5 is unavailable or the symbol can't be read.
    """
    if mt5 is None:
        return None
    if not mt5.initialize():
        return None
    si = mt5.symbol_info(symbol)
    if si is None:
        return None

    # MT5 fields:
    tick_size = getattr(si, "trade_tick_size", si.point)  # fallback to 'point'
    tick_value = getattr(si, "trade_tick_value", 0.0)
    lot_step = float(getattr(si, "volume_step", 0.01) or 0.01)

    # Value per 1.0 price point per 1 lot
    pv = (tick_value / tick_size) if tick_size else 0.0
    return SymbolSpec(name=symbol, point_value=float(pv), min_step=float(tick_size), lot_step=lot_step)


def get_spec(symbol: str) -> SymbolSpec:
    """
    Return a SymbolSpec for 'symbol'. Prefer DEFAULT_SPECS, else try auto-probing MT5,
    else return a conservative fallback.
    """
    if symbol in DEFAULT_SPECS:
        return DEFAULT_SPECS[symbol]
    base = symbol.split(".")[0]
    if base in DEFAULT_SPECS:
        return DEFAULT_SPECS[base]
    live = _auto_probe_mt5(symbol)
    if live:
        return live
    return SymbolSpec(name=symbol)


__all__ = ["SymbolSpec", "DEFAULT_SPECS", "get_spec"]
