# utils/specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import MetaTrader5 as mt5  # optional auto-probe
except Exception:
    mt5 = None

@dataclass(frozen=True)
class SymbolSpec:
    """Per-symbol trading specs.
    point_value : PnL per 1.0 price point per 1.00 lot
    min_step    : smallest price increment (tick size)
    lot_step    : broker volume step for size rounding
    """
    name: str
    point_value: float = 1.0
    min_step: float = 0.01
    lot_step: float = 0.01

# ← VUL AAN met jouw broker-waarden (FTMO)
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    # GER40.cash (FTMO demo): tick_size=0.01, tick_value≈0.011638 → pv≈1.16379
    "GER40.cash": SymbolSpec("GER40.cash", point_value=1.16379, min_step=0.01, lot_step=0.01),
    # Alias vangnet:
    "GER40":      SymbolSpec("GER40",      point_value=1.16379, min_step=0.01, lot_step=0.01),
    # XAUUSD: tick_size=0.01, tick_value=1.0 → pv=100.0 (contract 100)
    "XAUUSD":     SymbolSpec("XAUUSD",     point_value=100.0,   min_step=0.01, lot_step=0.01),
}

def _auto_probe_mt5(symbol: str) -> Optional[SymbolSpec]:
    """Try to read symbol spec from a running MT5 terminal."""
    if mt5 is None or not mt5.initialize():
        return None
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    tick_size  = float(getattr(si, "trade_tick_size", getattr(si, "point", 0.0)) or 0.0)
    tick_value = float(getattr(si, "trade_tick_value", 0.0))
    lot_step   = float(getattr(si, "volume_step", 0.01) or 0.01)
    pv = (tick_value / tick_size) if tick_size else 0.0
    return SymbolSpec(name=symbol, point_value=pv, min_step=tick_size, lot_step=lot_step)

def get_spec(symbol: str) -> SymbolSpec:
    """Prefer DEFAULT_SPECS, else MT5 auto-probe, else safe fallback."""
    if symbol in DEFAULT_SPECS:
        return DEFAULT_SPECS[symbol]
    base = symbol.split(".")[0]
    if base in DEFAULT_SPECS:
        return DEFAULT_SPECS[base]
    live = _auto_probe_mt5(symbol)
    return live if live else SymbolSpec(name=symbol)

__all__ = ["SymbolSpec", "DEFAULT_SPECS", "get_spec"]
