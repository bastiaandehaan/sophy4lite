# utils/specs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None  # MT5 optioneel; auto-probe werkt alleen als MT5 beschikbaar is

@dataclass(frozen=True)
class SymbolSpec:
    name: str
    point_value: float = 1.0   # PnL per 1.0 prijs-punt per 1.00 lot
    min_step: float = 0.01     # kleinste prijssprong (tick size)
    lot_step: float = 0.01     # lot afronding bij broker

# VERVANGEN met jouw gemeten waarden
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    # FTMO / GER40.cash (jouw output): tick_size=0.01, tick_value≈0.011638 -> pv≈1.16379
    "GER40.cash": SymbolSpec("GER40.cash", point_value=1.16379, min_step=0.01, lot_step=0.01),
    # Alias: als iemand per ongeluk 'GER40' gebruikt, map naar .cash specs
    "GER40":      SymbolSpec("GER40",      point_value=1.16379, min_step=0.01, lot_step=0.01),

    # XAUUSD (jouw output): tick_size=0.01, tick_value=1.0 -> pv=100.0, contract_size=100
    "XAUUSD":     SymbolSpec("XAUUSD",     point_value=100.0,   min_step=0.01, lot_step=0.01),
}

def _auto_probe_mt5(symbol: str) -> Optional[SymbolSpec]:
    """Lees MT5 symbol specification en maak er een SymbolSpec van."""
    if mt5 is None:
        return None
    if not mt5.initialize():
        return None
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    tick_size  = getattr(si, "trade_tick_size", si.point)
    tick_value = getattr(si, "trade_tick_value", 0.0)
    lot_step   = si.volume_step
    pv = (tick_value / tick_size) if tick_size else 0.0
    return SymbolSpec(name=symbol, point_value=pv, min_step=tick_size, lot_step=lot_step)

def get_spec(symbol: str) -> SymbolSpec:
    """Spec uit defaults of live uit MT5 (fallback)."""
    if symbol in DEFAULT_SPECS:
        return DEFAULT_SPECS[symbol]
    base = symbol.split(".")[0]
    if base in DEFAULT_SPECS:
        return DEFAULT_SPECS[base]
    live = _auto_probe_mt5(symbol)
    if live:
        return live
    return SymbolSpec(name=symbol)
