# strategies/breakout_params.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass(frozen=True)
class SymbolSpec:
    """Contract-/symboolspecificaties voor sizing en validatie."""
    name: str
    point_value: float = 1.0     # contract point value / pip value
    min_step: float = 0.01       # minimale prijsstap

# Voeg hier eventuele extra symbolen toe wanneer nodig.
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    "GER40.cash": SymbolSpec("GER40.cash", point_value=1.0, min_step=0.5),
}

@dataclass
class BreakoutParams:
    """Parameters voor een eenvoudige breakout-strategie."""
    window: int = 20
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0

def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per dag simpele high/low-levels teruggeven.
    Vereist kolommen: ['high', 'low'] en DatetimeIndex (tz mag, bijv. UTC).
    Output: DataFrame met index=dag en kolommen=['day_high','day_low'].
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    if not {"high", "low"}.issubset(df.columns):
        raise KeyError("DataFrame must contain 'high' and 'low' columns")

    # Groepeer per kalenderdag op basis van de index (tz-aware OK).
    groups = df.groupby(df.index.normalize())
    levels = groups.agg(day_high=("high", "max"), day_low=("low", "min"))

    # Zorg voor stabiele dtypes (handig voor tests/downstream)
    return levels.astype({"day_high": "float64", "day_low": "float64"})

__all__ = [
    "SymbolSpec",
    "DEFAULT_SPECS",
    "BreakoutParams",
    "daily_levels",
]
