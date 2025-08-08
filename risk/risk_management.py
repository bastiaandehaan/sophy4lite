from dataclasses import dataclass
from typing import Dict, Any
from config import logger, RISK_PER_TRADE

@dataclass
class PositionSizingResult:
    position_size: float
    risk_amount: float
    warnings: list[str]

def fixed_percent_sizing(equity: float, entry: float, stop: float, contract_size: float=1.0, min_volume: float=0.01, max_volume: float=100.0) -> PositionSizingResult:
    warnings: list[str] = []
    if entry <= 0 or stop <= 0: 
        warnings.append("Invalid prices")
        return PositionSizingResult(0.0, 0.0, warnings)
    risk_amount = equity * RISK_PER_TRADE
    price_risk = abs(entry - stop)
    if price_risk <= 0:
        warnings.append("Zero price risk")
        return PositionSizingResult(0.0, 0.0, warnings)
    size = risk_amount / (price_risk * contract_size)
    size = max(min_volume, min(size, max_volume))
    return PositionSizingResult(round(size, 2), risk_amount, warnings)
