from typing import Optional, Dict, Any
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import logger, RISK_PER_TRADE, MAX_DAILY_LOSS, MAX_TOTAL_LOSS, STOP_AFTER_LOSSES
from risk.ftmo import fixed_percent_sizing

class DailyState:
    def __init__(self):
        self.date = None
        self.loss_streak = 0
        self.start_equity = None
        self.blocked = False

_state = DailyState()

def _reset_if_new_day():
    today = datetime.utcnow().date()
    if _state.date != today:
        _state.date = today
        _state.loss_streak = 0
        _state.blocked = False
        _state.start_equity = None

def _get_equity() -> float:
    info = mt5.account_info()
    return float(info.equity) if info else 0.0

def _guardrails_pass() -> bool:
    if _state.blocked:
        logger.warning("Daily trading blocked due to guardrails")
        return False
    eq = _get_equity()
    if _state.start_equity is None:
        _state.start_equity = eq
    day_dd = max(0.0, (_state.start_equity - eq) / max(_state.start_equity, 1e-9))
    if day_dd >= MAX_DAILY_LOSS:
        logger.error("Max daily loss reached -> blocking trading for today")
        _state.blocked = True
        return False
    return True

def place_trade(symbol: str, entry_price: float, sl_price: float, tp_price: float) -> Dict[str, Any]:
    if not mt5.initialize():
        return {"success": False, "message": f"MT5 init failed: {mt5.last_error()}"}
    _reset_if_new_day()
    if not _guardrails_pass():
        return {"success": False, "message": "Guardrails: daily blocked"}
    info = mt5.symbol_info(symbol)
    if info is None: return {"success": False, "message": f"Symbol {symbol} not found"}
    if not info.visible:
        mt5.symbol_select(symbol, True)
    eq = _get_equity()
    sizing = fixed_percent_sizing(eq, entry_price, sl_price, contract_size=info.trade_contract_size, min_volume=info.volume_min, max_volume=info.volume_max)
    if sizing.position_size <= 0:
        return {"success": False, "message": f"Sizing failed: {sizing.warnings}"}
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": sizing.position_size,
        "type": mt5.ORDER_TYPE_BUY if entry_price<=tp_price else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "magic": 424242,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "Sophy4Lite"
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.comment} code={result.retcode}")
        return {"success": False, "message": result.comment, "retcode": result.retcode}
    logger.info(f"Trade executed {symbol} vol={sizing.position_size} @ {entry_price} SL={sl_price} TP={tp_price}")
    return {"success": True, "ticket": result.order, "volume": sizing.position_size}

def register_trade_result(pnl: float):
    _reset_if_new_day()
    if pnl < 0:
        _state.loss_streak += 1
        if _state.loss_streak >= STOP_AFTER_LOSSES:
            _state.blocked = True
            logger.warning("Stop after consecutive losses reached -> block for today")
    else:
        _state.loss_streak = 0
