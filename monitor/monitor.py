from config import logger
import vectorbt as vbt

def summarize(pf: vbt.Portfolio) -> dict:
    s = {
        "return": float(pf.total_return() or 0.0),
        "mdd": float(pf.max_drawdown() or 0.0),
        "sharpe": float(pf.sharpe_ratio() or 0.0),
        "trades": len(pf.trades.records) if hasattr(pf.trades, "records") else 0,
        "final_value": float(pf.value().iloc[-1]) if len(pf.value()) else 0.0
    }
    logger.info(f"Summary: {s}")
    return s
