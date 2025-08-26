from __future__ import annotations


def pretrade_checks(balance: float) -> bool:
    """Placeholder FTMO risk checks.

    In backtest we don't yet track intraday PnL.
    Always returns True for now.

    Later uitbreiden met:
    - daily loss limit
    - total loss limit
    - max position size checks
    """
    return True
