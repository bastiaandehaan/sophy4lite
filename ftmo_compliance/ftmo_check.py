from config import MAX_DAILY_LOSS, MAX_TOTAL_LOSS
def check_limits(day_return: float, max_drawdown: float) -> tuple[bool,bool]:
    daily_ok = abs(day_return) <= MAX_DAILY_LOSS
    total_ok = abs(max_drawdown) <= MAX_TOTAL_LOSS
    return daily_ok, total_ok
