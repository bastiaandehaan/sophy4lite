from utils.data_health import health_line
from utils.days import summarize_day_health

logger.info(health_line(df, expected_freq="15T"))
days, min_bars, mean_bars = summarize_day_health(df)
logger.info(f"DAYS { { 'count': days, 'min_bars': min_bars, 'mean_bars': round(mean_bars,1) } }")
