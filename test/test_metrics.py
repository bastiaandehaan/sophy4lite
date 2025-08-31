# test/test_metrics.py
import pandas as pd
from utils.metrics import summarize_equity_metrics

def test_metrics_basic():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df_eq = pd.DataFrame(index=idx, data={"equity": [10000 + i*10 for i in range(10)]})
    trades = pd.DataFrame({"pnl": [1, -2, 3]})
    m = summarize_equity_metrics(df_eq, trades)
    # verwachtte kern-keys moeten aanwezig zijn (er mogen extra keys zijn):
    expected = {"sharpe", "max_dd", "dd_duration", "total_return", "n_trades"}
    assert expected <= set(m.keys())
