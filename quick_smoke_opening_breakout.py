# quick_smoke_opening_breakout.py
import pandas as pd
from backtest.breakout_exec import backtest_breakout, BTExecCfg
from strategies.breakout_params import BreakoutParams

# Data: 15m, 200 bars vanaf 2024-01-01 00:00
dates = pd.date_range('2024-01-01', periods=200, freq='15min')
df = pd.DataFrame({
    'open': 100.0,
    'high': 100.2,
    'low' :  99.8,
    'close': 100.0,
}, index=dates)

# Forceer breakout op dag 2 09:00 (bar ~36)
df.loc['2024-01-02 09:00', 'close'] = 101.0   # > prev day high (100.2)
df.loc['2024-01-02 09:00', 'high']  = 101.5

cfg = BTExecCfg(equity0=10_000, fee_rate=0.0002)
params = BreakoutParams(atr_mult_sl=1.0, atr_mult_tp=2.0)

# Belangrijk: open_window_bars groot genoeg zodat 09:00 in-venster valt
eq, trades, metrics = backtest_breakout(df, "TEST", params, cfg,
                                        open_window_bars=40,  # <--
                                        confirm="close")

print(f"Trades: {metrics['n_trades']}")
print(f"Final equity: {metrics['final_equity']:.2f}")
assert metrics['n_trades'] >= 1, "Er is geen opening-breakout trade geplaatst"
