from backtest.data_loader import fetch_data
from strategies.breakout_signals import generate_breakout_signals
from strategies.order_block_signals import order_block_signals
from utils.metrics import summarize_equity_metrics

def run_backtest(
    strategy_name: str,
    params: dict,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv,
):
    df = fetch_data(csv_path=csv, symbol=symbol, timeframe=timeframe, start=start, end=end)

    # ===== Select Strategy =====
    if strategy_name == "breakout":
        signals = generate_breakout_signals(
            df,
            session_start=7,
            session_end=8,
            cancel_at=17,
            mode="close_confirm",
            confirm=2,
            min_range=10,
            vol_pctl=40,
            vol_lookback=60,
            atr_period=14,
            atr_sl=1.0,
            atr_tp=1.8,
            risk_pct=0.01,
            fee_bps=2,
            vpp=1.0,
        )
        # Je breakout-signals-functie geeft meestal een list[dict] met trades.
        # Convert eventueel naar DataFrame met entries/exits.
        trades = signals  # of pd.DataFrame(signals)
        # Genereer equity curve & metrics
        equity = ...  # vul in zoals bij breakout_exec.py
        metrics = summarize_equity_metrics(equity["equity"])
        return equity, trades, metrics

    elif strategy_name == "order_block":
        entries, exits = order_block_signals(df, swing_w=params.get("lookback_bos", 20))
        # Gebruik je eigen runner logic, bijv. vbt of native sim
        trades = ...  # logica voor order block trades
        equity = ...  # equity curve genereren
        metrics = summarize_equity_metrics(equity["equity"])
        return equity, trades, metrics

    else:
        raise ValueError(f"Unknown strategy {strategy_name}")
