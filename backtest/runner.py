from backtest.data_loader import fetch_data
from backtest.breakout_exec import backtest_breakout, BTExecCfg
from strategies.breakout_signals import BreakoutParams, DEFAULT_SPECS
from utils.metrics import summarize_equity_metrics
from strategies.order_block import order_block_signals  # Voor backward compatibiliteit met order_block

def run_backtest(
    strategy_name: str,
    params: dict,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    csv=None,  # csv path
):
    df = fetch_data(csv_path=csv, symbol=symbol, timeframe=timeframe, start=start, end=end)

    if strategy_name == "breakout":
        p = BreakoutParams(
            broker_tz="UTC",
            session_start_h=7,
            session_end_h=8,
            cancel_at_h=17,
            min_range_pts=params.get("min_range", 10),
            vol_percentile=params.get("vol_pctl", 40),
            vol_lookback=params.get("vol_lookback", 60),
            confirm_bars=params.get("confirm", 2),
            mode=params.get("mode", "close_confirm"),
            atr_period=params.get("atr_period", 14),
            atr_sl_mult=params.get("atr_sl", 1.0),
            atr_tp_mult=params.get("atr_tp", 1.8),
            offset_pts=0.0,
            spread_pts=0.0,
        )
        # Fix specs: gebruik DEFAULT_SPECS direct, geen split nodig tenzij symbol zoals "DE40.cash"
        specs = DEFAULT_SPECS.copy()
        if symbol in specs or symbol.split(".")[0] in specs:
            base_symbol = symbol.split(".")[0]
            specs[symbol] = specs.get(base_symbol, DEFAULT_SPECS["DE40"])
        cfg = BTExecCfg(
            equity0=20_000.0,
            fee_rate=params.get("fee_bps", 2) / 10000.0,  # Fix: /10000 voor bps naar fractie
            slippage_pts=0.5,
            specs=specs,
            risk_frac=params.get("risk_pct", 1.0) / 100.0,  # Fix: /100 voor pct naar fractie
        )

        equity, trades, metrics = backtest_breakout(df, symbol, p, cfg)
        return equity.to_frame("equity"), trades, metrics

    elif strategy_name == "order_block_simple":
        # Backward compatibiliteit: gebruik simpele order_block logic
        lookback = params.get("lookback_bos", 3)
        signals = order_block_signals(df, swing_w=lookback)
        # ... (behoud de rest van de simpele backtest engine uit je vorige versie, maar gebruik summarize_equity_metrics)
        # Voor nu: return een placeholder; pas aan met de engine uit eerdere code
        raise NotImplementedError("Order block native engine nog te implementeren; gebruik VBT voor nu.")
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")