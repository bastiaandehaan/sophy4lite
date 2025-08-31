# backtest/runner.py
from __future__ import annotations
from backtest.data_loader import fetch_data
from backtest.breakout_exec import backtest_breakout, BTExecCfg
from strategies.breakout_params import BreakoutParams, DEFAULT_SPECS, SymbolSpec
from strategies.order_block import order_block_signals  # (placeholder voor simple OB)
from utils.metrics import summarize_equity_metrics

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
        # Voor nu simpele param-set (past bij demo / lite)
        p = BreakoutParams(
            window=int(params.get("window", 20)),
            atr_mult_sl=float(params.get("atr_sl", 2.0)),
            atr_mult_tp=float(params.get("atr_tp", 3.0)),
        )

        # Specs opzetten
        specs = dict(DEFAULT_SPECS)  # kopie
        if symbol not in specs:
            base = symbol.split(".")[0]  # bv. "GER40" uit "GER40.cash"
            if base in specs:
                specs[symbol] = specs[base]
            else:
                # veilige default
                specs[symbol] = SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)

        cfg = BTExecCfg(
            equity0=float(params.get("equity0", 20_000.0)),
            fee_rate=float(params.get("fee_bps", 2.0)) / 10000.0,  # bps → fractie
            slippage_pts=float(params.get("slippage_pts", 0.5)),
            specs=specs,
            risk_frac=float(params.get("risk_pct", 1.0)) / 100.0,  # pct → fractie
        )

        equity, trades, metrics = backtest_breakout(df, symbol, p, cfg)
        return equity.to_frame("equity"), trades, metrics

    elif strategy_name == "order_block_simple":
        lookback = int(params.get("lookback_bos", 3))
        signals = order_block_signals(df, swing_w=lookback)
        # (je kunt hier desgewenst een simpele from_signals-engine toevoegen)
        raise NotImplementedError("Order block native engine nog te implementeren; gebruik VBT voor nu.")

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
