# live/live_confluence.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from live.mt5_feed import MT5FeedCfg, MT5LiveM1

from backtest.mtf_exec_fast import (
    backtest_mtf_confluence_fast as backtest_mtf_confluence,
    MTFExecCfg,
 )
from strategies.mtf_confluence import MTFParams

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("live_confluence")

@dataclass(frozen=True)
class LiveRunCfg:
    symbol: str = "GER40.cash"
    server_tz: str = "Europe/Athens"     # FTMO
    out_tz: str | None = None            # bv. "Europe/Berlin" of None = server
    session_start: str = "09:00"
    session_end: str = "17:00"
    lookback_minutes: int = 60 * 48
    poll_interval_sec: int = 5

def run_live_confluence(lcfg: LiveRunCfg) -> None:
    feed = MT5LiveM1(MT5FeedCfg(
        symbol=lcfg.symbol,
        lookback_minutes=lcfg.lookback_minutes,
        server_tz=lcfg.server_tz,
        out_tz=lcfg.out_tz,
        poll_interval_sec=lcfg.poll_interval_sec
    ))
    df = feed.bootstrap()
    log.info("Bootstrapped %d bars, tz=%s", len(df), df.index.tz)

    params = MTFParams()
    exec_cfg = MTFExecCfg()

    last_seen = None
    while True:
        df, new_bar = feed.poll()
        if new_bar is None or (last_seen is not None and new_bar <= last_seen):
            continue
        last_seen = new_bar

        # Gebruik je bestaande backtester voor 100% dezelfde logica (zonder orders)
        # → window = laatste n minuten (bijv. laatste dag) om sneller te rekenen
        window = df.last("3D") if len(df) > 0 else df
        eq, trades, metrics = backtest_mtf_confluence(
            window, lcfg.symbol, params, exec_cfg,
            session_start=lcfg.session_start, session_end=lcfg.session_end,
            max_trades_per_day=1
        )

        # Toon laatste event
        ts = window.index[-1]
        log.info("Bar %s | trades=%s | pnl=%s | eq_end=%s | sharpe=%s",
                 ts, metrics.get("trades"), metrics.get("pnl"),
                 metrics.get("equity_end"), metrics.get("sharpe_proxy"))

        # Laatste potentiële entry (debug)
        # NB: omdat we filter_best_daily_signal toepassen in backtest, komt er max 1/dag door.
        if len(trades) > 0 and trades.iloc[-1]["exit_time"] == ts:
            log.info("Exit @ %s reason=%s pnl=%.2f", ts, trades.iloc[-1]["reason"], trades.iloc[-1]["pnl"])
