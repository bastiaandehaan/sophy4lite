# live/mt5_feed.py
"""
MT5 Live Feed for real-time M1 data streaming.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError("MetaTrader5 required for live feed: pip install MetaTrader5") from e

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MT5FeedCfg:
    """Configuration for MT5 live feed."""
    symbol: str = "GER40.cash"
    lookback_minutes: int = 60 * 24 * 2  # 2 days
    server_tz: str = "Europe/Athens"     # FTMO server timezone
    out_tz: Optional[str] = None         # Output timezone (None = keep server tz)
    poll_interval_sec: int = 5
    reconnect_attempts: int = 3


class MT5LiveM1:
    """
    Live M1 data feed from MT5.
    Maintains a rolling window of recent bars.
    """

    def __init__(self, cfg: MT5FeedCfg):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None
        self._connected = False

    def _connect(self) -> bool:
        """Initialize MT5 connection."""
        if self._connected:
            return True

        for attempt in range(self.cfg.reconnect_attempts):
            try:
                if mt5.initialize():
                    self._connected = True
                    log.info("MT5 connected (attempt %d)", attempt + 1)

                    # Ensure symbol is available
                    if not mt5.symbol_select(self.cfg.symbol, True):
                        log.error("Failed to select symbol %s", self.cfg.symbol)
                        return False
                    return True
            except Exception as e:
                log.warning("MT5 connection attempt %d failed: %s", attempt + 1, str(e))
                time.sleep(2)

        return False

    def _fetch_bars(self, num_bars: int) -> pd.DataFrame:
        """Fetch last N M1 bars from MT5."""
        if not self._connect():
            raise RuntimeError("MT5 connection failed")

        rates = mt5.copy_rates_from_pos(self.cfg.symbol, mt5.TIMEFRAME_M1, 0, num_bars)
        if rates is None or len(rates) == 0:
            log.warning("No data received for %s", self.cfg.symbol)
            return pd.DataFrame()

        df = pd.DataFrame(rates)

        # Convert time to datetime index (UTC -> server tz)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index = df.index.tz_convert(self.cfg.server_tz)

        # Map columns and keep OHLCV
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Optional output tz
        if self.cfg.out_tz:
            df.index = df.index.tz_convert(self.cfg.out_tz)

        return df

    def bootstrap(self) -> pd.DataFrame:
        """Initial load of historical data."""
        log.info("Bootstrapping %d minutes of %s", self.cfg.lookback_minutes, self.cfg.symbol)

        # Safety margin to cover weekends/market pauses
        max_bars = self.cfg.lookback_minutes * 2

        self.df = self._fetch_bars(max_bars)
        if len(self.df) > 0:
            cutoff = pd.Timestamp.now(tz=self.cfg.server_tz) - pd.Timedelta(minutes=self.cfg.lookback_minutes)
            self.df = self.df[self.df.index >= cutoff]
            self.last_bar_time = self.df.index[-1]
            log.info("Bootstrapped %d bars, last: %s", len(self.df), self.last_bar_time)
        else:
            log.error("Bootstrap failed - no data received")

        return self.df

    def poll(self) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        """
        Poll for new bars.

        Returns:
            (updated_df, new_bar_timestamp). new_bar_timestamp is None if no new bar.
        """
        if self.df is None:
            raise RuntimeError("Call bootstrap() first")

        recent = self._fetch_bars(100)
        if len(recent) == 0:
            return self.df, None

        latest_time = recent.index[-1]
        if self.last_bar_time is None or latest_time > self.last_bar_time:
            # New bar(s)
            new_bars = recent[recent.index > self.last_bar_time] if self.last_bar_time else recent

            # Append + dedupe + sort
            self.df = pd.concat([self.df, new_bars])
            self.df = self.df[~self.df.index.duplicated(keep="last")].sort_index()

            # Trim to lookback window
            cutoff = pd.Timestamp.now(tz=self.cfg.server_tz) - pd.Timedelta(minutes=self.cfg.lookback_minutes)
            self.df = self.df[self.df.index >= cutoff]

            self.last_bar_time = latest_time
            log.debug("New bar: %s, total bars: %d", latest_time, len(self.df))
            return self.df, latest_time

        # No new bar
        return self.df, None

    def run_forever(self, callback=None) -> None:
        """Run the polling loop; call `callback(df, new_bar_time)` on new bars."""
        if self.df is None:
            self.bootstrap()

        log.info("Starting live feed for %s, polling every %ss", self.cfg.symbol, self.cfg.poll_interval_sec)

        while True:
            try:
                df, new_bar = self.poll()
                if new_bar and callback:
                    callback(df, new_bar)
                time.sleep(self.cfg.poll_interval_sec)
            except KeyboardInterrupt:
                log.info("Feed stopped by user")
                break
            except Exception as e:
                log.error("Feed error: %s", e)
                time.sleep(self.cfg.poll_interval_sec)
                self._connected = False
                if not self._connect():
                    log.error("Reconnection failed, stopping feed")
                    break

    def close(self) -> None:
        """Clean shutdown."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            log.info("MT5 connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = MT5FeedCfg(symbol="GER40.cash", lookback_minutes=60 * 24, server_tz="Europe/Athens", poll_interval_sec=5)
    feed = MT5LiveM1(cfg)
    df = feed.bootstrap()
    print(f"Loaded {len(df)} bars")
    print(df.tail())

    # Example callback
    def on_new_bar(df: pd.DataFrame, bar_time: pd.Timestamp) -> None:
        latest = df.iloc[-1]
        print(
            f"New bar {bar_time}: O={latest['open']:.2f} H={latest['high']:.2f} "
            f"L={latest['low']:.2f} C={latest['close']:.2f}"
        )

    # Uncomment to run live:
    # feed.run_forever(callback=on_new_bar)
