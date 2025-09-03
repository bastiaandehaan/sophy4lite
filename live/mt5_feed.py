# live/mt5_feed.py
"""
MT5 Live Feed for real-time M1 data streaming
MISSING FILE - This was imported but not provided
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time
import logging

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 required for live trading: pip install MetaTrader5")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MT5FeedCfg:
    """Configuration for MT5 live feed"""
    symbol: str = "GER40.cash"
    lookback_minutes: int = 60 * 24 * 2  # 2 days
    server_tz: str = "Europe/Athens"  # FTMO server timezone
    out_tz: Optional[str] = None  # Output timezone (None = keep server)
    poll_interval_sec: int = 5  # How often to check for new bars
    reconnect_attempts: int = 3  # Auto-reconnect attempts


class MT5LiveM1:
    """
    Live M1 data feed from MT5
    Maintains rolling window of recent bars
    """

    def __init__(self, cfg: MT5FeedCfg):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[pd.Timestamp] = None
        self._connected = False

    def _connect(self) -> bool:
        """Initialize MT5 connection"""
        if self._connected:
            return True

        for attempt in range(self.cfg.reconnect_attempts):
            if mt5.initialize():
                self._connected = True
                log.info(f"MT5 connected (attempt {attempt + 1})")

                # Ensure symbol is available
                if not mt5.symbol_select(self.cfg.symbol, True):
                    log.error(f"Failed to select symbol {self.cfg.symbol}")
                    return False

                return True

            log.warning(f"MT5 connection attempt {attempt + 1} failed")
            time.sleep(2)

        return False

    def _fetch_bars(self, num_bars: int) -> pd.DataFrame:
        """Fetch last N M1 bars from MT5"""
        if not self._connect():
            raise RuntimeError("MT5 connection failed")

        # Get rates
        rates = mt5.copy_rates_from_pos(self.cfg.symbol, mt5.TIMEFRAME_M1, 0,
            # From current bar
            num_bars)

        if rates is None or len(rates) == 0:
            log.warning(f"No data received for {self.cfg.symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime index
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('time')

        # Apply timezone
        df.index = df.index.tz_convert(self.cfg.server_tz)

        # Rename columns
        df = df.rename(columns={'tick_volume': 'volume', 'real_volume': 'real_volume'})

        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Apply output timezone if specified
        if self.cfg.out_tz:
            df.index = df.index.tz_convert(self.cfg.out_tz)

        return df

    def bootstrap(self) -> pd.DataFrame:
        """
        Initial load of historical data
        Call this once at startup
        """
        log.info(
            f"Bootstrapping {self.cfg.lookback_minutes} minutes of {self.cfg.symbol}")

        # Calculate number of bars (account for weekends)
        # Assume max 5 days of data needed for requested minutes
        max_bars = self.cfg.lookback_minutes * 2  # Safety margin

        self.df = self._fetch_bars(max_bars)

        if len(self.df) > 0:
            # Trim to requested lookback
            cutoff = pd.Timestamp.now(tz=self.cfg.server_tz) - pd.Timedelta(
                minutes=self.cfg.lookback_minutes)
            self.df = self.df[self.df.index >= cutoff]

            self.last_bar_time = self.df.index[-1]
            log.info(f"Bootstrapped {len(self.df)} bars, last: {self.last_bar_time}")
        else:
            log.error("Bootstrap failed - no data received")

        return self.df

    def poll(self) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        """
        Poll for new bars
        Returns: (updated_df, new_bar_timestamp)

        new_bar_timestamp is None if no new bar, else the timestamp of the new bar
        """
        if self.df is None:
            raise RuntimeError("Call bootstrap() first")

        # Fetch recent bars (last 100 should be enough)
        recent = self._fetch_bars(100)

        if len(recent) == 0:
            return self.df, None

        # Check for new bars
        latest_time = recent.index[-1]

        if self.last_bar_time is None or latest_time > self.last_bar_time:
            # New bar(s) detected
            new_bars = recent[
                recent.index > self.last_bar_time] if self.last_bar_time else recent

            # Append new bars
            self.df = pd.concat([self.df, new_bars])

            # Remove duplicates (shouldn't happen but safety)
            self.df = self.df[~self.df.index.duplicated(keep='last')]

            # Sort by time
            self.df = self.df.sort_index()

            # Trim to lookback window
            cutoff = pd.Timestamp.now(tz=self.cfg.server_tz) - pd.Timedelta(
                minutes=self.cfg.lookback_minutes)
            self.df = self.df[self.df.index >= cutoff]

            self.last_bar_time = latest_time

            log.debug(f"New bar: {latest_time}, total bars: {len(self.df)}")
            return self.df, latest_time

        # No new bar
        return self.df, None

    def run_forever(self, callback=None):
        """
        Convenience method to run polling loop forever

        Args:
            callback: Optional function(df, new_bar_time) to call on new bars
        """
        if self.df is None:
            self.bootstrap()

        log.info(
            f"Starting live feed for {self.cfg.symbol}, polling every {self.cfg.poll_interval_sec}s")

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
                log.error(f"Feed error: {e}")
                time.sleep(self.cfg.poll_interval_sec)

                # Try to reconnect
                self._connected = False
                if not self._connect():
                    log.error("Reconnection failed, stopping feed")
                    break

    def close(self):
        """Clean shutdown"""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            log.info("MT5 connection closed")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = MT5FeedCfg(symbol="GER40.cash", lookback_minutes=60 * 24,  # 1 day
        server_tz="Europe/Athens", poll_interval_sec=5)

    feed = MT5LiveM1(cfg)
    df = feed.bootstrap()

    print(f"Loaded {len(df)} bars")
    print(df.tail())


    # Example callback
    def on_new_bar(df, bar_time):
        latest = df.iloc[-1]
        print(f"New bar {bar_time}: O={latest['open']:.2f} H={latest['high']:.2f} "
              f"L={latest['low']:.2f} C={latest['close']:.2f}")

    # Run live (uncomment to test)  # feed.run_forever(callback=on_new_bar)