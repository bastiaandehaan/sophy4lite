"""
Data structures for trade records and results.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class TradeRec:
    """Trade record structure for backtest results."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    sl: float = 0.0
    tp: float = 0.0
    size: float = 0.0
    pnl_cash: float = 0.0
    pnl_pct: float = 0.0
    sl_pts: float = 0.0
    tp_pts: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {'symbol': self.symbol, 'side': self.side, 'entry_time': self.entry_time,
            'entry_price': self.entry_price, 'exit_time': self.exit_time,
            'exit_price': self.exit_price, 'sl': self.sl, 'tp': self.tp,
            'size': self.size, 'pnl_cash': self.pnl_cash, 'pnl_pct': self.pnl_pct,
            'sl_pts': self.sl_pts, 'tp_pts': self.tp_pts, }

    @property
    def is_winner(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl_cash > 0

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate realized R:R ratio."""
        if self.sl_pts == 0:
            return 0.0
        return self.tp_pts / self.sl_pts


@dataclass
class BacktestResult:
    """Container for complete backtest results."""
    equity_series: pd.Series
    trades: list[TradeRec]
    metrics: dict

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to equity and trades DataFrames."""
        equity_df = self.equity_series.to_frame('equity')
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        return equity_df, trades_df