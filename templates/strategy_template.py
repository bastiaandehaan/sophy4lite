"""
Template for new trading strategy implementations.

Usage:
1. Copy this file to strategies/your_strategy_name.py
2. Replace TemplateStrategy with your strategy name
3. Implement the required methods
4. Add comprehensive tests in test/test_your_strategy_name.py
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class SignalType(str, Enum):
    """Signal types for entry conditions"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(frozen=True)
class TemplateParams:
    """
    Strategy parameters for template strategy.
    
    All parameters should be documented with:
    - Purpose and impact on strategy behavior
    - Reasonable default values
    - Valid ranges where applicable
    """
    # Example parameters
    lookback_period: int = 20
    signal_threshold: float = 0.6
    risk_multiplier: float = 1.5
    
    # Risk management
    max_risk_pct: float = 0.02  # 2% max risk per trade
    min_score: float = 0.5      # Minimum signal quality
    
    def __post_init__(self):
        """Validate parameters on initialization"""
        assert self.lookback_period > 0, "Lookback period must be positive"
        assert 0 < self.signal_threshold < 1, "Threshold must be between 0 and 1"
        assert self.risk_multiplier > 0, "Risk multiplier must be positive"


class TemplateStrategy:
    """
    Template trading strategy implementation.
    
    This class provides a complete template for implementing new strategies
    following the framework conventions and best practices.
    """
    
    def __init__(self, params: TemplateParams = TemplateParams()):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Strategy parameters instance
        """
        self.params = params
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate strategy parameters"""
        # Add parameter validation logic here
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLC data.
        
        Args:
            df: DataFrame with OHLC data and timezone-aware index
            
        Returns:
            DataFrame with same index as input, containing signal columns:
            - long_entry: Boolean series for long entries
            - short_entry: Boolean series for short entries  
            - long_sl: Stop loss levels for long trades
            - long_tp: Take profit levels for long trades
            - short_sl: Stop loss levels for short trades
            - short_tp: Take profit levels for short trades
            - signal_strength: Quality score [0, 1]
        """
        # Initialize result DataFrame
        signals = pd.DataFrame(index=df.index)
        
        # Calculate indicators
        signals = self._calculate_indicators(df, signals)
        
        # Generate entry signals
        signals = self._generate_entry_signals(df, signals)
        
        # Calculate stop loss and take profit levels
        signals = self._calculate_levels(df, signals)
        
        # Apply filters and quality scores
        signals = self._apply_filters(df, signals)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for signal generation.
        
        Args:
            df: OHLC data
            signals: Signals DataFrame to populate
            
        Returns:
            Updated signals DataFrame with indicator columns
        """
        # Example: Simple moving average
        signals['sma'] = df['close'].rolling(self.params.lookback_period).mean()
        
        # Example: ATR for position sizing
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        signals['atr'] = true_range.rolling(14).mean()
        
        return signals
    
    def _generate_entry_signals(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry signals based on strategy logic.
        
        Args:
            df: OHLC data
            signals: Signals DataFrame with indicators
            
        Returns:
            Updated signals DataFrame with entry signals
        """
        # Initialize signal columns
        signals['long_entry'] = False
        signals['short_entry'] = False
        signals['signal_strength'] = 0.0
        
        # Example signal generation logic
        bullish_condition = df['close'] > signals['sma']
        bearish_condition = df['close'] < signals['sma']
        
        # Apply signal threshold
        signals.loc[bullish_condition, 'long_entry'] = True
        signals.loc[bullish_condition, 'signal_strength'] = self.params.signal_threshold
        
        signals.loc[bearish_condition, 'short_entry'] = True  
        signals.loc[bearish_condition, 'signal_strength'] = self.params.signal_threshold
        
        return signals
    
    def _calculate_levels(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            df: OHLC data
            signals: Signals DataFrame with entry signals
            
        Returns:
            Updated signals DataFrame with SL/TP levels
        """
        # Initialize level columns
        signals['long_sl'] = np.nan
        signals['long_tp'] = np.nan
        signals['short_sl'] = np.nan
        signals['short_tp'] = np.nan
        
        # Calculate levels based on ATR
        atr_distance = signals['atr'] * self.params.risk_multiplier
        
        # Long trade levels
        long_entries = signals['long_entry']
        signals.loc[long_entries, 'long_sl'] = df.loc[long_entries, 'close'] - atr_distance.loc[long_entries]
        signals.loc[long_entries, 'long_tp'] = df.loc[long_entries, 'close'] + (atr_distance.loc[long_entries] * 2)
        
        # Short trade levels  
        short_entries = signals['short_entry']
        signals.loc[short_entries, 'short_sl'] = df.loc[short_entries, 'close'] + atr_distance.loc[short_entries]
        signals.loc[short_entries, 'short_tp'] = df.loc[short_entries, 'close'] - (atr_distance.loc[short_entries] * 2)
        
        return signals
    
    def _apply_filters(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality filters to signals.
        
        Args:
            df: OHLC data
            signals: Signals DataFrame with raw signals
            
        Returns:
            Filtered signals DataFrame
        """
        # Filter by minimum signal strength
        min_strength_mask = signals['signal_strength'] >= self.params.min_score
        
        signals.loc[~min_strength_mask, 'long_entry'] = False
        signals.loc[~min_strength_mask, 'short_entry'] = False
        
        # Additional filters can be added here
        # - Volume filters
        # - Time-based filters  
        # - Market condition filters
        
        return signals
    
    def calculate_expected_value(self, entry: float, sl: float, tp: float, 
                               win_rate: float = 0.5) -> float:
        """
        Calculate expected value of a trade setup.
        
        Args:
            entry: Entry price
            sl: Stop loss price
            tp: Take profit price
            win_rate: Historical win rate [0, 1]
            
        Returns:
            Expected value in R (risk units)
        """
        if pd.isna(entry) or pd.isna(sl) or pd.isna(tp):
            return 0.0
            
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk == 0:
            return 0.0
            
        risk_reward_ratio = reward / risk
        expected_value = (win_rate * risk_reward_ratio) - ((1 - win_rate) * 1)
        
        return expected_value


# Example usage and testing functions
def example_usage():
    """Example of how to use the template strategy"""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    # Initialize strategy
    params = TemplateParams(lookback_period=10, signal_threshold=0.7)
    strategy = TemplateStrategy(params)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print(f"Generated {signals['long_entry'].sum()} long signals")
    print(f"Generated {signals['short_entry'].sum()} short signals")
    print(f"Average signal strength: {signals['signal_strength'].mean():.2f}")
    
    return signals


if __name__ == "__main__":
    # Run example usage
    example_signals = example_usage()
    print("Template strategy example completed successfully!")