# System Architecture

## Overview
Sophy4Lite trading research framework architecture and design decisions.

## Core Components

### Data Layer
- **Input**: M1 OHLC CSV files
- **Processing**: UTC normalization, timezone handling
- **Storage**: In-memory pandas DataFrames

### Strategy Layer
- **Signal Generation**: Multi-timeframe confluence analysis
- **Risk Management**: ATR-based position sizing
- **Entry/Exit Logic**: Confluence score thresholds

### Execution Layer
- **Backtesting**: Vectorized execution engine
- **Live Trading**: MT5 integration (Windows only)
- **Risk Guards**: FTMO compliance validation

### Output Layer
- **Results**: CSV exports, JSON metrics
- **Visualization**: Rich CLI tables
- **Logging**: Structured logging with levels

## Design Principles
1. **Modularity**: Clear separation of concerns
2. **Testability**: Comprehensive test coverage
3. **Performance**: Vectorized operations where possible
4. **Maintainability**: Clear interfaces and documentation

## Key Architectural Decisions
- Vectorized backtesting for performance
- UTC-first timezone handling
- Pre-trade risk validation
- Comprehensive signal filtering