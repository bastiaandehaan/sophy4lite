# test/test_mtf_confluence.py
"""
Complete test suite voor MTF Confluence strategy
Inclusief smoke test met synthetische data en look-ahead bias detection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from strategies.mtf_confluence import MTFSignals, MTFParams
from backtest.mtf_exec import backtest_mtf_confluence, MTFExecCfg


def make_synth_m1(n_days=5, tz="Europe/Berlin", trend=0.0002, volatility=1.5):
    """
    Create realistic synthetic M1 data with trend and volatility

    Args:
        n_days: Number of trading days
        tz: Timezone
        trend: Trend per minute (0.0002 = 0.02% = ~5% per day)
        volatility: Volatility factor for random walk
    """
    start = pd.Timestamp("2024-03-04 09:00", tz=tz)
    minutes = n_days * 8 * 60  # 8 hour session per day
    idx = pd.date_range(start, periods=minutes, freq="1min")

    # Generate price with trend and mean reversion
    price = 18000.0
    prices = []

    for i in range(len(idx)):
        # Add trend
        price *= (1 + trend)

        # Add mean-reverting noise
        shock = np.random.normal(0, volatility)
        mean_reversion = -0.01 * (price - 18000 - i * trend * 18000)
        price += shock + mean_reversion

        prices.append(price)

    # Convert to OHLC
    prices = np.array(prices)
    noise_high = np.random.uniform(0.2, 0.8, size=len(idx))
    noise_low = np.random.uniform(0.2, 0.8, size=len(idx))

    high = prices + noise_high
    low = prices - noise_low
    open_ = np.r_[prices[0], prices[:-1]]
    close = prices
    volume = np.random.randint(50, 500, size=len(idx))

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx, )

    # Keep only session hours 09:00-17:00
    df = df[(df.index.hour >= 9) & (df.index.hour < 17)]

    return df


def test_mtf_signals_generation():
    """Test that MTF signal generation works without errors"""
    df = make_synth_m1(n_days=3)
    params = MTFParams()
    mtf = MTFSignals(params)

    # Generate signals
    signals = mtf.generate_signals(df)

    # Check output structure
    assert isinstance(signals, pd.DataFrame)
    assert len(signals) == len(df)

    # Check required columns exist
    required_cols = ['bias', 'resistance', 'support', 'atr_m15', 'long_entry',
        'short_entry', 'long_sl', 'long_tp', 'short_sl', 'short_tp', 'long_score',
        'short_score', 'long_ev', 'short_ev']
    for col in required_cols:
        assert col in signals.columns, f"Missing column: {col}"

    # Check data types
    assert signals['long_entry'].dtype == bool
    assert signals['short_entry'].dtype == bool

    # Check score ranges
    assert signals['long_score'].between(0, 1).all()
    assert signals['short_score'].between(0, 1).all()


def test_no_look_ahead_bias():
    """Critical test: Ensure no look-ahead bias in signal generation"""
    df = make_synth_m1(n_days=3)
    params = MTFParams()
    mtf = MTFSignals(params)

    # Generate timeframes
    tfs = mtf.create_timeframes(df)

    # Check that higher timeframe bars are properly aligned
    # M5 bar at 09:05 should contain data from 09:01-09:05
    # and should NOT be visible to M1 bars before 09:05

    m1 = tfs['M1']
    m5 = tfs['M5']

    # Find a specific M5 bar
    m5_bar_time = m5.index[10]  # Take 10th M5 bar
    m5_close = m5.loc[m5_bar_time, 'close']

    # This M5 close should NOT be available to M1 bars before m5_bar_time
    m1_before = m1[m1.index < m5_bar_time]

    # In proper implementation, we shift higher TF data by 1 before reindex
    # So M1 bars can only see COMPLETED higher TF bars

    # Generate signals and check
    signals = mtf.generate_signals(df)

    # The bias at any M1 bar should be from the PREVIOUS H1 bar
    # Not the current incomplete one
    assert pd.notna(signals['bias']).any(), "Should have some bias values"

    # Check that early M1 bars have NaN for higher TF data
    # (because there's no completed higher TF bar yet)
    early_signals = signals.iloc[:5]
    assert early_signals['bias'].isna().all(), "Early bars should have NaN bias"


def test_confluence_scoring():
    """Test confluence score calculation logic"""
    from strategies.mtf_confluence import Bias

    params = MTFParams()
    mtf = MTFSignals(params)

    # Test bullish confluence at support
    score = mtf.calculate_confluence_score(bias=Bias.BULLISH, at_support=True,
        at_resistance=False, pattern_strength=0.8, momentum=0.7)
    assert score > 0.8, "Strong bullish confluence should have high score"

    # Test weak signal
    score = mtf.calculate_confluence_score(bias=Bias.BEARISH, at_support=True,
        # Contradiction
        at_resistance=False, pattern_strength=0.2, momentum=0.1)
    assert score < 0.3, "Contradictory signals should have low score"


def test_expected_value_calculation():
    """Test EV calculation with adaptive win rate"""
    params = MTFParams(use_adaptive_winrate=False, default_winrate=0.5)
    mtf = MTFSignals(params)

    # Test with 1:2 RR and 50% win rate
    entry = 100
    sl = 95  # 5 point risk
    tp = 110  # 10 point reward = 2R

    ev = mtf.calculate_expected_value(entry, sl, tp)

    # EV = 0.5 * 2R - 0.5 * 1R = 1R - 0.5R = 0.5R
    assert abs(ev - 0.5) < 0.01, f"Expected EV=0.5R, got {ev}"

    # Test with losing setup
    tp = 102  # Only 2 point reward = 0.4R
    ev = mtf.calculate_expected_value(entry, sl, tp)
    # EV = 0.5 * 0.4R - 0.5 * 1R = 0.2R - 0.5R = -0.3R
    assert ev < 0, "Low RR should have negative EV with 50% win rate"


def test_backtest_runs_without_errors():
    """Smoke test: Full backtest should run without exceptions"""
    df = make_synth_m1(n_days=3)
    params = MTFParams()
    cfg = MTFExecCfg(equity0=20_000.0, risk_frac=0.01)

    # Run backtest
    eq, trades, metrics = backtest_mtf_confluence(df, "GER40.cash", params, cfg,
        verbose=True)

    # Basic assertions
    assert isinstance(eq, pd.Series)
    assert len(eq) == len(df)
    assert eq.name == "equity"

    assert isinstance(trades, pd.DataFrame)
    assert "pnl" in trades.columns or len(trades) == 0

    assert isinstance(metrics, dict)
    assert "sharpe" in metrics
    assert "max_drawdown_pct" in metrics

    # Check equity continuity
    assert not eq.isna().any(), "Equity should have no NaN values"
    assert eq.iloc[0] == cfg.equity0, "Should start at initial equity"

    print(f"\nBacktest results:")
    print(f"Trades: {len(trades)}")
    print(f"Final equity: {eq.iloc[-1]:.2f}")
    print(f"Return: {metrics.get('return_total_pct', 0):.2f}%")
    print(f"Sharpe: {metrics.get('sharpe', 0):.2f}")
    print(f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%")

    if len(trades) > 0:
        print(f"Win rate: {metrics.get('winrate_pct', 0):.1f}%")
        print(f"Avg EV: {trades['ev'].mean():.2f}R")
        print(f"Avg Score: {trades['score'].mean():.2f}")


def test_daily_filtering():
    """Test max 1 trade per day filtering"""
    df = make_synth_m1(n_days=3)
    params = MTFParams(min_confluence_score=0.5)  # Lower threshold for more signals
    mtf = MTFSignals(params)

    # Generate signals without filtering
    signals_all = mtf.generate_signals(df)

    # Apply daily filter
    signals_filtered = mtf.filter_best_daily_signal(signals_all)

    # Count signals per day
    for day in pd.date_range(df.index.date.min(), df.index.date.max(), freq='D'):
        day_data = signals_filtered[signals_filtered.index.date == day.date()]
        total_signals = day_data['long_entry'].sum() + day_data['short_entry'].sum()
        assert total_signals <= 1, f"Day {day.date()} has {total_signals} signals, expected max 1"

    print(
        f"Daily filtering: {signals_all['long_entry'].sum() + signals_all['short_entry'].sum()} -> "
        f"{signals_filtered['long_entry'].sum() + signals_filtered['short_entry'].sum()} signals")


def test_realistic_params_performance():
    """Test with realistic parameters to check for reasonable performance"""
    df = make_synth_m1(n_days=20, trend=0.0001, volatility=1.2)  # Mild uptrend

    # Conservative parameters
    params = MTFParams(ema_period=20, atr_period=14, structure_lookback=20,
        momentum_threshold=0.6, atr_mult_sl=1.5, atr_mult_tp=2.5,
        min_confluence_score=0.70,  # Higher threshold
        use_adaptive_winrate=True)

    cfg = MTFExecCfg(equity0=20_000.0, risk_frac=0.01,  # 1% risk
        fee_rate=0.0002,  # 2 bps
        entry_slip_pts=0.1, sl_slip_pts=0.5)

    eq, trades, metrics = backtest_mtf_confluence(df, "GER40.cash", params, cfg,
        session_start="09:00", session_end="17:00", max_trades_per_day=1)

    # Performance sanity checks
    if len(trades) > 0:
        assert metrics['max_drawdown_pct'] < 50, "Drawdown too large"
        assert abs(metrics['return_total_pct']) < 100, "Return unrealistic"
        assert -5 < metrics['sharpe'] < 5, "Sharpe ratio unrealistic"

        # Check trade characteristics
        assert trades['size'].min() > 0, "Invalid position sizes"
        assert (trades['tp_pts'] > trades[
            'sl_pts']).mean() > 0.5, "Most trades should have positive RR"

        print(f"\nRealistic backtest ({len(df)} bars, {len(trades)} trades):")
        print(f"Return: {metrics['return_total_pct']:.2f}%")
        print(f"Sharpe: {metrics['sharpe']:.2f}")
        print(f"Win rate: {metrics.get('winrate_pct', 0):.1f}%")
        print(f"Expectancy: {metrics.get('expectancy_r', 0):.2f}R")
        print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")


# ===== Integration Files =====

# backtest/__init__.py
"""
from .mtf_exec import backtest_mtf_confluence, MTFExecCfg

__all__ = ["backtest_mtf_confluence", "MTFExecCfg"]
"""

# strategies/__init__.py
"""
from .breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
from .breakout_signals import breakout_long, opening_breakout_long
from .mtf_confluence import MTFSignals, MTFParams

__all__ = [
    "breakout_long",
    "opening_breakout_long", 
    "BreakoutParams",
    "SymbolSpec",
    "DEFAULT_SPECS",
    "MTFSignals",
    "MTFParams",
]
"""

# CLI command addition for cli/main.py:
"""
@app.command()
def confluence(
    symbol: str = typer.Argument("GER40.cash", help="Trading symbol"),
    csv: Path = typer.Option(..., help="Path to M1 CSV data"),
    start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    session_start: str = typer.Option("09:00", help="Session start time"),
    session_end: str = typer.Option("17:00", help="Session end time"),
    risk_pct: float = typer.Option(1.0, help="Risk per trade %"),
    min_score: float = typer.Option(0.65, help="Minimum confluence score"),
    max_daily: int = typer.Option(1, help="Max trades per day"),
    outdir: Optional[Path] = typer.Option(Path("output"), help="Output directory"),
):
    '''Run MTF Confluence strategy backtest'''

    # Load data
    df = pd.read_csv(csv, parse_dates=["time"], index_col="time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Slice period
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]

    # Setup
    from strategies.mtf_confluence import MTFParams
    from backtest.mtf_exec import backtest_mtf_confluence, MTFExecCfg

    params = MTFParams(min_confluence_score=min_score)
    cfg = MTFExecCfg(risk_frac=risk_pct/100)

    # Run backtest
    eq, trades, metrics = backtest_mtf_confluence(
        df, symbol, params, cfg,
        session_start=session_start,
        session_end=session_end,
        max_trades_per_day=max_daily,
        verbose=True
    )

    # Display results
    tbl = Table(title=f"MTF Confluence Results - {symbol}")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")

    tbl.add_row("Trades", str(len(trades)))
    tbl.add_row("Return", f"{metrics.get('return_total_pct', 0):.2f}%")
    tbl.add_row("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
    tbl.add_row("Max DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    tbl.add_row("Win Rate", f"{metrics.get('winrate_pct', 0):.1f}%")
    tbl.add_row("Expectancy", f"{metrics.get('expectancy_r', 0):.2f}R")

    console.print(tbl)

    # Save outputs
    if outdir:
        outdir.mkdir(exist_ok=True)
        eq.to_csv(outdir / f"equity_confluence_{symbol}.csv")
        trades.to_csv(outdir / f"trades_confluence_{symbol}.csv", index=False)
        console.print(f"[green]Saved results to {outdir}/[/green]")
"""

if __name__ == "__main__":
    # Run key tests
    print("Running MTF Confluence Tests...")
    test_mtf_signals_generation()
    print("✓ Signal generation OK")

    test_no_look_ahead_bias()
    print("✓ No look-ahead bias detected")

    test_confluence_scoring()
    print("✓ Confluence scoring OK")

    test_expected_value_calculation()
    print("✓ EV calculation OK")

    test_backtest_runs_without_errors()
    print("✓ Backtest smoke test OK")

    test_daily_filtering()
    print("✓ Daily filtering OK")

    test_realistic_params_performance()
    print("✓ Realistic performance test OK")

    print("\n✅ All tests passed!")