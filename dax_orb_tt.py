#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAX ORB Realistic - MT5 data fetch + realistic spread/slippage simulation
CRITICAL: This version includes transaction costs for realistic results
"""
from __future__ import annotations
import argparse, glob, math, time, multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# Optional MT5 integration
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 package not installed. Install with: pip install MetaTrader5")
    print("   Continuing with CSV file support only.\n")


# ============================= REALISTIC COSTS =====================================
class TradingCosts:
    """
    Realistic trading costs for DAX CFD/Futures
    Based on typical retail broker conditions
    """
    # Spread (bid-ask) - varies by time of day
    SPREAD_OPENING = 2.0  # Points during first 30 min (high volatility)
    SPREAD_NORMAL = 1.0  # Points during normal hours
    SPREAD_CLOSING = 1.5  # Points during last hour

    # Slippage on stop orders (worse during volatility)
    SLIPPAGE_STOP_OPENING = 1.5  # Extra points on stops during opening
    SLIPPAGE_STOP_NORMAL = 0.5  # Normal slippage on stops

    # Commission per side (points equivalent)
    COMMISSION = 0.5  # Per trade side (entry + exit = 1.0 total)

    @staticmethod
    def get_entry_cost(hour: int, minute: int, is_opening_window: bool) -> float:
        """Get realistic entry cost based on time of day"""
        total_minutes = hour * 60 + minute

        # Opening window (09:00-09:30) - highest costs
        if is_opening_window and 540 <= total_minutes <= 570:
            return TradingCosts.SPREAD_OPENING + TradingCosts.COMMISSION
        # Closing hour (17:30-18:30) - elevated costs
        elif 1050 <= total_minutes <= 1110:
            return TradingCosts.SPREAD_CLOSING + TradingCosts.COMMISSION
        # Normal hours
        else:
            return TradingCosts.SPREAD_NORMAL + TradingCosts.COMMISSION

    @staticmethod
    def get_stop_slippage(hour: int, minute: int) -> float:
        """Get realistic slippage for stop orders"""
        total_minutes = hour * 60 + minute

        # Opening 30 minutes - high slippage
        if 540 <= total_minutes <= 570:
            return TradingCosts.SLIPPAGE_STOP_OPENING
        else:
            return TradingCosts.SLIPPAGE_STOP_NORMAL


# ============================= MT5 Integration ======================================
def fetch_mt5_data(symbol: str = "DE40", days_back: int = 60, server: str = None,
                   login: int = None, password: str = None) -> pd.DataFrame:
    """
    Fetch M1 data directly from MT5

    Args:
        symbol: Symbol to fetch (DE40, GER40, etc.)
        days_back: Number of days of history
        server: MT5 server (optional, uses current if None)
        login: MT5 account (optional)
        password: MT5 password (optional)
    """
    if not MT5_AVAILABLE:
        raise ImportError("MetaTrader5 package not installed")

    # Initialize MT5
    if login and password and server:
        if not mt5.initialize(login=login, password=password, server=server):
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
    else:
        if not mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")

    # Check symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        mt5.shutdown()
        raise ValueError(
            f"Symbol {symbol} not found. Available symbols: {[s.name for s in mt5.symbols_get()][:10]}...")

    # Calculate date range
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=days_back)

    # Fetch M1 bars
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)

    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise ValueError(f"No data received for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(
        columns={'tick_volume': 'volume'})

    mt5.shutdown()

    print(f"✓ Fetched {len(df):,} M1 bars from MT5 for {symbol}")
    print(f"  Period: {df['time'].min()} to {df['time'].max()}")

    return df


# ============================= CLI ==================================================
def parse_args():
    p = argparse.ArgumentParser(description="DAX ORB Realistic - MT5 + True Costs")

    # Data source
    p.add_argument("paths", nargs="*", default=None,
                   help="CSV files (if none, tries MT5 or searches for CSVs)")
    p.add_argument("--mt5", action="store_true", help="Fetch data from MT5")
    p.add_argument("--symbol", default="DE40", help="MT5 symbol (DE40, GER40, etc.)")
    p.add_argument("--days", type=int, default=60,
                   help="Days of history to fetch from MT5")

    # Trading parameters
    p.add_argument("--tz", default="Europe/Athens", help="Server timezone")
    p.add_argument("--pre", default="08:00-09:00", help="Pre-market window")
    p.add_argument("--sess", default="09:00-18:30", help="Session window")

    # Strategy modes
    p.add_argument("--tradertom", action="store_true",
                   help="TraderTom fixed params (SL=9, TP=6)")
    p.add_argument("--realistic", action="store_true", default=True,
                   help="Apply realistic costs (default: ON)")
    p.add_argument("--no-costs", dest="realistic", action="store_false",
                   help="Disable transaction costs")
    p.add_argument("--opening-window", type=int, default=30,
                   help="Minutes after open for entry")

    # Parameter sweep (if not --tradertom)
    p.add_argument("--sl-min", type=float, default=5.0)
    p.add_argument("--sl-max", type=float, default=15.0)
    p.add_argument("--sl-step", type=float, default=1.0)
    p.add_argument("--tp-min", type=float, default=3.0)
    p.add_argument("--tp-max", type=float, default=12.0)
    p.add_argument("--tp-step", type=float, default=1.0)

    # Analysis
    p.add_argument("--min-trades", type=int, default=30, help="Min trades for validity")
    p.add_argument("--train-pct", type=float, default=0.7,
                   help="Training set percentage (0.7 = 70%)")

    # Output
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--save-trades", action="store_true", help="Save trade list")
    p.add_argument("--verbose", action="store_true", help="Detailed output")

    return p.parse_args()


# ============================= Data Loading =========================================
def load_data(args) -> pd.DataFrame:
    """Load data from MT5 or CSV files"""

    # Try MT5 first if requested
    if args.mt5:
        if not MT5_AVAILABLE:
            print("❌ MT5 requested but MetaTrader5 package not installed")
            print("   Install with: pip install MetaTrader5")
            print("   Falling back to CSV files...")
        else:
            try:
                return fetch_mt5_data(args.symbol, args.days)
            except Exception as e:
                print(f"❌ MT5 fetch failed: {e}")
                print("   Falling back to CSV files...")

    # Load from CSV files
    if not args.paths:
        # Search for CSV files
        current_dir = Path.cwd()
        data_dir = current_dir / "data"
        patterns = ["*.csv", "*_M1.csv", "GER*.csv", "DAX*.csv", "DE40*.csv",
            "data/*.csv", "data/*_M1.csv"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(str(current_dir / pattern)))
            if data_dir.exists():
                files.extend(glob.glob(str(data_dir / pattern)))
        files = list(set(files))  # Remove duplicates
    else:
        files = [f for pat in args.paths for f in glob.glob(pat)]

    if not files:
        raise FileNotFoundError("No data source available!\n"
                                "Options:\n"
                                "1. Use --mt5 to fetch from MetaTrader5\n"
                                "2. Place CSV files in current directory\n"
                                "3. Specify CSV path as argument")

    # Read and merge CSV files
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'time' not in df.columns and 'date' in df.columns:
            df['time'] = df['date']
        required = ['time', 'open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            print(f"⚠️ Skipping {fp}: missing required columns")
            continue
        dfs.append(df[required + (['volume'] if 'volume' in df.columns else [])])

    if not dfs:
        raise ValueError("No valid CSV files found")

    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time')

    print(f"✓ Loaded {len(df):,} M1 bars from {len(files)} file(s)")
    print(f"  Period: {df['time'].min()} to {df['time'].max()}")

    return df


# ============================= Realistic Backtest ===================================
def run_realistic_backtest(df: pd.DataFrame, tz: str, pre_s: str, sess_s: str,
                           sl: float, tp: float, use_costs: bool = True,
                           tradertom_mode: bool = False, opening_window: int = 30,
                           verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Realistic backtest with proper costs and execution simulation

    CRITICAL IMPROVEMENTS:
    1. Spread costs on entry
    2. Slippage on stop losses
    3. Commission on both sides
    4. No lookahead bias on pre-market levels
    5. Proper position reset between tests
    """

    # Convert to local timezone
    df = df.copy()
    df['local'] = df['time'].dt.tz_convert(ZoneInfo(tz))
    df = df.set_index('local').sort_index()

    # Add time components for cost calculation
    df['hour'] = df.index.hour
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute

    # Parse time windows
    pre_start, pre_end = [int(h) * 60 + int(m) for h, m in
                          [t.split(':') for t in pre_s.split('-')]]
    sess_start, sess_end = [int(h) * 60 + int(m) for h, m in
                            [t.split(':') for t in sess_s.split('-')]]

    # Build pre-market levels (with proper timing)
    orb_levels = {}
    for day, day_data in df.groupby(df.index.date):
        # Only use data up to pre-market end (no lookahead)
        pre_data = day_data[(day_data['minute_of_day'] >= pre_start) & (
                    day_data['minute_of_day'] < pre_end)]
        if not pre_data.empty:
            orb_levels[day] = {'low': float(pre_data['low'].min()),
                'high': float(pre_data['high'].max()),
                'pre_close': float(pre_data['close'].iloc[-1])  # Where pre-market ended
            }

    trades = []
    position = None  # Proper position tracking

    for day, day_data in df.groupby(df.index.date):
        # Skip if no pre-market data or already in position from previous day
        if day not in orb_levels or position is not None:
            continue

        orb = orb_levels[day]
        if not (math.isfinite(orb['high']) and math.isfinite(orb['low']) and orb[
            'high'] > orb['low']):
            continue

        # Determine entry window
        if tradertom_mode:
            entry_end = sess_start + opening_window
        else:
            entry_end = sess_end

        entry_window = day_data[(day_data['minute_of_day'] >= sess_start) & (
                    day_data['minute_of_day'] <= entry_end)]

        if entry_window.empty:
            continue

        # Check for breakout (pending order simulation)
        for idx, bar in entry_window.iterrows():
            if position is not None:  # Already entered
                break

            # Calculate costs for this specific time
            if use_costs:
                entry_cost = TradingCosts.get_entry_cost(bar['hour'],
                    bar['minute_of_day'] % 60, tradertom_mode)
            else:
                entry_cost = 0

            # Buy stop order simulation (triggers if high touches level)
            if bar['high'] >= orb['high']:
                position = {'side': 'long', 'entry_time': idx, 'entry_raw': orb['high'],
                    'entry': orb['high'] + entry_cost,  # Pay spread + commission
                    'stop': orb['high'] - sl, 'target': orb['high'] + tp, 'day': day}
                if verbose:
                    print(
                        f"LONG entry: {idx} @ {position['entry']:.1f} (raw: {position['entry_raw']:.1f}, cost: {entry_cost:.1f})")
                break

            # Sell stop order simulation
            elif bar['low'] <= orb['low']:
                position = {'side': 'short', 'entry_time': idx, 'entry_raw': orb['low'],
                    'entry': orb['low'] - entry_cost,  # Pay spread + commission
                    'stop': orb['low'] + sl, 'target': orb['low'] - tp, 'day': day}
                if verbose:
                    print(
                        f"SHORT entry: {idx} @ {position['entry']:.1f} (raw: {position['entry_raw']:.1f}, cost: {entry_cost:.1f})")
                break

        # Check for exit if position was opened
        if position is not None:
            # Use all remaining data for exit (not limited to entry window)
            exit_data = day_data.loc[position['entry_time']:]

            for idx, bar in exit_data.iterrows():
                exit_triggered = False

                if position['side'] == 'long':
                    # Check stop loss
                    if bar['low'] <= position['stop']:
                        if use_costs:
                            slippage = TradingCosts.get_stop_slippage(bar['hour'], bar[
                                'minute_of_day'] % 60)
                            exit_price = position['stop'] - slippage
                        else:
                            exit_price = position['stop']

                        pnl = exit_price - position['entry']
                        trades.append({'day': str(position['day']),
                            'entry_time': position['entry_time'], 'exit_time': idx,
                            'side': position['side'], 'entry': position['entry_raw'],
                            # Show raw entry in results
                            'exit': exit_price, 'reason': 'SL', 'pnl': pnl,
                            'costs': position['entry'] - position[
                                'entry_raw'] + slippage if use_costs else 0})
                        exit_triggered = True

                    # Check take profit
                    elif bar['high'] >= position['target']:
                        if use_costs:
                            exit_price = position['target'] - TradingCosts.COMMISSION
                        else:
                            exit_price = position['target']

                        pnl = exit_price - position['entry']
                        trades.append({'day': str(position['day']),
                            'entry_time': position['entry_time'], 'exit_time': idx,
                            'side': position['side'], 'entry': position['entry_raw'],
                            'exit': exit_price, 'reason': 'TP', 'pnl': pnl,
                            'costs': position['entry'] - position[
                                'entry_raw'] + TradingCosts.COMMISSION if use_costs else 0})
                        exit_triggered = True

                else:  # Short position
                    # Check stop loss
                    if bar['high'] >= position['stop']:
                        if use_costs:
                            slippage = TradingCosts.get_stop_slippage(bar['hour'], bar[
                                'minute_of_day'] % 60)
                            exit_price = position['stop'] + slippage
                        else:
                            exit_price = position['stop']

                        pnl = position['entry'] - exit_price
                        trades.append({'day': str(position['day']),
                            'entry_time': position['entry_time'], 'exit_time': idx,
                            'side': position['side'], 'entry': position['entry_raw'],
                            'exit': exit_price, 'reason': 'SL', 'pnl': pnl,
                            'costs': position['entry_raw'] - position[
                                'entry'] + slippage if use_costs else 0})
                        exit_triggered = True

                    # Check take profit
                    elif bar['low'] <= position['target']:
                        if use_costs:
                            exit_price = position['target'] + TradingCosts.COMMISSION
                        else:
                            exit_price = position['target']

                        pnl = position['entry'] - exit_price
                        trades.append({'day': str(position['day']),
                            'entry_time': position['entry_time'], 'exit_time': idx,
                            'side': position['side'], 'entry': position['entry_raw'],
                            'exit': exit_price, 'reason': 'TP', 'pnl': pnl,
                            'costs': position['entry_raw'] - position[
                                'entry'] + TradingCosts.COMMISSION if use_costs else 0})
                        exit_triggered = True

                if exit_triggered:
                    position = None
                    break

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'winrate': 0,
            'pf': 0, 'avg_winner': 0, 'avg_loser': 0, 'total_costs': 0, 'gross_pnl': 0}

    # Calculate statistics
    wins = (trades_df['reason'] == 'TP').sum()
    losses = (trades_df['reason'] == 'SL').sum()
    total_pnl = trades_df['pnl'].sum()
    total_costs = trades_df['costs'].sum() if use_costs else 0
    gross_pnl = total_pnl + total_costs  # PnL before costs

    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    stats = {'trades': len(trades_df), 'wins': wins, 'losses': losses, 'pnl': total_pnl,
        'gross_pnl': gross_pnl, 'total_costs': total_costs,
        'winrate': 100 * wins / len(trades_df) if len(trades_df) > 0 else 0,
        'pf': winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if len(
            losing_trades) > 0 else float('inf'),
        'avg_winner': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loser': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
        'max_dd': calculate_max_drawdown(trades_df['pnl']),
        'sharpe': calculate_sharpe_ratio(trades_df['pnl'])}

    return trades_df, stats


def calculate_max_drawdown(pnl_series):
    """Calculate maximum drawdown"""
    cumsum = pnl_series.cumsum()
    running_max = cumsum.expanding().max()
    drawdown = cumsum - running_max
    return float(drawdown.min())


def calculate_sharpe_ratio(pnl_series, periods_per_year=252 * 390):
    """Calculate Sharpe ratio (assuming minute data)"""
    if len(pnl_series) < 2:
        return 0
    returns = pnl_series
    if returns.std() == 0:
        return 0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


# ============================= Visualization ========================================
def create_comprehensive_plots(df, trades_df, stats, args):
    """Create comprehensive analysis plots"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Equity Curve with Drawdown
    ax1 = fig.add_subplot(gs[0, :2])
    if not trades_df.empty:
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']

        ax1.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 'b-',
                 label='Equity', linewidth=2)
        ax1.fill_between(range(len(trades_df)), trades_df['cumulative_pnl'],
                         trades_df['running_max'], alpha=0.3, color='red',
                         label='Drawdown')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title(f'Equity Curve ({"With" if args.realistic else "Without"} Costs)')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative PnL (points)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Win/Loss Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    if stats['trades'] > 0:
        sizes = [stats['wins'], stats['losses']]
        colors = ['green', 'red']
        ax2.pie(sizes, labels=['Wins', 'Losses'], colors=colors, autopct='%1.1f%%')
        ax2.set_title(f"Win Rate: {stats['winrate']:.1f}%")

    # 3. PnL Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if not trades_df.empty:
        ax3.hist(trades_df['pnl'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax3.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', alpha=0.8,
                    label=f'Mean: {trades_df["pnl"].mean():.2f}')
        ax3.set_title('PnL Distribution')
        ax3.set_xlabel('PnL (points)')
        ax3.set_ylabel('Frequency')
        ax3.legend()

    # 4. Time of Day Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    if not trades_df.empty:
        trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_pnl = trades_df.groupby('entry_hour')['pnl'].sum()
        colors = ['green' if x > 0 else 'red' for x in hourly_pnl.values]
        ax4.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.7,
                edgecolor='black')
        ax4.set_title('PnL by Entry Hour')
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Total PnL')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 5. Recent Trades Performance
    ax5 = fig.add_subplot(gs[1, 2])
    if not trades_df.empty:
        recent = trades_df.tail(20)
        colors = ['green' if x > 0 else 'red' for x in recent['pnl']]
        ax5.bar(range(len(recent)), recent['pnl'], color=colors, alpha=0.7,
                edgecolor='black')
        ax5.set_title('Last 20 Trades')
        ax5.set_xlabel('Trade')
        ax5.set_ylabel('PnL')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 6. Statistics Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')

    # Create statistics table
    table_data = [['Metric', 'Value', 'Metric', 'Value'],
        ['Total Trades', f"{stats['trades']}", 'Win Rate', f"{stats['winrate']:.1f}%"],
        ['Wins', f"{stats['wins']}", 'Losses', f"{stats['losses']}"],
        ['Net PnL', f"{stats['pnl']:.1f}", 'Gross PnL', f"{stats['gross_pnl']:.1f}"],
        ['Total Costs', f"{stats['total_costs']:.1f}", 'Profit Factor',
         f"{stats['pf']:.2f}"],
        ['Avg Winner', f"{stats['avg_winner']:.1f}", 'Avg Loser',
         f"{stats['avg_loser']:.1f}"],
        ['Max Drawdown', f"{stats['max_dd']:.1f}", 'Sharpe Ratio',
         f"{stats['sharpe']:.3f}"], ['Required WR (BE)',
                                     f"{100 * args.sl_min / (args.sl_min + args.tp_min):.1f}%" if not args.tradertom else "60.0%",
                                     'Actual vs Required',
                                     f"{stats['winrate'] - 100 * args.sl_min / (args.sl_min + args.tp_min):.1f}%" if not args.tradertom else f"{stats['winrate'] - 60:.1f}%"]]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color cells based on values
    if stats['pnl'] > 0:
        table[(3, 1)].set_facecolor('#90EE90')
    else:
        table[(3, 1)].set_facecolor('#FFB6C1')

    fig.suptitle(
        f'DAX ORB Analysis - {"TraderTom Mode" if args.tradertom else "Parameter Sweep"} - {"With Realistic Costs" if args.realistic else "Without Costs"}',
        fontsize=16, fontweight='bold')

    plt.savefig('dax_orb_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Analysis saved to dax_orb_analysis.png")


# ============================= Main =================================================
def main():
    args = parse_args()

    print("=" * 80)
    print("DAX ORB REALISTIC BACKTEST - With Transaction Costs")
    print("=" * 80)

    # Load data
    df = load_data(args)

    # Split into train/test if not in TraderTom mode
    if not args.tradertom and args.train_pct < 1.0:
        split_idx = int(len(df) * args.train_pct)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        print(
            f"\n✓ Data split: {len(df_train):,} train bars, {len(df_test):,} test bars")
    else:
        df_train = df.copy()
        df_test = None

    # Run backtest
    if args.tradertom:
        # TraderTom fixed parameters
        print(
            f"\n🎯 TRADERTOM MODE: SL=9, TP=6, Opening window={args.opening_window} min")
        print(f"   Realistic costs: {'ON' if args.realistic else 'OFF'}")

        # Run with costs
        trades_with_costs, stats_with_costs = run_realistic_backtest(df_train, args.tz,
            args.pre, args.sess, sl=9.0, tp=6.0, use_costs=True, tradertom_mode=True,
            opening_window=args.opening_window, verbose=args.verbose)

        # Run without costs for comparison
        trades_no_costs, stats_no_costs = run_realistic_backtest(df_train, args.tz,
            args.pre, args.sess, sl=9.0, tp=6.0, use_costs=False, tradertom_mode=True,
            opening_window=args.opening_window, verbose=False)

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON:")
        print("=" * 60)

        print("\n📊 WITHOUT COSTS (Unrealistic):")
        print(f"   Trades: {stats_no_costs['trades']}")
        print(
            f"   Win Rate: {stats_no_costs['winrate']:.1f}% (Need 60.0% for break-even)")
        print(f"   Net PnL: {stats_no_costs['pnl']:.1f} points")
        print(f"   Profit Factor: {stats_no_costs['pf']:.2f}")

        print("\n💰 WITH REALISTIC COSTS:")
        print(f"   Trades: {stats_with_costs['trades']}")
        print(f"   Win Rate: {stats_with_costs['winrate']:.1f}% (Need ~70% with costs)")
        print(f"   Net PnL: {stats_with_costs['pnl']:.1f} points")
        print(f"   Total Costs Paid: {stats_with_costs['total_costs']:.1f} points")
        print(f"   Profit Factor: {stats_with_costs['pf']:.2f}")
        print(f"   Max Drawdown: {stats_with_costs['max_dd']:.1f} points")

        impact = stats_no_costs['pnl'] - stats_with_costs['pnl']
        print(
            f"\n⚠️  COST IMPACT: -{impact:.1f} points ({100 * impact / max(abs(stats_no_costs['pnl']), 1):.1f}% of gross PnL)")

        # Critical assessment
        print("\n" + "=" * 60)
        print("🔍 CRITICAL ASSESSMENT:")
        print("=" * 60)

        if stats_with_costs['winrate'] < 60:
            print("❌ Win rate below break-even threshold (60% without costs)")
        if stats_with_costs['winrate'] < 70:
            print("❌ Win rate below realistic break-even (~70% with costs)")
        if stats_with_costs['pnl'] < 0:
            print("❌ Strategy is UNPROFITABLE with realistic costs")
        if stats_with_costs['trades'] < 100:
            print("⚠️  Sample size too small for statistical significance")
        if abs(stats_with_costs['max_dd']) > 50:
            print("⚠️  Large drawdown risk")

        # Save trades if requested
        if args.save_trades:
            trades_with_costs.to_csv('tradertom_trades_realistic.csv', index=False)
            print(f"\n✓ Trades saved to tradertom_trades_realistic.csv")

        # Create plots
        if args.plot:
            create_comprehensive_plots(df_train, trades_with_costs, stats_with_costs,
                                       args)

    else:
        # Parameter sweep mode
        print(f"\n🔍 PARAMETER SWEEP MODE")
        print(f"   SL range: {args.sl_min}-{args.sl_max} (step {args.sl_step})")
        print(f"   TP range: {args.tp_min}-{args.tp_max} (step {args.tp_step})")
        print(f"   Realistic costs: {'ON' if args.realistic else 'OFF'}")

        # Create parameter combinations
        sl_values = np.arange(args.sl_min, args.sl_max + args.sl_step, args.sl_step)
        tp_values = np.arange(args.tp_min, args.tp_max + args.tp_step, args.tp_step)

        results = []
        best_score = -float('inf')
        best_params = None
        best_trades = None

        total_combos = len(sl_values) * len(tp_values)
        completed = 0

        print(f"\n   Testing {total_combos} combinations...")

        # Test each combination
        with mp.Pool() as pool:
            tasks = [
                (df_train, args.tz, args.pre, args.sess, sl, tp, args.realistic, False,
                 args.opening_window, False) for sl in sl_values for tp in tp_values]

            async_results = [pool.apply_async(run_realistic_backtest, task) for task in
                tasks]

            for i, (sl, tp) in enumerate(
                    [(sl, tp) for sl in sl_values for tp in tp_values]):
                trades_df, stats = async_results[i].get()
                completed += 1

                if completed % 10 == 0 or completed == total_combos:
                    print(
                        f"\r   Progress: {completed}/{total_combos} ({100 * completed / total_combos:.1f}%)",
                        end='')

                if stats['trades'] >= args.min_trades:
                    score = stats['pnl'] / max(abs(stats['max_dd']),
                                               1)  # Risk-adjusted score

                    results.append(
                        {'sl': sl, 'tp': tp, 'rr': tp / sl, 'trades': stats['trades'],
                            'winrate': stats['winrate'], 'pnl': stats['pnl'],
                            'pf': stats['pf'], 'max_dd': stats['max_dd'],
                            'sharpe': stats['sharpe'], 'score': score})

                    if score > best_score:
                        best_score = score
                        best_params = (sl, tp)
                        best_trades = trades_df.copy()

        print()  # New line after progress

        if results:
            # Sort and display results
            results_df = pd.DataFrame(results).sort_values('score', ascending=False)

            print("\n" + "=" * 60)
            print("TOP 10 PARAMETER COMBINATIONS:")
            print("=" * 60)
            print(results_df.head(10).to_string(index=False))

            # Test on out-of-sample data if available
            if df_test is not None and best_params is not None:
                print("\n" + "=" * 60)
                print("OUT-OF-SAMPLE TEST:")
                print("=" * 60)

                test_trades, test_stats = run_realistic_backtest(df_test, args.tz,
                    args.pre, args.sess, sl=best_params[0], tp=best_params[1],
                    use_costs=args.realistic, tradertom_mode=False,
                    opening_window=args.opening_window)

                print(f"Best params: SL={best_params[0]:.1f}, TP={best_params[1]:.1f}")
                print(f"In-sample PnL: {results_df.iloc[0]['pnl']:.1f}")
                print(f"Out-of-sample PnL: {test_stats['pnl']:.1f}")
                print(f"Out-of-sample Win Rate: {test_stats['winrate']:.1f}%")

                if test_stats['pnl'] < 0:
                    print("❌ Strategy FAILED on out-of-sample data!")

            # Save results
            results_df.to_csv('parameter_sweep_results.csv', index=False)
            print(f"\n✓ Results saved to parameter_sweep_results.csv")

            # Create plots for best combination
            if args.plot and best_trades is not None:
                create_comprehensive_plots(df_train, best_trades,
                                           results_df.iloc[0].to_dict(), args)
        else:
            print("\n❌ No parameter combinations met minimum trade requirement")


if __name__ == "__main__":
    main()