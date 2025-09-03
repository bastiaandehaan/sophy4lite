#!/usr/bin/env python3
"""
Trade Analysis - Diagnose waarom de strategie verliest
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from backtest.data_loader import fetch_data
from strategies.mtf_confluence import MTFSignals, MTFParams
from backtest.mtf_exec_fast import backtest_mtf_confluence_fast, MTFExecCfg


def analyze_losing_strategy(csv_path: str, session_end: str = "18:30"):
    """Deep dive into why the strategy is losing money"""

    print("=" * 60)
    print("MTF CONFLUENCE STRATEGY ANALYSIS")
    print("=" * 60)

    # 1. Load data
    df = fetch_data(csv_path=csv_path)
    print(f"\n📊 Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # 2. Generate signals to analyze
    params = MTFParams(min_confluence_score=0.65)
    mtf = MTFSignals(params)
    signals = mtf.generate_signals(df, session_end=session_end)

    # Signal statistics
    long_signals = signals['long_entry'].sum()
    short_signals = signals['short_entry'].sum()
    print(f"\n📈 Raw Signals Generated:")
    print(f"  Long entries:  {long_signals}")
    print(f"  Short entries: {short_signals}")
    print(f"  Total:         {long_signals + short_signals}")

    # Score distribution
    print(f"\n📊 Signal Quality Distribution:")
    long_scores = signals.loc[signals['long_entry'], 'long_score']
    short_scores = signals.loc[signals['short_entry'], 'short_score']

    if len(long_scores) > 0:
        print(
            f"  Long scores:  mean={long_scores.mean():.3f}, std={long_scores.std():.3f}")
        print(
            f"               min={long_scores.min():.3f}, max={long_scores.max():.3f}")

    if len(short_scores) > 0:
        print(
            f"  Short scores: mean={short_scores.mean():.3f}, std={short_scores.std():.3f}")
        print(
            f"               min={short_scores.min():.3f}, max={short_scores.max():.3f}")

    # Expected Value analysis
    print(f"\n💰 Expected Value (theoretical):")
    long_evs = signals.loc[signals['long_entry'], 'long_ev']
    short_evs = signals.loc[signals['short_entry'], 'short_ev']

    if len(long_evs) > 0:
        print(
            f"  Long EV:  mean={long_evs.mean():.3f}R, positive={100 * (long_evs > 0).mean():.1f}%")
    if len(short_evs) > 0:
        print(
            f"  Short EV: mean={short_evs.mean():.3f}R, positive={100 * (short_evs > 0).mean():.1f}%")

    # 3. Run backtest for actual results
    cfg = MTFExecCfg(risk_frac=0.01)
    eq, trades, metrics = backtest_mtf_confluence_fast(df, "GER40.cash", params, cfg,
        session_end=session_end, max_trades_per_day=1)

    if len(trades) == 0:
        print("\n❌ No trades executed!")
        return None

    # 4. Trade Analysis
    print(f"\n📋 EXECUTED TRADES: {len(trades)}")
    print("-" * 40)

    # Direction bias
    n_long = (trades['side'] == 'long').sum()
    n_short = (trades['side'] == 'short').sum()
    print(f"\n🎯 Direction Split:")
    print(f"  Longs:  {n_long} ({100 * n_long / len(trades):.1f}%)")
    print(f"  Shorts: {n_short} ({100 * n_short / len(trades):.1f}%)")

    # Win rates by direction
    long_trades = trades[trades['side'] == 'long']
    short_trades = trades[trades['side'] == 'short']

    if len(long_trades) > 0:
        long_wr = 100 * (long_trades['pnl'] > 0).mean()
        long_pnl = long_trades['pnl'].sum()
        print(f"\n📈 Long Performance:")
        print(f"  Win rate: {long_wr:.1f}%")
        print(f"  Total PnL: ${long_pnl:.2f}")
        print(f"  Avg PnL:  ${long_trades['pnl'].mean():.2f}")

    if len(short_trades) > 0:
        short_wr = 100 * (short_trades['pnl'] > 0).mean()
        short_pnl = short_trades['pnl'].sum()
        print(f"\n📉 Short Performance:")
        print(f"  Win rate: {short_wr:.1f}%")
        print(f"  Total PnL: ${short_pnl:.2f}")
        print(f"  Avg PnL:  ${short_trades['pnl'].mean():.2f}")

    # Exit reason analysis
    print(f"\n🎯 Exit Reasons:")
    exit_counts = trades['reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = 100 * count / len(trades)
        avg_pnl = trades[trades['reason'] == reason]['pnl'].mean()
        total_pnl = trades[trades['reason'] == reason]['pnl'].sum()
        print(
            f"  {reason:3}: {count:3} trades ({pct:5.1f}%) | Avg: ${avg_pnl:7.2f} | Total: ${total_pnl:8.2f}")

    # Risk:Reward actual vs theoretical
    print(f"\n📊 Risk:Reward Analysis:")
    actual_rr = trades['tp_pts'] / trades['sl_pts']
    print(f"  Theoretical RR: {params.atr_mult_tp / params.atr_mult_sl:.2f}:1")
    print(f"  Actual avg RR:  {actual_rr.mean():.2f}:1")
    print(f"  RR std dev:     {actual_rr.std():.2f}")

    # Time of day analysis
    trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
    hourly_stats = trades.groupby('hour').agg(
        {'pnl': ['count', 'sum', 'mean'], 'side': lambda x: (x == 'long').mean()})

    print(f"\n⏰ Performance by Entry Hour:")
    print("  Hour | Trades |  Total PnL  | Avg PnL | % Long")
    print("  -----|--------|-------------|---------|-------")
    for hour in sorted(trades['hour'].unique()):
        hour_trades = trades[trades['hour'] == hour]
        n = len(hour_trades)
        total_pnl = hour_trades['pnl'].sum()
        avg_pnl = hour_trades['pnl'].mean()
        pct_long = 100 * (hour_trades['side'] == 'long').mean()
        print(
            f"   {hour:02d}  |   {n:3}  | ${total_pnl:9.2f} | ${avg_pnl:6.2f} | {pct_long:5.1f}%")

    # Worst trades
    print(f"\n💥 Worst 5 Trades:")
    worst = trades.nsmallest(5, 'pnl')[
        ['side', 'entry_time', 'reason', 'pnl', 'sl_pts', 'tp_pts']]
    for _, t in worst.iterrows():
        rr = t['tp_pts'] / t['sl_pts'] if t['sl_pts'] > 0 else 0
        print(
            f"  {t['side']:5} | {t['entry_time']} | {t['reason']:2} | PnL: ${t['pnl']:8.2f} | RR: {rr:.2f}:1")

    # Score vs outcome correlation
    if 'score' in trades.columns:
        trades['win'] = (trades['pnl'] > 0).astype(int)
        score_corr = trades[['score', 'win']].corr().iloc[0, 1]
        print(f"\n📊 Signal Quality Correlation:")
        print(f"  Score vs Win correlation: {score_corr:.3f}")

        # Binned analysis
        trades['score_bin'] = pd.qcut(trades['score'], q=3,
                                      labels=['Low', 'Mid', 'High'])
        bin_stats = trades.groupby('score_bin').agg(
            {'win': 'mean', 'pnl': ['mean', 'sum', 'count']})
        print(f"\n  Score Bin | Win Rate | Avg PnL  | Total PnL | Count")
        print(f"  ----------|----------|----------|-----------|------")
        for bin_name in ['Low', 'Mid', 'High']:
            if bin_name in bin_stats.index:
                wr = 100 * bin_stats.loc[bin_name, ('win', 'mean')]
                avg_pnl = bin_stats.loc[bin_name, ('pnl', 'mean')]
                total_pnl = bin_stats.loc[bin_name, ('pnl', 'sum')]
                count = int(bin_stats.loc[bin_name, ('pnl', 'count')])
                print(
                    f"  {bin_name:9} |  {wr:5.1f}%  | ${avg_pnl:7.2f} | ${total_pnl:8.2f} |  {count:3}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

    # Save detailed trades for further analysis
    trades.to_csv("output/trade_analysis.csv", index=False)
    print("\n📁 Detailed trades saved to: output/trade_analysis.csv")

    return trades, metrics


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/GER40.cash_M1.csv"
    session_end = sys.argv[2] if len(sys.argv) > 2 else "18:30"

    trades, metrics = analyze_losing_strategy(csv, session_end)

    # Recommendations
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS BASED ON ANALYSIS:")
    print("=" * 60)

    if metrics and metrics['winrate_pct'] < 40:
        print("\n1. ❌ Win rate too low ({}%)".format(metrics['winrate_pct']))
        print("   → Increase min_confluence_score to 0.70-0.75")
        print("   → Add trend filter on higher timeframe")

    if metrics and metrics['expectancy_R'] < 0:
        print("\n2. ❌ Negative expectancy ({:.2f}R)".format(metrics['expectancy_R']))
        print("   → Adjust TP multiplier (try 2.0 instead of 2.5)")
        print("   → Tighten SL (try 1.2 instead of 1.5)")

    if trades is not None and (trades['reason'] == 'SL').mean() > 0.65:
        sl_pct = 100 * (trades['reason'] == 'SL').mean()
        print(f"\n3. ❌ Too many stop losses ({sl_pct:.1f}%)")
        print("   → Entry timing issue - try enter_on='close' instead of 'next_open'")
        print("   → Session timing - avoid first/last hour")

    print("\n🔧 Try this configuration:")
    print("""
    params = MTFParams(
        min_confluence_score=0.70,  # Higher threshold
        atr_mult_sl=1.2,            # Tighter stop
        atr_mult_tp=2.0,            # More realistic target
        momentum_threshold=0.7,      # Stronger momentum required
    )

    cfg = MTFExecCfg(
        enter_on='close',           # Better entry timing
        prefer_metric='ev',         # Prioritize positive EV trades
    )
    """)