# Quick debug script - sla op als test_signals.py
import pandas as pd
from backtest.data_loader import fetch_data
from strategies.mtf_confluence import MTFSignals, MTFParams

# Load data
df = fetch_data(csv_path="data/GER40.cash_M1.csv")
print(f"Loaded {len(df)} bars")

# Generate signals
mtf = MTFSignals(MTFParams(min_confluence_score=0.5))  # Lager voor meer signalen
signals = mtf.generate_signals(df)

# Check signals
long_signals = signals['long_entry'].sum()
short_signals = signals['short_entry'].sum()

print(f"Long signals: {long_signals}")
print(f"Short signals: {short_signals}")

if long_signals > 0:
    print("\nFirst 5 long signals:")
    print(signals[signals['long_entry']].head())