# Sophy4 Lite

A minimal, FTMO-oriented trading framework using your existing **Bollong** and **Simple Order Block** strategies.
Focus: simplicity, stability, and strict FTMO guardrails.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py backtest --strategy bollong --symbol GER40.cash --timeframe H1 --days 365
python main.py live --strategy bollong --symbol GER40.cash --timeframe H1
```

## FTMO guardrails (built-in)
- Max daily loss: 5% of equity (blocks new trades for that day)
- Max total loss: 10% from starting balance (stops trading)
- Risk per trade: default 1% (configurable)
- Stop after 2 consecutive losing trades (daily)
