import argparse
import logging
from data.data_loader import fetch_data
from backtest import run_backtest
from strategies import get_strategy
from live.live_trading import start_live_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sophy4lite")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sophy4Lite")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--strategy", required=True, choices=["bollong", "simple_ob"], help="Strategy name")
    backtest_parser.add_argument("--symbol", required=True, help="Trading symbol")
    backtest_parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    backtest_parser.add_argument("--days", type=int, default=200, help="How many days of data to fetch")
    backtest_parser.add_argument("--start", type=str, help="Optional start date (format: YYYY-MM-DD)")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("--strategy", required=True, choices=["bollong", "simple_ob"], help="Strategy name")
    live_parser.add_argument("--symbol", required=True, help="Trading symbol")
    live_parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")

    args = parser.parse_args()

    if args.command == "backtest":
        df = fetch_data(symbol=args.symbol, timeframe=args.timeframe, days=args.days, start=args.start)
        strategy = get_strategy(args.strategy)
        portfolio = strategy.run(df)
        run_backtest(df, portfolio.entries, portfolio.sl_stop, portfolio.tp_stop)
    elif args.command == "live":
        start_live_loop(args.strategy, args.symbol, args.timeframe)
    else:
        parser.print_help()
