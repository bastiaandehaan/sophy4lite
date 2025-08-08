# main.py
import argparse
import logging
from backtest.data_loader import fetch_historical_data as fetch_data
from backtest.backtest import run_backtest, metrics
from strategies import get_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sophy4lite")

def main():
    p = argparse.ArgumentParser(description="Sophy4Lite")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="Run backtest")
    b.add_argument("--strategy", required=True, choices=["bollong", "simple_ob"])
    b.add_argument("--symbol", required=True)
    b.add_argument("--timeframe", default="H1")
    b.add_argument("--days", type=int, default=200)
    b.add_argument("--start", type=str, help="YYYY-MM-DD (optioneel)")

    l = sub.add_parser("live", help="(placeholder) Live trading")
    l.add_argument("--strategy", required=True, choices=["bollong", "simple_ob"])
    l.add_argument("--symbol", required=True)
    l.add_argument("--timeframe", default="H1")

    args = p.parse_args()

    if args.cmd == "backtest":
        # fetch_data uit jouw backtest/data_loader.py ondersteunt start
        df = fetch_data(symbol=args.symbol, timeframe=args.timeframe, days=args.days, end_date=None)
        strat = get_strategy(args.strategy)
        entries, sl, tp = strat.generate_signals(df)
        pf = run_backtest(df, entries, sl, tp, freq=args.timeframe, slippage=0.0002)
        m = metrics(pf)
        logger.info(f"Backtest metrics: {m}")
        try:
            from utils.plotting import plot_equity_and_drawdown, plot_trades_on_price
            plot_equity_and_drawdown(pf, outpath="output/equity_dd.png")
            plot_trades_on_price(pf, outpath="output/trades.png")
            logger.info("Saved plots to output/equity_dd_*.png and output/trades.png")
        except Exception as e:
            logger.warning(f"Plotting skipped: {e}")

    elif args.cmd == "live":
        logger.info("Live loop placeholder. Eerst backtests afronden.")

if __name__ == "__main__":
    main()
