import argparse
import datetime as dt
import numpy as np

from trading.ManualStrategy import ManualLearner
from trading.StrategyLearner import StrategyLearner

def main():
    parser = argparse.ArgumentParser(description="Run Manual or Strategy Learner for trading.")
    parser.add_argument('--learner', choices=['manual', 'strategy'], required=True, help='Choose which learner to use: manual or strategy')
    parser.add_argument('--symbol', type=str, default='JPM', help='Stock symbol to trade')
    parser.add_argument('--start', type=str, default='2008-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2009-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--sv', type=int, default=100000, help='Starting portfolio value')
    parser.add_argument('--impact', type=float, default=0.005, help='Market impact')
    parser.add_argument('--commission', type=float, default=9.95, help='Commission per trade')
    parser.add_argument('--future_start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--future_end', type=str, default='2011-12-31', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Parse dates
    sd = dt.datetime.strptime(args.start, "%Y-%m-%d")
    ed = dt.datetime.strptime(args.end, "%Y-%m-%d")

    future_sd = dt.datetime.strptime(args.future_start, "%Y-%m-%d")
    future_ed = dt.datetime.strptime(args.future_end, "%Y-%m-%d")

    if args.learner == 'manual':
        learner = ManualLearner(impact=args.impact, commission=args.commission)
    else:
        learner = StrategyLearner(impact=args.impact, commission=args.commission)

    learner.add_evidence(args.symbol, sd, ed, args.sv)
    trades = learner.testPolicy(args.symbol, sd, ed, args.sv)
    port_val = learner.compute_portvals(trades, args.symbol, sd, ed, args.sv)

    future_trades = learner.testPolicy(args.symbol, future_sd, future_ed, args.sv)
    future_port_val = learner.compute_portvals(future_trades, args.symbol, future_sd, future_ed, args.sv)

    print("Recommended Trades:")
    print(trades)
    print("\nIn-sample Portfolio Values:")
    print(port_val)
    print("Recommended Future Trades:")
    print(future_trades)
    print("\nOut-of-sample Portfolio Values:")
    print(future_port_val)

if __name__ == "__main__":
    main()
