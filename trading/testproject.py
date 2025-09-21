import datetime as dt
import numpy as np

from ManualStrategy import ManualLearner
from experiment1 import exp_one
from experiment2 import exp_two

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "rdon3"

def study_group():
    return "rdon3"

if __name__ == "__main__":
    np.random.seed(904080336)
    # Generate charts for the Manual Strategy

    # In-sample testing
    impact = 0.005
    commission = 9.95
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    title = "Manual Strategy vs Benchmark (In-sample)"
    path = "manual_strategy_is.png"

    ml = ManualLearner(impact=impact, commission=commission)
    trades = ml.testPolicy()

    benchmark = trades.copy(deep=True)
    benchmark.loc[:, "trades"] = 0
    benchmark.iloc[0] = 1000

    ml_portval = ml.compute_portvals(tos_trades=trades)
    bm_portval = ml.compute_portvals(tos_trades=benchmark)

    ml.plot(bm_portval, ml_portval, trades, title, path)

    # print("In-sample")
    # print("Benchmark statistics:")
    # print(ml.compute_statistics(bm_portval))
    # print("Portfolio statistics:")
    # print(ml.compute_statistics(ml_portval))

    # Out-of-sample testing
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    title = "Manual Strategy vs Benchmark (Out-of-sample)"
    path = "manual_strategy_oos.png"

    ml = ManualLearner(impact=impact, commission=commission)
    trades = ml.testPolicy(sd=sd, ed=ed)

    benchmark = trades.copy(deep=True)
    benchmark.loc[:, "trades"] = 0
    benchmark.iloc[0] = 1000

    ml_portval = ml.compute_portvals(tos_trades=trades, start_date=sd, end_date=ed)
    bm_portval = ml.compute_portvals(tos_trades=benchmark, start_date=sd, end_date=ed)

    ml.plot(bm_portval, ml_portval, trades, title, path)

    # print("Out-of-sample")
    # print("Benchmark statistics:")
    # print(ml.compute_statistics(bm_portval))
    # print("Portfolio statistics:")
    # print(ml.compute_statistics(ml_portval))

    # Generate Experiment One charts
    exp_one()

    # Generate Experiment Two charts
    exp_two()
