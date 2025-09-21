import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from ManualStrategy import ManualLearner
from StrategyLearner import StrategyLearner

impact = 0.005
commission = 9.95
symbol = "JPM"
sv = 100000

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "rdon3"

def study_group():
    return "rdon3"

def compute_statistics(data):
    cr = data.iloc[-1]["port_val"] - data.iloc[0]["port_val"]
    dr = (data["port_val"] / data["port_val"].shift(1)) - 1
    dr.iloc[0] = 0
    mean_dr = np.mean(dr.iloc[1:])
    std_dr = np.std(dr.iloc[1:], ddof=1)

    return cr, std_dr, mean_dr

def exp_one():
    # Create in-sample chart
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    sl = StrategyLearner(impact=impact, commission=commission)
    sl.add_evidence(symbol, sd, ed, sv)
    trades = sl.testPolicy(symbol, sd, ed, sv)
    sl_portval = sl.compute_portvals(trades, symbol, sd, ed, sv)
    norm_sl = sl_portval["port_val"] / sl_portval.iloc[0]["port_val"]

    ml = ManualLearner(impact=impact, commission=commission)
    trades = ml.testPolicy(symbol, sd, ed, sv)
    ml_portval = ml.compute_portvals(trades, symbol, sd, ed, sv)
    norm_ml = ml_portval["port_val"] / ml_portval.iloc[0]["port_val"]

    benchmark = trades.copy(deep=True)
    benchmark.loc[:, "trades"] = 0
    benchmark.iloc[0] = 1000
    bm_portval = ml.compute_portvals(benchmark, symbol, sd, ed, sv)
    norm_bm = bm_portval["port_val"] / bm_portval.iloc[0]["port_val"]

    plt.figure(figsize=(12, 5))
    plt.plot(norm_bm.index, norm_bm.values, color='purple', label="Benchmark")
    plt.plot(norm_ml.index, norm_ml.values, color='red', label="Manual Strategy")
    plt.plot(norm_sl.index, norm_sl.values, color='blue', label="Strategy Learner")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Manual Strategy vs Strategy Learner (in-sample)")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp1_is.png")
    plt.clf()

    # Create out-of-sample chart
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    trades = sl.testPolicy(symbol, sd, ed, sv)
    sl_portval = sl.compute_portvals(trades, symbol, sd, ed, sv)
    norm_sl = sl_portval["port_val"] / sl_portval.iloc[0]["port_val"]

    trades = ml.testPolicy(symbol, sd, ed, sv)
    ml_portval = ml.compute_portvals(trades, symbol, sd, ed, sv)
    norm_ml = ml_portval["port_val"] / ml_portval.iloc[0]["port_val"]

    benchmark = trades.copy(deep=True)
    benchmark.loc[:, "trades"] = 0
    benchmark.iloc[0] = 1000
    bm_portval = ml.compute_portvals(benchmark, symbol, sd, ed, sv)
    norm_bm = bm_portval["port_val"] / bm_portval.iloc[0]["port_val"]

    plt.figure(figsize=(12, 5))
    plt.plot(norm_bm.index, norm_bm.values, color='purple', label="Benchmark")
    plt.plot(norm_ml.index, norm_ml.values, color='red', label="Manual Strategy")
    plt.plot(norm_sl.index, norm_sl.values, color='blue', label="Strategy Learner")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Manual Strategy vs Strategy Learner (out-of-sample)")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp1_oos.png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(904080336)
    exp_one()
