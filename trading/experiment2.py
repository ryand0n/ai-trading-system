import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from StrategyLearner import StrategyLearner

colors = ["purple", "blue", "red"]
impacts = [0.00, 0.005, 0.02]
commission = 9.95
sd = dt.datetime(2008, 1, 1)
ed = dt.datetime(2009, 12, 31)
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
    cr = data.iloc[-1] - data.iloc[0]
    dr = (data / data.shift(1)) - 1
    dr.iloc[0] = 0
    mean_dr = np.mean(dr.iloc[1:])
    std_dr = np.std(dr.iloc[1:], ddof=1)
    sharpe_ratio = np.sqrt(252) * (mean_dr / std_dr)

    return cr, std_dr, mean_dr, sharpe_ratio

def exp_two():
    portvals = []
    for impact in impacts:
        sl = StrategyLearner(impact=impact, commission=commission)
        sl.add_evidence(symbol, sd, ed, sv)
        trades = sl.testPolicy(symbol, sd, ed, sv)
        sl_portval = sl.compute_portvals(trades, symbol, sd, ed, sv)
        norm_sl = sl_portval["port_val"] / sl_portval.iloc[0]["port_val"]
        portvals.append(norm_sl)

    portstats = []
    for port in portvals:
        portstats.append(compute_statistics(port))

    plt.figure(figsize=(12, 5))
    for i in range(len(portvals)):
        plt.plot(portvals[i].index, portvals[i].values, color=colors[i], label=f"Impact: {impacts[i]}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("The Effect of Market Impact on Strategy Learner Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp2_portval.png")
    plt.clf()

    plt.figure(figsize=(12, 5))
    impact = [str(i) for i in impacts]
    sr = [i[-1] for i in portstats]
    plt.bar(impact, sr)
    plt.xlabel("Impact")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio across Different Impact Values")
    plt.grid(True)
    plt.savefig("exp2_sharpe.png")
    plt.clf()

    plt.figure(figsize=(12, 5))
    impact = [str(i) for i in impacts]
    cr = [i[0] for i in portstats]
    plt.bar(impact, cr)
    plt.xlabel("Impact")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Return across Different Impact Values")
    plt.grid(True)
    plt.savefig("exp2_cr.png")
    plt.clf()

if __name__ == "__main__":
    np.random.seed(904080336)
    exp_two()
