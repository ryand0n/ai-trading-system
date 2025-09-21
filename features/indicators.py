import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
from datetime import timedelta

def get_prices(symbol, sd, ed):
    prices = get_data([symbol], pd.date_range(sd - timedelta(90), ed), addSPY=True, colname="Adj Close").drop('SPY', axis=1)
    prices[symbol] = prices[symbol] / prices.iloc[0][symbol]

    return prices

def calc_sma(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    prices = get_prices(symbol, sd, ed)
    prices["sma"] = prices[symbol].rolling(window=n).mean()
    prices["sma_ratio"] = prices[symbol] / prices["sma"]

    return prices.loc[sd:ed]

def calc_momentum(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    prices = get_prices(symbol, sd, ed)
    prices["momentum"] = ((prices[symbol] / prices[symbol].shift(n)) - 1)
    prices["norm_momentum"] = (prices["momentum"] - prices["momentum"].mean()) / prices["momentum"].std()

    return prices.loc[sd:ed]

def calc_bollinger_bands(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    prices = get_prices(symbol, sd, ed)
    prices["sma"] = prices[symbol].rolling(window=n).mean()
    prices["std"] = prices[symbol].rolling(window=n).std()
    prices["upper_band"] = prices["sma"] + (2 * prices["std"])
    prices["lower_band"] = prices["sma"] - (2 * prices["std"])
    prices["bbp"] = (prices[symbol] - prices["lower_band"]) / (prices["upper_band"] - prices["lower_band"])

    return prices.loc[sd:ed]

def calc_stochastic(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    prices = get_data([symbol], pd.date_range(sd - timedelta(90), ed), addSPY=True, colname="Adj Close").drop('SPY', axis=1)
    prices["low"] = prices[symbol].rolling(window=n).min()
    prices["high"] = prices[symbol].rolling(window=n).max()
    prices["k"] = ((prices[symbol] - prices["low"]) / (prices["high"] - prices["low"])) * 100

    return prices.loc[sd:ed]

def calc_ppo(symbol="JPM", short=9, long=26, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    prices = get_data([symbol], pd.date_range(sd - timedelta(90), ed), addSPY=True, colname="Adj Close").drop('SPY', axis=1)

    prices["ema_short"] = prices[symbol].ewm(span=short, adjust=False).mean()
    prices["ema_long"] = prices[symbol].ewm(span=long, adjust=False).mean()
    prices["ppo"] = ((prices["ema_short"] - prices["ema_long"]) / prices["ema_long"])
    prices["signal"] = prices['ppo'].ewm(span=short, adjust=False).mean()

    return prices.loc[sd:ed]

def get_momentum(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    ans = calc_momentum(symbol, n, sd, ed)
    return ans["norm_momentum"]

def get_sma(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    ans = calc_sma(symbol, n, sd, ed)
    return ans["sma_ratio"]

def get_bb(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    ans = calc_bollinger_bands(symbol, n, sd, ed)
    return ans["bbp"]

def get_sto(symbol="JPM", n=1, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    ans = calc_stochastic(symbol, n, sd, ed)
    return ans["k"]

def get_ppo(symbol="JPM", short=9, long=26, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    ans = calc_ppo(symbol, short, long, sd, ed)
    return ans["ppo"]
