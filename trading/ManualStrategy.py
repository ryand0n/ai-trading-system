import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util as ut

from features.indicators import get_momentum, get_sma, get_bb, get_sto, get_prices

class ManualLearner(object):
    """
    A manual learner that can learn (essenitally human coded rules) a trading policy using the same indicators used in StrategyLearner=.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    def __init__(self, verbose=False, impact=0.00, commission=0.00):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        pass

    def rules(
            self,
            symbol="JPM",
            n=20,
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31)
    ):
        """
        Implements a ruleset that determines buy/hold/sell signals based on calculated indicator values
        """
        # Calculate indicator values
        prices = get_prices(symbol, sd, ed).loc[sd:ed]
        momentum = get_momentum(symbol, n, sd, ed).values
        sma = get_sma(symbol, n, sd, ed).values
        bb = get_bb(symbol, n, sd, ed).values
        sto = get_sto(symbol, n, sd, ed).values

        signals = []
        initial_signal = None
        indicators = ["momentum", "sma", "bb", "sto"]

        # Calculate signals using the indicator values
        for i in range(len(prices)):
            # Initialize signal
            signal = 0

            # Momentum signal rule
            if "momentum" in indicators:
                if momentum[i] >= 2:
                    signal += -1
                elif momentum[i] <= -2:
                    signal += 1
                elif momentum[i] >= 1:
                    signal += -0.25
                elif momentum[i] <= -1:
                    signal += 0.25

            # Bollinger bands signal rule
            if "bb" in indicators:
                if bb[i] >= 1:
                    signal += -1
                elif bb[i] <= 0:
                    signal += 1

            # SMA signal rule
            if "sma" in indicators:
                if sma[i] >= 1.2:
                    signal += -1
                elif sma[i] <= .8:
                    signal += 1
                elif sma[i] >= 1.1:
                    signal += -.25
                elif sma[i] <= .9:
                    signal += .25

            # Stochastic indicator signal rule
            if "sto" in indicators:
                if sto[i] >= 80:
                    signal += -.25
                if sto[i] <= 20:
                    signal += .25

            # Save the initial signal value
            if i == 0:
                initial_signal = signal

            # Determine trade signal
            trade = 0
            if signal / initial_signal >= 2:
                trade = 1
            elif signal / initial_signal <= -2:
                trade = -1

            # Add signal to the signals list
            signals.append(trade)

        return signals

    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        # Create a prices df
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop('SPY', axis=1)

        # Calculate trading signals
        num_shares = 0
        trades = []
        signals = self.rules(symbol=symbol, n=20, sd=sd, ed=ed)
        for sig in signals:
            trade = 0
            if sig == 1 and num_shares == 0:
                trade = 1000
                num_shares += 1000
            elif sig == 1 and num_shares == -1000:
                trade = 2000
                num_shares += 2000
            elif sig == -1 and num_shares == 0:
                trade = -1000
                num_shares -= 1000
            elif sig == -1 and num_shares == 1000:
                trade = -2000
                num_shares -= 2000
            trades.append(trade)

        prices["trades"] = trades
        return prices.drop(symbol, axis=1)

    def calc_num_orders(self, x):
        if x == 1000 or x == -1000:
            return 1
        elif x == 2000 or x == -2000:
            return 2
        else:
            return 0

    def compute_portvals(
            self,
            tos_trades,
            symbols="JPM",
            start_date=dt.datetime(2008, 1, 1),
            end_date=dt.datetime(2009, 12, 31),
            start_val=100000,
    ):
        # Create a prices df
        prices = ut.get_data([symbols], pd.date_range(start_date, end_date), addSPY=True, colname="Adj Close").drop('SPY', axis=1)
        prices["cash"] = 1

        # Create a trades df
        trades = prices.copy(deep=True)
        trades[symbols] = tos_trades["trades"].values
        trades["num_orders"] = trades[symbols].apply(lambda trade: self.calc_num_orders(trade))
        trades["impact_fees"] = trades.apply(lambda row: prices.at[row.name, symbols] * row[symbols] * self.impact, axis=1)
        trades["cash"] = trades.apply(lambda row: (prices.at[row.name, symbols] * row[symbols] * -1) - (self.commission * row["num_orders"]) - row["impact_fees"], axis=1)
        trades = trades.drop(['num_orders', 'impact_fees'], axis=1)

        # Create a holdings df
        holdings = trades.copy(deep=True)
        holdings.loc[:, symbols] = 0
        holdings.loc[:, "cash"] = 0

        # Initialize the holdings df with starting portfolio value
        holdings.at[holdings.index[0], "cash"] = start_val

        # Calculate holdings
        prev = 0
        for row in holdings.itertuples():
            if prev == 0:
                for symbol, trade in trades.loc[row.Index].items():
                    holdings.at[row.Index, symbol] = holdings.at[row.Index, symbol] + trades.at[row.Index, symbol]
            else:
                holdings.loc[row.Index] = holdings.loc[prev]
                for symbol, trade in trades.loc[row.Index].items():
                    holdings.at[row.Index, symbol] = holdings.at[row.Index, symbol] + trades.at[row.Index, symbol]
            prev = row.Index

        # Create a values df
        values = prices * holdings

        # Calculate portfolio value for each day
        port_vals = values.sum(axis=1)
        ans = pd.DataFrame({"port_val": port_vals})

        return ans

    def plot(self, benchmark, tos, trades, title, path):
        norm_bench = benchmark["port_val"] / benchmark.iloc[0]["port_val"]
        norm_tos = tos["port_val"] / tos.iloc[0]["port_val"]

        l = 0
        s = 0
        plt.figure(figsize=(12, 5))
        for dt, val in trades["trades"].items():
            if val > 0 and l == 0:
                plt.axvline(x=dt, color='blue', linestyle='--', label='LONG')
                l += 1
            elif val < 0 and s == 0:
                plt.axvline(x=dt, color='black', linestyle=':', label='SHORT')
                s += 1
            elif val > 0:
                plt.axvline(x=dt, color='blue', linestyle='--')
            elif val < 0:
                plt.axvline(x=dt, color='black', linestyle=':')
        plt.plot(norm_bench.index, norm_bench.values, color='purple', label="Benchmark")
        plt.plot(norm_bench.index, norm_tos.values, color='red', label="Manual Strategy")
        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(path)

    def compute_statistics(self, data):
        cr = data.iloc[-1]["port_val"] - data.iloc[0]["port_val"]
        dr = (data["port_val"] / data["port_val"].shift(1)) - 1
        dr.iloc[0] = 0
        mean_dr = np.mean(dr.iloc[1:])
        std_dr = np.std(dr.iloc[1:], ddof=1)

        return cr, std_dr, mean_dr
