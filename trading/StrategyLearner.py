import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import datetime as dt
import pandas as pd
import util as ut

from datetime import timedelta
from features.indicators import get_momentum, get_sma, get_bb, get_sto
from models.RTLearner import RTLearner
from models.BagLearner import BagLearner
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 

    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        self.n = None
        self.model = BagLearner(learner=RTLearner, kwargs={"leaf_size":1}, bags=10, boost=False, verbose=False)

    def calc_y(self, x, impact_fee, boundary=0.0):
        # Determine the trading signal based on return and impact fee
        if x > boundary + impact_fee:
            return 1
        elif x < -boundary - impact_fee:
            return -1
        else:
            return 0
  		  	   		 	 	 			  		 			     			  	 
    def create_training_data(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        n=15,
        boundary=1
    ):
        # Calculate the indicator values
        self.n = n
        prices = ut.get_data([symbol], pd.date_range(sd, ed + timedelta(n * 2)), addSPY=True, colname="Adj Close").drop('SPY', axis=1)
        momentum = get_momentum(symbol, n, sd, ed)
        sma = get_sma(symbol, n, sd, ed)
        bb = get_bb(symbol, n, sd, ed)
        sto = get_sto(symbol, n, sd, ed)

        # Create a training dataset
        data = pd.concat([prices, momentum.rename("momentum"), sma.rename("sma"), bb.rename("bb"), sto.rename("sto")], axis=1).loc[sd:ed]
        data = data.reset_index(drop=False).rename(columns={"index": "date"})
        prices = prices.reset_index(drop=False).rename(columns={"index": "date"})
        data["future_price"] = data.apply(lambda row: prices.at[row.name + n, symbol], axis=1)
        data["impact_fee"] = data["future_price"] * self.impact
        data["return"] = data["future_price"] - data[symbol]
        data["signal"] = data.apply(lambda x: self.calc_y(x["return"], x["impact_fee"], boundary), axis=1)
        data = data.set_index("date")
        data.index.name = None

        return data[["momentum", "sma", "bb", "sto", "signal"]]

    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
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
        # Generate the training data
        data = self.create_training_data(symbol, sd, ed, 3, 0.3)
        data = data.to_numpy()
        X_train = data[:, :-1]
        y_train = data[:, -1]

        # Train the bag learner
        self.model.add_evidence(X_train, y_train)

    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=100000,
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
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
        # Calculate the indicators for the input symbol
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop('SPY', axis=1)
        momentum = get_momentum(symbol, self.n, sd, ed)
        sma = get_sma(symbol, self.n, sd, ed)
        bb = get_bb(symbol, self.n, sd, ed)
        sto = get_sto(symbol, self.n, sd, ed)
        data = pd.concat([prices, momentum.rename("momentum"), sma.rename("sma"), bb.rename("bb"), sto.rename("sto")], axis=1).loc[sd:ed]
        data = data[["momentum", "sma", "bb", "sto"]]
        data = data.to_numpy()

        # Get the predicted signals
        signals = self.model.query(data)
        signals = np.clip(np.round(signals), -1, 1)

        # Determine optimal trades based on predicted signals
        trades = []
        prev_sig = None
        for i in range(len(signals)):
            if i % self.n == 0 or i == len(signals) - 1:
                trade = 0
                sig = signals[i]

                # Make a trade depending on the previous trade
                if prev_sig == 1:
                    trade -= 1000
                elif prev_sig == -1:
                    trade += 1000

                # Determine future trade
                if sig == 1:
                    trade += 1000
                elif sig == -1:
                    trade -= 1000

                trades.append(trade)
                prev_sig = sig
            else:
                trades.append(0)

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
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":
    np.random.seed(904080336)
    impact = 0.005
    commission = 9.95
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    sl = StrategyLearner(impact=impact, commission=commission)
    sl.add_evidence(symbol, sd, ed, sv)
    trades = sl.testPolicy(symbol, sd, ed, sv)
    port_val = sl.compute_portvals(trades, symbol, sd, ed, sv)

    print(port_val)

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000

    trades = sl.testPolicy(symbol, sd, ed, sv)
    print(trades)

    port_val = sl.compute_portvals(trades, symbol, sd, ed, sv)

    print(port_val)
