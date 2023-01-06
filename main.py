import plotly.express as px
import pandas as pd
import numpy as np
import random
from scipy.stats import skewnorm
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


# graphs all columns for x, uses df.index as y
def graph_stock(df, title):
  # time_column is a string
    pd.options.plotting.backend = "plotly"
  #fig = px.line(df, x='time', y=['price', 'price_2'])
    fig = px.line(df, x=df.index, y=df.columns, title=title)
    fig.show()


def graph_OCHL(df_OCHL, title):
  #fig_1 = px.line(df, x=df.index, y=df.columns, title=title)
    fig_2 = go.Figure(data=go.Ohlc(x=df_OCHL.index,
                                   open=df_OCHL['Open'],
                                   high=df_OCHL['High'],
                                   low=df_OCHL['Low'],
                                   close=df_OCHL['Close']))
    fig_2.update(layout_xaxis_rangeslider_visible=False)
    fig_2.show()


def simulate_stock(initial_price, drift, volatility, trend, days):
    def create_pdf(sd, mean, alfa):

        x = skewnorm.rvs(alfa, size=1000000)

        def calc(k, sd, mean):
            return (k*sd)+mean
        x = calc(x, sd, mean)  # standard distribution
        # graph pdf
        # pd.DataFrame(x).hist(bins=100)
        # pick one random number from the distribution
        # formally I would use cdf, but I just have to pick randomly from  the 1000000 samples
        # np.random.choice(x)
        return x

    def create_empty_df(days):
        # create an empty DataFrame with the day
        empty = pd.DatetimeIndex(
            pd.date_range("2020-01-01", periods=days, freq="D")
        )
        empty = pd.DataFrame(empty)
        # hours ,minutes and seconds are cut
        empty
        # hours ,minutes and seconds are cut
        empty.index = [str(x)[0:empty.shape[0]] for x in list(empty.pop(0))]
        empty
        # final dataset con values
        stock = pd.DataFrame([x for x in range(0, empty.shape[0])])
        stock.index = empty.index
        return stock
    # skeleton
    stock = create_empty_df(days)
    # initial price
    stock[0][0] = initial_price
    # create entire stock DataFrame
    x = create_pdf(volatility, drift, trend)
    for _ in range(1, stock.shape[0]):
        stock.iloc[_] = stock.iloc[_-1]*(1+np.random.choice(x))

    stock.index = pd.DatetimeIndex(stock.index)
    return stock


def OCHL(group_values):
    min_ = min(group_values)
    max_ = max(group_values)
    range = max_ - min_
    open = min_+range*random.random()
    close = min_+range*random.random()
    return min_, max_, open, close


df = simulate_stock(1000, 0, 0.01, 0, 8760)


df_ = list()
# df.groupby(np.arange(len(df))//24).apply(OCHL) not working
# that would be the correct way, but i have to create a new df from 0
for a, b in df.groupby(np.arange(len(df))//24):
    group_values = np.array(b.values).flatten()
    low, high, open, close = OCHL(group_values)
    df_.append([low, high, open, close])
#
df_OCHL = pd.DataFrame(df_, index=pd.Series(pd.date_range(
    "2020-01-01", periods=365, freq="D")), columns=['Low', 'High', 'Open', 'Close'])
# graph
graph_stock(df, "")
fig = go.Figure(data=go.Ohlc(x=df_OCHL.index,
                             open=df_OCHL['Open'],
                             high=df_OCHL['High'],
                             low=df_OCHL['Low'],
                             close=df_OCHL['Close']))
fig.update(layout_xaxis_rangeslider_visible=False)
fig.show()


class MySMAStrategy(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 1)
        self.ma2 = self.I(SMA, price, 2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


backtest = Backtest(df_OCHL, MySMAStrategy,
                    commission=.002, exclusive_orders=True)
stats = backtest.run()
backtest.plot()

print(stats)
