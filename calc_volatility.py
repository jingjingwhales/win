from basic_trade_helper import get_historical_price, get_last_4_hours_price, get_1_day_historical_price_ts
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime
from coin_market_data import getMarketDataFromCoinmarketcap
import pandas as pd

def calc_volatility(ticker):
    p = get_1_day_historical_price_ts(ticker)
    log_p = np.log(p)
    log_p_diff = np.diff(log_p)
    av_growth = np.average(log_p_diff)
    variance = np.var(log_p_diff)


    mean_1hr = 60 * av_growth
    var_1hr = variance * 60
    return mean_1hr, var_1hr
    # price_range_2_sigma = 1.96 * math.sqrt(var_1hr)

    # percentile = [0.5, 0.75, 0.90, 0.95]

    # n, bins, patches = plt.hist(x=log_p_diff, bins='auto', color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    #
    # plt.show()
    # print (mean_1hr, var_1hr)
    # print ("95% chance the price change (up or down) within in 1 hr is: {price_range}".format(price_range = price_range_2_sigma))

# for pct in percentile:
#     ppf = -1 * norm.ppf((1 - pct) / 2)
#     print (ppf)
#     price_chg_interval = (av_growth - ppf* math.sqrt(var_1hr), av_growth + ppf * math.sqrt(var_1hr))
#     print (price_chg_interval)

def getFullListOfCryptos(numer_of_crypto_currenty, market_cap_min):
    df_crypto_data = getMarketDataFromCoinmarketcap(numer_of_crypto_currenty=numer_of_crypto_currenty, market_cap_min=market_cap_min,
                                                    sort_method="percent_change_24h")
    data = []
    for index, row in df_crypto_data.iterrows():
        ticker = row["Symbol"]
        market_cap = row["Market_cap"]
        # print("calculating growth rate and volatility of crypto {crypto}, with market cap = {market_cap}".format(
        #     crypto=ticker, market_cap=market_cap))
        binance_ticker = ticker + "USDT"
        try:
            mean_1hr, volatility_1hr = calc_volatility(binance_ticker)
            print("ticker {ticker} is trading on Binance. Currently mean_growth rate is: {growth_rate}; "
                  "variance is : {variance}".format(ticker=binance_ticker, growth_rate=mean_1hr,
                                                    variance=volatility_1hr))
            data.append([binance_ticker, mean_1hr, volatility_1hr, market_cap])
        except Exception as e:
            print(e.message)
    df_vol = pd.DataFrame(data, columns=["ticker", "average_growht_rate", "volatility", "market_cap"])

    df_vol.sort_values(by=["average_growht_rate", "volatility"], inplace=True, ascending=False)
    return df_vol

if __name__ == "__main__":
    df_vol = getFullListOfCryptos(5000, 1000000000)
    today = datetime.today().strftime("%Y%m%d%H%M")
    df_vol.to_csv("daily_volatility_{today}.csv".format(today=today))

