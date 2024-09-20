import pandas as pd
import gspread
from basic_trade_helper import get_historical_price, get_1_day_historical_price_ts,bi_client_us
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from datetime import timezone
from scipy.stats import norm
from collections import defaultdict


client = bi_client_us()
def get_ts_his_price_data(ticker, time_window="1 months ago UTC", freq=client.KLINE_INTERVAL_1DAY):
    binance_ticker = ticker + "USDT"
    now = datetime.utcnow()
    whole_dt = datetime(now.year, now.month, now.day, now.hour, tzinfo=timezone.utc)
    end_timestamp = int(whole_dt.timestamp() * 1000 - 1)
    data = client.get_historical_klines(binance_ticker, interval=freq, start_str=time_window, end_str=str(end_timestamp))
    oc_price = [(item[1], item[4]) for item in data if item[6] <= int(end_timestamp)]
    return oc_price
def calc_volatility_ndiffs(prices, diffs=1):
    p_close = [float(val[1]) for val in prices]
    log_p = np.log(p_close)
    log_p_roll = np.roll(log_p, -diffs)
    log_p_roll_diff = log_p_roll - log_p
    av_growth = np.average(log_p_roll_diff[0:len(log_p)-diffs])
    stdev = np.std(log_p_roll_diff)

    return av_growth, stdev, p_close[-1]

def calc_prob_under_target_price(mean, stdev, target):
    ##the price follow the log normal distribution
    normalized_val = (np.log(target) - (mean - stdev ** 2 / 2)) / stdev
    return norm.cdf(normalized_val)

if __name__ == "__main__":
    # crypto_list = ["BTC", "ETH", "ICP", "DOGE", "LTC", "BNB", "ADA", "AGIX", "TVK", "DYDX", "SOL", "AVAX"]
    target_crypto_price = {"BTC": [50000 + 50 * i for i in range(1001)],
                           "ETH": [2000 + 10 * i for i in range(151)],
                           "LTC": [50 + 0.5 * i for i in range(101)],
                           "DOGE": [0.075 + 0.0005 * i for i in range(100)],
                           "SOL": [110 + 1 * i for i in range(101)]
                           }
    #cvalculate how many hrs the contract ends, always at 4:00am the next day
    now = datetime.now()
    target_end_datetime_hr = 4
    horizon_grid_days = [1, 2, 3, 4, 5, 6, 7]
    if now.hour < 4:
        horizon_grid_days = [0] + horizon_grid_days
    horizon_grid_end_dates = [now + timedelta(days=diff) for diff in horizon_grid_days]
    horizon_grid_target_datetime = [datetime(dt.year, dt.month, dt.day, target_end_datetime_hr) for dt in horizon_grid_end_dates]
    time_horizons_in_hrs = [int((dt - now).total_seconds() // 3600 + 1) for dt in horizon_grid_target_datetime]

    results = defaultdict(lambda: defaultdict(lambda: []))
    for ticker in target_crypto_price:
        prices = get_ts_his_price_data(ticker, "1 months ago UTC", client.KLINE_INTERVAL_1HOUR)
        p_now = float(prices[-1][1])
        for K in target_crypto_price[ticker]:
            target = K / p_now
            for horizon in time_horizons_in_hrs:
                mu, sigma, p_now = calc_volatility_ndiffs(prices, horizon)
                results[ticker][horizon].append(calc_prob_under_target_price(mu, sigma, target))
    gc = gspread.oauth()
    now = datetime.now()
    yyyymmdd = now.strftime("%y-%m-%d-%H-%M-%S")
    sh = gc.create("strike_price_prob_{yyyymmdd}".format(yyyymmdd = yyyymmdd))
    for ticker in target_crypto_price:
        wsh = sh.add_worksheet(title=ticker, rows=len(target_crypto_price[ticker]), cols=len(horizon_grid_target_datetime))
        data = {"Strike": target_crypto_price[ticker]}
        for i in range(len(time_horizons_in_hrs)):
            dt = horizon_grid_target_datetime[i]
            data["{dt}".format(dt=dt.strftime("%Y/%m/%d-%H"))] = results[ticker][time_horizons_in_hrs[i]]
        df = pd.DataFrame(data = data)
        wsh.update([df.columns.values.tolist()] + df.values.tolist())
        data.clear()
    print ("done")
