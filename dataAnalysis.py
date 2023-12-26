from basic_trade_helper import bi_client_us
# from binance.client import Client
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from coin_market_data import getMarketDataFromCoinmarketcap
import os
import logging
from datetime import timezone
TEST = True
TEST_BATCH = 5

client = bi_client_us()
# client = Client()

def get_crypto_list(file_name):
    if not os.path.isfile(file_name):
        df_crypto_data = getMarketDataFromCoinmarketcap(numer_of_crypto_currenty=5000,
                                                        market_cap_min=5000000,
                                                        sort_method="market_cap")
        df = pd.DataFrame(data={"Symbol": df_crypto_data["Symbol"].tolist()})
        df.to_csv(file_name, index=False)
    else:
        df = pd.read_csv(file_name)
    return df

def get_all_crytpo_data(start_timestamp, end_timestamp, test=False):
    crypto_counter = 0
    df_list = get_crypto_list("crypto_list.csv")
    df_all_crypto_data = pd.DataFrame()
    for index, row in df_list.iterrows():
        if test and crypto_counter >= TEST_BATCH:
            break
        ticker = row["Symbol"]
        if "USD" in ticker:
            continue
        try:
            binance_ticker = ticker + "USDT"
            data = client.get_historical_klines(symbol=binance_ticker, interval=client.KLINE_INTERVAL_1DAY,
                                                start_str=start_timestamp, end_str=end_timestamp)
            data = [item for item in data if
                    item[6] <= int(end_timestamp)]  # in case the data end at timestamp greater than desired
            time_stamp = [datetime.strftime(datetime.fromtimestamp(item[6]//1000), "%Y%m%d%H%M%S") for item in data]
            data = [[float(x) for x in item] for item in data]
            vol = [round(item[5],4) for item in data]
            vol_in_quote_value = [round(item[7],2) for item in data]
            price_close = [round(item[4],4) for item in data]
            price_chg_daily = [np.nan] * len(price_close)
            for i in range(1, len(price_close)):
                price_chg_daily[i] = (price_close[i] / price_close[i-1] - 1) * 100
            currency_data = {"date": time_stamp,
                             "price_close": price_close,
                             "price_chg_daily": price_chg_daily,
                             "volume": vol,
                             "volume_in_quote_value": vol_in_quote_value,
                             "Symbol": ticker
                             }
            df_tmp = pd.DataFrame(currency_data)

            # df_tmp["Symbol"] = ticker
            # df_all_crypto_data = pd.concat([df_all_crypto_data, df_tmp])
            crypto_counter += 1
        except Exception as e:
            # print(e.message)
            print (e)
    print ("Successfully got all crypto data...")
    return df_all_crypto_data

if __name__ == "__main__":
    now = datetime.utcnow()
    whole_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end_timestamp = int(whole_dt.timestamp() * 1000 - 1)
    get_all_crytpo_data = get_all_crytpo_data("3 months ago UTC", str(end_timestamp), TEST)
    print ("done")