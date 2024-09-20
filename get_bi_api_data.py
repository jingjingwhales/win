from basic_trade_helper import bi_client_us
from datetime import datetime, timezone
import json
import pandas as pd
import os
from coin_market_data import getMarketDataFromCoinmarketcap
import multiprocessing
from collections import defaultdict


client = bi_client_us()
# HISTORY_START_DATE = "2 years ago UTC"
HISTORY_START_DATE = "2016-12-26"
HISTORY_END_DATE = "2024-03-23"
NEW_RUN_START_DATE = "2024-03-24"
DATA_DIR = "./ticker_data"
counter = None

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
def get_data(ticker, start_timestamp, end_timestamp):
    binance_ticker = ticker + "USDT"
    data = client.get_historical_klines(symbol=binance_ticker, interval=client.KLINE_INTERVAL_1DAY, start_str=start_timestamp, end_str=end_timestamp)
    return data

def get_crypto_list(file_name):
    if not os.path.isfile(file_name):
        df_crypto_data = getMarketDataFromCoinmarketcap(numer_of_crypto_currenty=5000,
                                                        market_cap_min=20000000,
                                                        sort_method="market_cap")
        df = pd.DataFrame(data={"Symbol": df_crypto_data["Symbol"].tolist()})
        df.to_csv(file_name, index=False)
    else:
        df = pd.read_csv(file_name)
    return df

def get_historical_data(ticker, start_timestamp, end_timestamp):
    global counter
    data = {}
    with counter.get_lock():
        counter.value += 1
    if counter.value % 10 == 0:
        print("finished {count} tickers".format(count=counter.value))
    try:
        ticker_json_file = DATA_DIR + "/{ticker}.json".format(ticker=ticker)
        if os.path.isfile(ticker_json_file):
            with open(ticker_json_file, 'r') as openfile:
                data = json.load(openfile)
        ticker_hist = get_data(ticker, start_timestamp, end_timestamp)
        if data:
            for dp in ticker_hist:
                data[ticker].append(dp)
        else:
            data[ticker] = ticker_hist
        if len(data[ticker]) > 0:
            json_data = json.dumps(data, indent=4)
            with open(DATA_DIR + "/{ticker}.json".format(ticker=ticker), "w") as outfile:
                outfile.write(json_data)
    except Exception as e:
        pass
        # print ("can not get {ticker} data".format(ticker=ticker))

if __name__ == "__main__":
    counter = multiprocessing.Value('i', 0)
    df_list = get_crypto_list("crypto_list.csv")
    ticker_list = [item for item in df_list["Symbol"].tolist() if "USD" not in item]
    now = datetime.now(timezone.utc)
    whole_hr_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end_timestamp = int(whole_hr_dt.timestamp() * 1000 - 1)
    # executions = [(ticker, HISTORY_START_DATE, HISTORY_END_DATE) for ticker in ticker_list]
    executions = [(ticker, NEW_RUN_START_DATE, end_timestamp) for ticker in ticker_list]
    print("total amount of tickers = {ticker_amount}".format(ticker_amount=len(ticker_list)))
    # get_historical_data("SOL", HISTORY_START_DATE, HISTORY_END_DATE)
    with multiprocessing.Pool(processes=48, initializer=init, initargs=(counter,)) as pool:
        pool.starmap(get_historical_data, executions)
    print ("done")