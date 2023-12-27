import requests, zipfile, os
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing
from coin_market_data import getMarketDataFromCoinmarketcap
TEST = False
TEST_BATCH = 24
counter = None
##TO-DO: need to use share memory https://www.geeksforgeeks.org/multiprocessing-python-set-2/
BASE_URL = "https://data.binance.vision"

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args

def get_crypto_list(file_name):
    if not os.path.isfile(file_name):
        df_crypto_data = getMarketDataFromCoinmarketcap(numer_of_crypto_currenty=5000,
                                                        market_cap_min=10000000,
                                                        sort_method="market_cap")
        df = pd.DataFrame(data={"Symbol": df_crypto_data["Symbol"].tolist()})
        df.to_csv(file_name, index=False)
    else:
        df = pd.read_csv(file_name)
    return df

def getBIDailyPricingData(ticker, date, freq='1d'):
    url = BASE_URL + "/data/spot/daily/klines/{ticker}USDT/{freq}/{ticker}USDT-{freq}-{date}.zip"\
           .format(ticker=ticker, freq=freq, date=date)
    columns = ["open_time",
               "open_price",
               "high_price",
               "low_price",
               "close_price",
               "volume",
               "close_time",
               "quote_asset_volume",
               "numner_of_trades",
               "taker_buy_base_volume",
               "taker_buy_quote_volume",
               "ignore"]
    df_oneday = pd.read_csv(url, compression='zip', header=None, names=columns)
    df_oneday["date"] = int(date.replace('-', ''))
    df_oneday["symbol"] = ticker
    return df_oneday[["symbol", "date"] + columns]

def getBIAllDatesPricingData(ticker, from_date, end_date, freq='1d'):
    global counter
    # print("getting {ticker}".format(ticker=ticker))
    df_all = []
    from_date_datetime = datetime.strptime(from_date, "%Y%m%d")
    end_date_datetime = datetime.strptime(end_date, "%Y%m%d")
    day = from_date_datetime + timedelta(days=1)
    while day <= end_date_datetime:
        day_str = datetime.strftime(day, "%Y-%m-%d")
        try:
            df_tmp = getBIDailyPricingData(ticker, day_str, freq)
            df_all.append(df_tmp.copy())
        except Exception as e:
            pass
            # print ("can not get {ticker} at {date} due to: ".format(ticker=ticker, date=day_str), e)
        day = day + timedelta(days=1)
    # print ("getting {ticker} successfully".format(ticker=ticker))
    with counter.get_lock():
        counter.value += 1
    # if counter.value % 10 == 0:
    print ("finished {count} tickers".format(count=counter.value))
    if df_all:
        return pd.concat(df_all)
    return pd.DataFrame()

# def getBIAllCryptoPricingData(ticker):
#     # global df
#     print ("dealing with {ticker}".format(ticker=ticker))
#     df_tmp = getBIAllDatesPricingData(ticker, "20231201", "20231224", '1d')
#     if not df_tmp.empty:
#         print ("appending {ticker} data".format(ticker=ticker))
#         df = pd.concat([df, df_tmp], ignore_index=True)
#         print (df)
    # print (df_tmp)

if __name__ == "__main__":
    counter = multiprocessing.Value('i', 0)
    df_list = get_crypto_list("crypto_list.csv")
    ticker_list = df_list["Symbol"].to_list()
    ticker_list = [item for item in ticker_list if "USD" not in item]
    START_DATE = "20151224"
    END_DATE = "20191224"
    FREQ = "1d"
    if TEST:
        ticker_list = ticker_list[:TEST_BATCH]
    executions = [(ticker, START_DATE, END_DATE, FREQ) for ticker in ticker_list]
    print ("total amount of tickers = {ticker_amount}".format(ticker_amount=len(ticker_list)))
    with multiprocessing.Pool(processes=72, initializer=init, initargs=(counter, )) as pool:
        df_res = pool.starmap(getBIAllDatesPricingData, executions)
    df_all = pd.concat(df_res, axis=0)
    df_all.to_csv("./all_data_{start}_{end}.csv".format(start=START_DATE, end=END_DATE), index=False)
