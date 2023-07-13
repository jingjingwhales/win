from basic_trade_helper import get_historical_price, get_1_day_historical_price_ts,bi_client_us
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime
from datetime import timezone
from coin_market_data import getMarketDataFromCoinmarketcap
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.optimize import linprog

client = bi_client_us()
def get_one_year_his_price_data(ticker):
    binance_ticker = ticker + "USDT"
    now = datetime.utcnow()
    whole_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end_timestamp = int(whole_dt.timestamp() * 1000 - 1)
    data = client.get_historical_klines(binance_ticker, interval=client.KLINE_INTERVAL_4HOUR, start_str="2 months ago UTC", end_str=str(end_timestamp))
    oc_price = [(item[1], item[4]) for item in data if item[6] <= int(end_timestamp)]
    return oc_price

def get_df_for_list(ticker_list):
    list_required_lentgh = 1
    data = defaultdict(lambda: [])
    for ticker in ticker_list:
        price_lst = get_one_year_his_price_data(ticker)
        if len(price_lst) < list_required_lentgh:
            raise Exception("ticker {ticker} has history price record less than one Year! Please chose a different combination".format(ticker=ticker))
        for p in price_lst:
            data[ticker].append((float(p[1])/float(p[0]) - 1))
    max_len = 0
    for ticker in data:
        max_len = max(max_len, len(data[ticker]))
    for ticker in data:
        if len(data[ticker]) < max_len:
            data[ticker] = [np.nan] * (max_len - len(data[ticker])) + data[ticker]
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    return df
def calc_volatility(ticker):
    p = get_1_day_historical_price_ts(ticker)
    log_p = np.log(p)
    log_p_diff = np.diff(log_p)
    av_growth = np.average(log_p_diff)
    variance = np.var(log_p_diff)


    mean_1hr = 60 * av_growth
    var_1hr = math.sqrt(variance * 60)
    return mean_1hr, var_1hr

def find_optimal_weights_given_ret(mean_ret, cov_matrix, target_ret):
    beta = [1] * mean_ret.shape[0]
    def optimize(func, beta, exp_ret, cov, target_return):
        opt_bounds = Bounds(0, np.inf)
        opt_constraints = (
                           {'type': 'eq',
                            'fun': lambda beta: target_return - np.dot(beta.T, exp_ret)})
        # optimal_weights = minimize(func, beta,
        #                            args=(exp_ret, cov),
        #                            method='SLSQP',
        #                            bounds=opt_bounds,
        #                            constraints=opt_constraints)
        optimal_sol = minimize(func, beta,
                                   args=(cov),
                                   method='SLSQP',
                                   bounds=opt_bounds,
                                   constraints=opt_constraints)
        return optimal_sol


    # def ret_risk(beta, exp_ret, cov):
    #     return -((beta.T @ exp_ret) / (beta.T @ cov @ beta) ** 0.5)

    def ret_risk(beta, cov):
        return (beta.T @ cov @ beta) ** 0.5

    sol = optimize(ret_risk, beta, mean_ret, cov_matrix,
                 target_return=target_ret)
    return sol

def find_sharp_boundary(mean_ret, cov_matrix, target_ret_list):
    res = []
    for target_ret in target_ret_list:
        sol = find_optimal_weights_given_ret(mean_ret, cov_matrix, target_ret)
        # weight = sol['x']
        sigma = sol.fun
        res.append([target_ret, sigma])
    return res

if __name__ == "__main__":
    crypto_list = ["BTC", "ICP", "DOGE", "LTC", "BNB", "ADA", "AGIX", "TVK", "DYDX", "SOL", "AVAX"]
    df = get_df_for_list(crypto_list)
    time_frame = len(df)
    cov_matrix = df.cov()
    mean_ret = df.mean()
    freq = 24 / 4 * 30
    target_ret = 0.10 / freq
    sol = find_optimal_weights_given_ret(mean_ret, cov_matrix, target_ret)
    target_wts = sol['x']
    min_std = (target_wts.T @ cov_matrix @ target_wts)**0.5 * freq

    print ("minimal std dev for given target return {target_ret} is: {min_std}. Need the weights as following {target_wts}".\
           format(target_ret=target_ret * freq, min_std=min_std, target_wts=target_wts))

    target_ret_list = [0.0001 * i for i in range(1, 21)]
    boundary = find_sharp_boundary(mean_ret, cov_matrix, target_ret_list)
    df_bd = pd.DataFrame(boundary, columns=["expeted_ret", "std"])
    for col in df_bd.columns:
        df_bd[col] = df_bd[col] * freq
    df_bd["sharp_ratio"] = df_bd["expeted_ret"] / df_bd["std"]
    df_bd.plot(x='std',
            y='expeted_ret')
    print ("done")
