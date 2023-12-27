# from basic_trade_helper import bi_client_us
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
import pickle
import os
import logging
from datetime import timezone
import json
from get_treasury import get_treasury_yield_scenario

logging.basicConfig(filename="./modeling.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# client = bi_client_us()
DATA_DIR = "./ticker_data"
SCEN = {
            "30YRYieldFlat": 0,
            "30YRYieldUp1bps": 1,
            "30YRYieldDn1bps": -1,
            "30YRYieldDn5bps": -5,
            "30YRYieldUp1bps": 5,
            "30YRYieldDn10bps": -10,
            "30YRYieldUp10bps": 10
            }

class cryptoMLModel:

    def __init__(self, time_window, frequency="daily"):
        # self.crypto_name = crypto_name
        self.frequency = frequency
        self.window = time_window
        self.X_vars = []
        self.X_vars_range = {}
        self.next_prediction_data = []
        self.price_movement_model = defaultdict()
        self.next_predict_value = None
        self.price_change_model = None
        self.close_oepn_price_chg_threshold = [-np.inf, -2.0/100, -1.5/100, -1.0/100, -0.5/100, 0.0/100,
                                               0.5/100, 1.0/100, 1.5/100, 2.0/100, np.inf]
        self.bins = ["<-2pct", "-2--1.5pct", "-1.5--1pct", "-1--0.5pct", "-0.5-0pct",
                        "0-0.5pct", "0.5-1pct", "1-1.5pct", "1.5-2pct", ">2pct"]
        self.forecasting_data = None

    def model_traning(self):

        now = datetime.now(timezone.utc)
        whole_hr_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        end_timestamp = int(whole_hr_dt.timestamp() * 1000 - 1)

        if not os.path.isfile("./all_data.csv"):
            df_all_data, Xvars = get_all_crytpo_data(self.window)
            df_all_data.to_csv("./all_data.csv", index=False)
        else:
            df_all_data = pd.read_csv("./all_data.csv")
            Xvars = get_x_var(self.window)

        self.X_vars = Xvars

        df_all_data["date"] = df_all_data["timestamp"].apply(lambda val: \
                            datetime.strftime(datetime.utcfromtimestamp((val) / 1000),"%m-%d-%Y"))

        df_all_data["price_mv_category"] = pd.cut(df_all_data["close_open_price_chg"], self.close_oepn_price_chg_threshold, labels=self.bins)
        for bin in self.bins:
            price_mv_model_classfication = "price_movement_model_{mdl}".format(mdl=bin)
            df_all_data[price_mv_model_classfication] = df_all_data["price_mv_category"] == bin
            df_all_data[price_mv_model_classfication] = df_all_data[price_mv_model_classfication].astype("float")

        logger.info("getting fear and greedy index...")
        df_index = pd.read_csv("FNG_index.index")
        fng_lag_chg_cols = []
        for lag in self.window:
            lag_column = "fng_value_lag_{lag}".format(lag=lag)
            pct_chg_column = "fng_value_chg_{lag}".format(lag=lag)
            df_index[lag_column] = df_index["fng_value"].shift(periods=lag)
            df_index[pct_chg_column] = df_index["fng_value"] / df_index[lag_column] - 1
            fng_lag_chg_cols.append(pct_chg_column)
        df_all_data = pd.merge(left=df_all_data, right=df_index[fng_lag_chg_cols + ["date"]], how="left", on=["date"])
        self.X_vars = self.X_vars + fng_lag_chg_cols

        logger.info("getting treasury yield data...")
        df_yield = pd.read_csv("historical_treasury_yield_curve.csv")
        df_yield = df_yield[["Date", "30 YR"]]
        df_yield.rename(columns={"30 YR": "yield", "Date": "date"}, inplace=True)
        df_yield["date"] = df_yield["date"].apply(lambda val: datetime.strftime(datetime.strptime(val, "%Y-%m-%d"), "%m-%d-%Y"))
        df_yield["date"] = df_yield["date"].apply(
            lambda val: datetime.strftime(datetime.strptime(val, "%m-%d-%Y") + timedelta(days=1), "%m-%d-%Y"))
        yield_lag_chg_cols = []
        for lag in self.window:
            lag_column = "30Y_yield_lag_{lag}".format(lag=lag)
            pct_chg_column = "30Y_yield_chg_{lag}".format(lag=lag)
            df_yield[lag_column] = df_yield["yield"].shift(periods=lag)
            df_yield[pct_chg_column] = df_yield["yield"] / df_yield[lag_column] - 1
            yield_lag_chg_cols.append(pct_chg_column)

        df_all_data = pd.merge(left=df_all_data, right=df_yield[yield_lag_chg_cols + ["date"]], how="left", on=["date"])
        self.X_vars = self.X_vars + yield_lag_chg_cols

        df_all_data = self.normalize_vars(df_all_data)

        df_all_data = self.get_fix_effect(df_all_data)
        columns = list(df_all_data.columns)
        fixed_effect_vars = [x for x in columns if "Symbol_" in x]
        self.X_vars = self.X_vars + fixed_effect_vars
        prediction_timestamp = end_timestamp + 3600000 * 24
        self.forecasting_data = df_all_data[df_all_data["timestamp"]==prediction_timestamp]
        df_all_data.dropna(inplace=True)

        # for var in self.X_vars:
        #     x_max = df_all_data[var].max()
        #     x_min = df_all_data[var].min()
        #     df_all_data[var] = (df_all_data[var] - x_min) / (x_max - x_min)

        msk = np.random.rand(len(df_all_data)) > 0.9
        df_model_data_traning = df_all_data[~msk]
        df_all_data["oot"] = msk

        # self.price_change_model = MLPRegressor(solver="adam", activation="relu", alpha=1e-5, hidden_layer_sizes=(150,150,150), max_iter=500)

        #########there are bunch of models to train depends on different threshold of price movement##############

        for bin in self.bins:
            price_mv_model_classfication = "price_movement_model_{mdl}".format(mdl=bin)
            self.price_movement_model[price_mv_model_classfication] = MLPClassifier(solver="adam", activation="relu", alpha=1e-5,
                                                      hidden_layer_sizes=(10, 10, 10), max_iter=500)
            logger.info("training model {model}".format(model=price_mv_model_classfication))
            logger.info("training sample size={sample_size}".format(sample_size=df_model_data_traning.shape))
            self.price_movement_model[price_mv_model_classfication].fit(df_model_data_traning[self.X_vars], df_model_data_traning[price_mv_model_classfication])
            price_move_predict = self.price_movement_model[price_mv_model_classfication].predict_proba(df_all_data[self.X_vars])

            price_mv_bin_pred_prob_var = "price_chg_{bin}_prob_pred".format(bin=bin)
            price_mv_bin_cl_pred_var = "price_chg_{bin}_pred".format(bin=bin)
            df_all_data[price_mv_bin_pred_prob_var] = price_move_predict[:, 1]
            roc = roc_auc_score(df_all_data[price_mv_model_classfication], df_all_data[price_mv_bin_pred_prob_var])
            logger.info("ROC score = {roc} for model {price_movement_model}".format(roc=roc, price_movement_model=price_mv_model_classfication))
            df_all_data[price_mv_bin_cl_pred_var] = df_all_data[price_mv_bin_pred_prob_var].apply(
                lambda val: 1 if val > 0.5 else 0)

            df_all_data["correct_{val}".format(val=price_mv_bin_cl_pred_var)] = \
                (df_all_data[price_mv_model_classfication] == df_all_data[price_mv_bin_cl_pred_var])
            training_sample_accuracy = df_all_data[df_all_data["oot"] == 0]["correct_{val}".format(val=price_mv_bin_cl_pred_var)].sum() / len(
                df_model_data_traning)
            oot_sample_accuracy = df_all_data[df_all_data["oot"] == 1]["correct_{val}".format(val=price_mv_bin_cl_pred_var)].sum() / (
                df_all_data["oot"].sum())

            logger.info("training accuracy for model {price_movement_model} = {training_sample_accuracy}".format(
                training_sample_accuracy=training_sample_accuracy, price_movement_model=price_mv_model_classfication))
            logger.info("oot accuracy = {oot_sample_accuracy} for model {price_movement_model}".format(oot_sample_accuracy=oot_sample_accuracy,
                                                                                                       price_movement_model=price_mv_model_classfication))

        # self.price_change_model.fit(df_model_data_traning[self.X_vars], df_model_data_traning["close_open_price_chg"])
        # price_change_prediction = self.price_change_model.predict(df_all_data[self.X_vars])
        # df_all_data["close_open_price_chg_pred"] = price_change_prediction
        # r_square = self.price_change_model.score(df_model_data_traning[self.X_vars], df_model_data_traning["close_open_price_chg"])
        # df_model_data["oot"] = df_model_data["timestamp"].apply(lambda val: 1 if val in set(oot) else 0)
        # print ("R-square={r_square}".format(r_square=r_square))
        # df_all_data["error_square"] = (df_all_data["close_open_price_chg"] - df_all_data["close_open_price_chg_pred"]) ** 2
        # rmse_training = math.sqrt(df_all_data[df_all_data["oot"] == 0]["error_square"].sum() / len(df_model_data_traning))
        # rmse_oot = math.sqrt(df_all_data[df_all_data["oot"] == 1]["error_square"].sum()) / (df_all_data["oot"].sum())
        # print("training rmse = {training_sample_rmse}".format(training_sample_rmse=rmse_training))
        # print("oot rmse = {rmse_oot}".format(rmse_oot=rmse_oot))

    def get_fix_effect(self, df):
        symbol = df["Symbol"].tolist()
        df = pd.get_dummies(df, columns=["Symbol"])
        df.drop(columns=["Symbol_BTC"], inplace=True)
        df["Symbol"] = symbol
        return df

    def normalize_vars(self, df):
        # vars_range = defaultdict(lambda: [])
        for var in self.X_vars:
            var_max = df[var].max()
            var_min = df[var].min()
            self.X_vars_range[var] = [var_min, var_max]
            df[var] = (df[var] - var_min) / (var_max - var_min)
        return df

def save_model(model, output_file):
    output = open(output_file, 'wb')
    pickle.dump(model, output)
    output.close()

def read_model(model_file):
    pkl_file = open(model_file, 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model

def get_price_up_down_within_window(data, window):
    res = []
    sum = 0
    for i in range(window):
        if data[i][4] >= data[i][1]:
            sum += 1
    res.append(sum / window)
    for i in range(window, len(data)):
        if data[i - window][4] >= data[i-window][1]:
            sum -= 1
        if data[i][4] >= data[i][1]:
            sum += 1
        res.append(sum / window)
    return res

def get_highest_price_info_within_window(data, window):
    queue = deque()
    res = []
    for i in range(len(data)):
        if len(queue) == 0:
            queue.append((i, [data[i][0], data[i][6], data[i][2]]))
            continue
        if i >= window:
            res.append(queue[0])
        while i - queue[0][0] >= window:
            queue.popleft()
            if len(queue) == 0:
                queue.append((i, [data[i][0], data[i][6], data[i][2]]))
                break
        while data[i][2] >= queue[-1][1][2]:
            queue.pop()
            if len(queue) == 0:
                break
        queue.append((i, [data[i][0], data[i][6], data[i][2]]))
    res.append(queue[0])
    return res

def get_open_close_price_change(data):
    res = []
    for i in range(len(data)):
        res.append(data[i][4] / data[i][1] - 1)
    return res

def get_pct_change_lag(data, window):
    res = [np.nan] * window
    for i in range(window, len(data)):
        res.append(data[i] / data[i - window] - 1)
    return res

def get_data(crypto_name, time_window):
    binance_ticker = crypto_name + "USDT"
    ticker_json_file = DATA_DIR + "/{ticker}.json".format(ticker=crypto_name)
    with open(ticker_json_file, 'r') as openfile:
        data_dict = json.load(openfile)
    data = data_dict[crypto_name]
    time_stamp = [item[6] for item in data]
    data = [[float(x) for x in item] for item in data]
    avg_price = [(item[1] + item[4]) / 2 for item in data]
    vol = [item[5] for item in data]
    highest_price_window = {}
    lowest_price_window = {}
    price_up_and_down = {}
    vol_pct_chg_window = {}
    peak_trough_pct_chg_window = defaultdict(lambda: [])
    all_data = {}
    X_vars = []

    # self.df_crypto = pd.DataFrame({"avg_price"})
    data_price_neg = [[item[0], item[1], -item[3], item[2], item[4], item[5], item[6]] for item in data]
    for window in time_window:
        # print (window)
        highest_price_window[window] = get_highest_price_info_within_window(data, window=window)
        lowest_price_window[window] = get_highest_price_info_within_window(data_price_neg, window=window)
        price_up_and_down[window] = get_price_up_down_within_window(data, window=window)
        vol_pct_chg_window[window] = get_pct_change_lag(vol, window=window)

        for i in range(len(highest_price_window[window])):
            pct_chg = highest_price_window[window][i][1][2] / (-lowest_price_window[window][i][1][2]) - 1
            if highest_price_window[window][i][1][1] < lowest_price_window[window][i][1][1]:
                pct_chg = pct_chg * (-1)
            elif highest_price_window[window][i][1][1] == lowest_price_window[window][i][1][1]:
                index = highest_price_window[window][i][0]
                if data[index][4] < data[index][1]:
                    pct_chg = pct_chg * (-1)
            peak_trough_pct_chg_window[window].append(pct_chg)
    all_data["price_movement"] = price_up_and_down[1].copy() + [np.nan]
    all_data["pct_chg"] = peak_trough_pct_chg_window[1] + [np.nan]
    # all_data["vol_pct_chg"]

    open_to_close_price_chg = get_open_close_price_change(data)
    all_data["close_open_price_chg"] = open_to_close_price_chg + [np.nan]

    for window in time_window:
        var = "price_up_down_lag_" + str(window)
        X_vars.append(var)
        all_data[var] = ([np.nan] * window + price_up_and_down[window])#[:len(self.all_data["pct_chg"])]

        var = "peak_trough_pct_chg_" + str(window)
        X_vars.append(var)
        all_data[var] = ([np.nan] * window + peak_trough_pct_chg_window[window])#[:len(self.all_data["pct_chg"])]

        var = "close_open_price_chg_lag_" + str(window)
        X_vars.append(var)
        all_data[var] = ([np.nan] * window + open_to_close_price_chg)[:len(all_data["close_open_price_chg"])]

        var = "vol_pct_chg_lag_" + str(window)
        X_vars.append(var)
        all_data[var] = ([np.nan] + vol_pct_chg_window[window])[:len(all_data["close_open_price_chg"])]

    all_data["timestamp"] = time_stamp + [time_stamp[-1] + 3600000 * 24]

    logger.info("\t\tdone with {crypto} data!".format(crypto=crypto_name))
    df_data = pd.DataFrame(data=all_data)
    # for var in X_vars:
    #     df_data[var] = df_data[var].apply(lambda val: val if val >= -40 and val <= 40 else (40 if val > 40 else -40))
    return df_data, X_vars

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

def get_all_crytpo_data(time_window):
    logger.info("Getting crypto data...")
    df_list = get_crypto_list("crypto_list.csv")
    df_all_crypto_data = pd.DataFrame()
    Xvars = None
    for index, row in df_list.iterrows():
        ticker = row["Symbol"]
        if "USD" in ticker:
            continue
        try:
            df_crypto, Xvars = get_data(ticker, time_window)
            df_crypto["Symbol"] = ticker
            df_all_crypto_data = pd.concat([df_all_crypto_data, df_crypto])
        except Exception as e:
            # print(e.message)
            print (e)
    logger.info("Successfully got all crypto data...")
    return df_all_crypto_data, Xvars

def get_x_var(time_window):
    X_vars = []
    for window in time_window:
        var = "price_up_down_lag_" + str(window)
        X_vars.append(var)

        var = "peak_trough_pct_chg_" + str(window)
        X_vars.append(var)

        var = "close_open_price_chg_lag_" + str(window)
        X_vars.append(var)

        var = "vol_pct_chg_lag_" + str(window)
        X_vars.append(var)
    return X_vars

def normalizePredProd(prob1, prob_sum):
    if prob_sum >= 1:
        return prob1 / prob_sum
    return prob1

if __name__ == "__main__":
    time_window = [1,2,3,4,7,12,24,30,60,90]
    if os.path.isfile("model_output.pkl"):
        crypto_model = read_model("model_output.pkl")
    else:
        crypto_model = cryptoMLModel(time_window)
        crypto_model.model_traning()
        save_model(crypto_model, "model_output.pkl")

    now = datetime.utcnow()
    whole_dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    end_timestamp = int(whole_dt.timestamp() * 1000 - 1)

    df_predict_data = crypto_model.forecasting_data
    Xvars = crypto_model.X_vars
    # Xvars = get_x_var(time_window)
    # # df_predict_data = crypto_model.get_fix_effect(df_predict_data)
    # columns = list(df_predict_data.columns)
    # fixed_effect_vars = [x for x in columns if "Symbol_" in x]
    # Xvars = Xvars + fixed_effect_vars
    # df_index = pd.read_csv("FNG_index.index")
    # # df_index["date"] = df_index["date"].apply(
    # #     lambda val: datetime.strftime(datetime.strptime(val, "%m-%d-%Y") + timedelta(days=1), "%m-%d-%Y"))
    # df_predict_data["date"] = df_predict_data["timestamp"].apply(lambda val: \
    #                                                          datetime.strftime(
    #                                                              datetime.utcfromtimestamp((val) / 1000), "%m-%d-%Y"))
    # index_lag_chg_cols = []
    # for lag in time_window:
    #     lag_column = "fng_value_lag_{lag}".format(lag=lag)
    #     pct_chg_column = "fng_value_chg_{lag}".format(lag=lag)
    #     df_index[lag_column] = df_index["fng_value"].shift(periods=lag)
    #     df_index[pct_chg_column] = df_index["fng_value"] / df_index[lag_column] - 1
    #     index_lag_chg_cols.append(pct_chg_column)
    # df_predict_data = pd.merge(left=df_predict_data, right=df_index[index_lag_chg_cols + ["date"]], how="left", on=["date"])
    # Xvars = Xvars + index_lag_chg_cols
    #
    # logger.info("getting treasury yield data...")
    # df_yield = pd.read_csv("historical_treasury_yield_curve.csv")
    # df_yield = df_yield[["Date", "30 YR"]]
    # df_yield.rename(columns={"30 YR": "yield", "Date": "date"}, inplace=True)
    # df_yield["date"] = df_yield["date"].apply(
    #     lambda val: datetime.strftime(datetime.strptime(val, "%Y-%m-%d"), "%m-%d-%Y"))
    # df_yield["date"] = df_yield["date"].apply(
    #     lambda val: datetime.strftime(datetime.strptime(val, "%m-%d-%Y") + timedelta(days=1), "%m-%d-%Y"))
    # yield_lag_chg_cols = []
    # for lag in time_window:
    #     lag_column = "30Y_yield_lag_{lag}".format(lag=lag)
    #     pct_chg_column = "30Y_yield_chg_{lag}".format(lag=lag)
    #     df_yield[lag_column] = df_yield["yield"].shift(periods=lag)
    #     df_yield[pct_chg_column] = df_yield["yield"] / df_yield[lag_column] - 1
    #     yield_lag_chg_cols.append(pct_chg_column)
    # Xvars = Xvars + yield_lag_chg_cols
    # df_predict_data = pd.merge(left=df_predict_data, right=df_yield[yield_lag_chg_cols + ["date"]], how="left", on=["date"])

    last_hr_X = df_predict_data[df_predict_data["timestamp"] == end_timestamp + 24 * 3600000]

    # for var in crypto_model.X_vars_range:
    #     last_hr_X[var] = (last_hr_X[var] - crypto_model.X_vars_range[var][0]) / (
    #                 crypto_model.X_vars_range[var][1] - crypto_model.X_vars_range[var][0])
    # last_hr_X.dropna(subset=Xvars, inplace=True)
    # for var in crypto_model.X_vars:
    #     if var not in Xvars:
    #         logger.info("{var} is missing in the prediction dataset...defaulting it to 0".format(var=var))
    #         last_hr_X[var] = 0
    predict_cols = []
    for bin in crypto_model.bins:
        model_name = "price_movement_model_{mdl_bin}".format(mdl_bin=bin)
        predictive_model = crypto_model.price_movement_model[model_name]
        prediction = predictive_model.predict_proba(last_hr_X[crypto_model.X_vars])
        price_mv_bin_pred_prob_var = "price_chg_{bin}_prob_pred".format(bin=bin)
        last_hr_X[price_mv_bin_pred_prob_var] = prediction[:, 1]
    # close_open_price_chg_prediction = crypto_model.price_change_model.predict(last_hr_X[Xvars])

    last_hr_X["sum_pred_prob"] = 0
    for bin in crypto_model.bins[:-1]:
        price_mv_bin_pred_prob_var = "price_chg_{bin}_prob_pred".format(bin=bin)
        last_hr_X["sum_pred_prob"] = last_hr_X["sum_pred_prob"] + last_hr_X[price_mv_bin_pred_prob_var]
    for bin in crypto_model.bins[:-1]:
        price_mv_bin_pred_prob_var = "price_chg_{bin}_prob_pred".format(bin=bin)
        last_hr_X[price_mv_bin_pred_prob_var] = last_hr_X.apply(lambda row: normalizePredProd(row[price_mv_bin_pred_prob_var], row["sum_pred_prob"]), axis=1)

    prev_pred_pct_prob_var = ""
    for i in range(1, len(crypto_model.close_oepn_price_chg_threshold)-1):
        price_mv_up_pct_name = "price_up_{0:.1f}pct_prob_pred".format(crypto_model.close_oepn_price_chg_threshold[i]*100)
        price_mv_bin_pred_prob_var = "price_chg_{bin}_prob_pred".format(bin=crypto_model.bins[i-1])
        if i == 1:
            last_hr_X[price_mv_up_pct_name] = 1 - last_hr_X[price_mv_bin_pred_prob_var]
        else:
            last_hr_X[price_mv_up_pct_name] = last_hr_X[prev_pred_pct_prob_var] - last_hr_X[price_mv_bin_pred_prob_var]
        prev_pred_pct_prob_var = price_mv_up_pct_name
        predict_cols.append(price_mv_up_pct_name)

    prediction_dt_from = datetime.utcfromtimestamp((end_timestamp + 1) / 1000)
    pred_time_from_str = prediction_dt_from.strftime("%Y%m%d%H%M")
    prediction_dt_to = datetime.utcfromtimestamp((end_timestamp + 24 * 3600000) / 1000)
    pred_time_to_str = prediction_dt_to.strftime("%Y%m%d%H%M")
    last_hr_X[["Symbol"] + predict_cols].to_csv(
        "./prediction_from_{dt_from}_to_{dt_to}.csv".format(dt_from=pred_time_from_str,
                                                            dt_to=pred_time_to_str), index=False)
    # if not os.path.isfile("./all_data.csv"):
    #     df_all_data, Xvars = get_all_crytpo_data([1,2,3,7,12,24,30,60,90,180,360])
    # else:
    #     df_all_data = pd.read_csv("./all_data.csv")
    #     Xvars = get_x_var([1,2,3,7,12,24,30,60,90,180,360])

    print("done with prediction")

    # for scenario in SCEN:
    #     df_predict_data_scen = df_predict_data.copy()
    #     df_yield_scen = get_treasury_yield_scenario(SCEN[scenario], df_yield)
    #     df_yield_scen["date"] = df_yield_scen["date"].apply(
    #         lambda val: datetime.strftime(datetime.strptime(val, "%m-%d-%Y") + timedelta(days=1), "%m-%d-%Y"))
    #     yield_lag_chg_cols = []
    #     for lag in time_window:
    #         lag_column = "30Y_yield_lag_{lag}".format(lag=lag)
    #         pct_chg_column = "30Y_yield_chg_{lag}".format(lag=lag)
    #         df_yield_scen[lag_column] = df_yield_scen["yield"].shift(periods=lag)
    #         df_yield_scen[pct_chg_column] = df_yield_scen["yield"] / df_yield_scen[lag_column] - 1
    #         yield_lag_chg_cols.append(pct_chg_column)
    #
    #     df_predict_data_scen = pd.merge(left=df_predict_data_scen, right=df_yield_scen[yield_lag_chg_cols + ["date"]], how="left", on=["date"])
    #     Xvars = Xvars + yield_lag_chg_cols
    #
    #     last_hr_X = df_predict_data_scen[df_predict_data_scen["timestamp"] == end_timestamp - 4 * 3600000 + 24 * 3600000]
    #
    #     for var in crypto_model.X_vars_range:
    #         last_hr_X[var] = (last_hr_X[var] - crypto_model.X_vars_range[var][0]) / (crypto_model.X_vars_range[var][1] - crypto_model.X_vars_range[var][0])
    #     last_hr_X.dropna(subset=Xvars, inplace=True)
    #     for var in crypto_model.X_vars:
    #         if var not in Xvars:
    #             logger.info("{var} is missing in the prediction dataset...defaulting it to 0".format(var=var))
    #             last_hr_X[var] = 0
    #     prediction = crypto_model.price_movement_model.predict_proba(last_hr_X[crypto_model.X_vars])
    #     # close_open_price_chg_prediction = crypto_model.price_change_model.predict(last_hr_X[Xvars])
    #     prediction_dt_from = datetime.fromtimestamp((end_timestamp - 4 * 3600000 + 1) / 1000)
    #     pred_time_from_str = prediction_dt_from.strftime("%Y%m%d%H%M")
    #     prediction_dt_to = datetime.fromtimestamp((end_timestamp - 4 * 3600000 + 24 * 3600000) / 1000)
    #     pred_time_to_str = prediction_dt_to.strftime("%Y%m%d%H%M")
    #     last_hr_X["price_up_prob"] = prediction[:, 1]
    #     # last_hr_X["close_open_price_chg_prediction"] = close_open_price_chg_prediction
    #     last_hr_X[["Symbol", "price_up_prob"]].to_csv("./{scenario}_prediction_from_{dt_from}_to_{dt_to}.csv".format(scenario=scenario, dt_from=pred_time_from_str, dt_to=pred_time_to_str), index=False)

        # if not os.path.isfile("./all_data.csv"):
        #     df_all_data, Xvars = get_all_crytpo_data([1,2,3,7,12,24,30,60,90,180,360])
        # else:
        #     df_all_data = pd.read_csv("./all_data.csv")
        #     Xvars = get_x_var([1,2,3,7,12,24,30,60,90,180,360])

        # print ("done with scenario {scenario}".format(scenario=scenario))

    # crypto.model_traning()
    #
    # loss_curve = crypto_model.price_movement_model.loss_curve_
    # df_loss_curve = pd.DataFrame({"loss_curve": loss_curve})
    # df_loss_curve.to_csv("./loss_curve.csv", index=False)
    #
    # print ("prediction = {forecast_probablity}".format(forecast_probablity=crypto.next_predict_value))
