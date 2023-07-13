import quandl
import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def strtime2utctimestamp(dt):
    # dt = datetime.strptime(val, "%m-%d-%Y")
    ts = datetime.timestamp(dt) - 4*3600
    dt_back = datetime.utcfromtimestamp(ts)
    if dt_back.hour != 0:
        ts -= dt_back.hour * 3600
    return int(ts * 1000 - 1)


def get_treasury_yield_scenario(scen_bump, df_yield):
    df_yield.sort_values(by=["timestamp"], inplace=True)
    date = df_yield["date"].tolist()[-1]
    today_date = datetime.strptime(date, "%m-%d-%Y") + timedelta(days=1)
    today_date_str = datetime.strftime(today_date, "%m-%d-%Y")
    time_stamp = strtime2utctimestamp(datetime.strptime(today_date_str, "%m-%d-%Y"))
    df_new_scen = {
                    "date": today_date_str,
                    "timestamp": time_stamp
                    }
    # tenor = ["1 MO","2 MO","3 MO","6 MO","1 YR","2 YR","3 YR","5 YR","7 YR","10 YR","20 YR","30 YR"]
    prev_yield = df_yield[df_yield["date"] == date]["yield"].values[0]
    df_new_scen["yield"] = scen_bump / 100 + prev_yield
    df = df_yield.append(df_new_scen, ignore_index=True)
    return df

if __name__ == "__main__":
    print ("done")
