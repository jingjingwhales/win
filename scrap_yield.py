import requests
import pandas as pd
from datetime import datetime, timedelta
from datetime import timezone
import pytz
import holidays

us_holidays = holidays.US()
def strtime2utctimestampPrevDayEnd(val):
    dt = datetime.strptime(val, "%Y-%m-%d")
    utc_time = dt.replace(tzinfo=timezone.utc)
    ts = int(utc_time.timestamp() * 1000 - 1)

    return ts

def isTodayWeekendDayOrHoliday():
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz)
    if today.weekday() > 4:
        return True
    if today in us_holidays:
        return True
    return False

URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2023/all?type=daily_treasury_yield_curve&field_tdr_date_value=2022&page&_format=csv"
if isTodayWeekendDayOrHoliday():
    df_history = pd.read_csv("historical_treasury_yield_curve.csv")
    latest_day_str = df_history["Date"].tolist()[-1]
    latest_day = datetime.strptime(latest_day_str, "%Y-%m-%d")
    today_str = datetime.strftime(latest_day + timedelta(days=1), "%Y-%m-%d")
    latest_yield = df_history[df_history["Date"] == latest_day_str]
    latest_yield["Date"] = today_str
    latest_yield["timestamp"] = latest_yield["Date"].apply(lambda val: strtime2utctimestampPrevDayEnd(val))
    df_all = pd.concat([df_history, latest_yield])
else:
    page = requests.get(URL)
    page_text = page.text

    pg_lines = page_text.strip().split('\n')
    yield_data = [line.split(",")for line in pg_lines]
    yield_header = [val.replace('"', "").upper() for val in yield_data[0]]
    tenor_header = []
    for val in yield_header:
        if val != "DATE":
            tenor_header.append(val)
    df = pd.DataFrame(data=yield_data[1:], columns=yield_header)
    df.rename(columns={"DATE": "Date"}, inplace=True)
    df["Date"] = df["Date"].apply(lambda val: datetime.strftime(datetime.strptime(val, "%m/%d/%Y"), "%Y-%m-%d"))
    # df[tenor_header] = df[tenor_header].astype("float")
    df[tenor_header] = df[tenor_header].apply(pd.to_numeric, errors='coerce')
    df["timestamp"] = df["Date"].apply(lambda val: strtime2utctimestampPrevDayEnd(val))

    df_history = pd.read_csv("historical_treasury_yield_curve.csv")
    new_data = set(df["Date"])
    df_history = df_history[~df_history["Date"].isin(new_data)]
    df_all = pd.concat([df_history, df], ignore_index=True)
df_all.drop_duplicates(inplace=True)
df_all.sort_values(by=["timestamp"], inplace=True)

df_all["Date"] = df_all["Date"].apply(lambda val: datetime.strptime(val, "%Y-%m-%d"))
# df_all["timestamp"] = df_all["timestamp"].apply(lambda val: int(val))
df_all.set_index(["Date"], inplace=True)
df_reindexed = df_all.reindex(pd.date_range(start=df_all.index.min(),
                                                  end=df_all.index.max(),
                                                  freq='1D'))
df_all = df_reindexed.interpolate(method="linear")
df_all.reset_index(inplace=True)
df_all.rename(columns={"index": "Date"}, inplace=True)
df_all["Date"] = df_all["Date"].apply(lambda val: datetime.strftime(val, "%Y-%m-%d"))
df_all["timestamp"] = df_all["timestamp"].apply(lambda val: int(val))
print ("Saving treasury yield curve......")
df_all.to_csv("historical_treasury_yield_curve.csv", index=False)

print ("Saving treasury yield curve done")
