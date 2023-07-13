import requests
from requests.exceptions import HTTPError
from datetime import datetime
import json
import pandas as pd
from datetime import timezone

def strtime2utctimestamp(val):
    dt = datetime.strptime(val, "%m-%d-%Y")
    utc_time = dt.replace(tzinfo=timezone.utc)
    ts = int(utc_time.timestamp() * 1000 - 1)

    return ts

index_web = "https://api.alternative.me/fng/?limit=360"

try:
    response = requests.get(index_web)
except HTTPError as http_error:
    print (http_error)

content = response.text
data = json.loads(content)["data"]
date = []
fng_value = []
fng_classification = []
time_stamp = []
for info in data:
    fng_value.append(int(info["value"]))
    timestamp = int(info["timestamp"]) * 1000 - 1
    time_stamp.append(timestamp)
    fng_classification.append(info["value_classification"])
    date.append(datetime.strftime(datetime.utcfromtimestamp((timestamp + 1) / 1000),"%m-%d-%Y"))
df_recent = pd.DataFrame({"date": date, "fng_value": fng_value, "fng_classification": fng_classification, "timestamp": time_stamp})
df_hist = pd.read_csv("FNG_index.index")
df_all = pd.concat([df_hist, df_recent])
df_all.sort_values(by=["timestamp"], inplace=True)
df_all.drop_duplicates(inplace=True)
df_all.to_csv("FNG_index.index", index=False)

print ("getting fear and greedy index done")
