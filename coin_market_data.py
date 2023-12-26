from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import numpy as np
import pandas as pd
# from coinbase.wallet.client import Client
# from coinbase.wallet.model import APIObject
# import cbpro

def getMarketDataFromCoinmarketcap(numer_of_crypto_currenty, market_cap_min, sort_method):
    """
    :param numer_of_crypto_currenty:
    :param market_cap_min:
    :param sort_method: ["name" "symbol" "date_added" "market_cap" "market_cap_strict" "price" "circulating_supply" "total_supply" "max_supply" "num_market_pairs" "volume_24h" "percent_change_1h" "percent_change_24h" "percent_change_7d" "market_cap_by_total_supply_strict" "volume_7d" "volume_30d"]
    :return:
    """
    # https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': numer_of_crypto_currenty,
        'market_cap_min': market_cap_min,
        'convert': 'USD',
        'sort': sort_method,
        'sort_dir': "desc"
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '901fa460-1f4d-405d-bafc-7e88a1f8b958',
    }

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        currency_data = []
        for currency in data["data"]:
            one_currency = {}
            one_currency["Name"] = currency.get("name", '')
            one_currency["Symbol"] = currency.get("symbol", '')
            one_currency["Price"] = currency.get("quote", {}).get("USD", {}).get("price", np.nan)
            one_currency["Volume_24h"] = currency.get("quote", {}).get("USD", {}).get("volume_24h", np.nan)
            one_currency["Pct_chg_1h"] = currency.get("quote", {}).get("USD", {}).get("percent_change_1h", np.nan)
            one_currency["Pct_chg_24h"] = currency.get("quote", {}).get("USD", {}).get("percent_change_24h", np.nan)
            one_currency["Pct_chg_7d"] = currency.get("quote", {}).get("USD", {}).get("percent_change_7d", np.nan)
            one_currency["Pct_chg_30d"] = currency.get("quote", {}).get("USD", {}).get("percent_change_30d", np.nan)
            one_currency["Market_cap"] = currency.get("quote", {}).get("USD", {}).get("market_cap", np.nan)
            one_currency["Last_updated"] = currency.get("quote", {}).get("USD", {}).get("last_updated", np.nan)
            currency_data.append(one_currency.copy())
        return pd.DataFrame.from_dict(currency_data)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

# def getDataFromCoinBase(crypto_currency, fiat_currency, time_interval):
#     public_client = cbpro.PublicClient()
#     request_pair = "{currency1}-{currency2}".format(currency1=crypto_currency, currency2=fiat_currency)
#     public_client.get_product_historic_rates(request_pair, granularity=time_interval)
#
#     print("done")

if __name__ == "__main__":
    df = getMarketDataFromCoinmarketcap(numer_of_crypto_currenty=5000, market_cap_min=10000000, sort_method="percent_change_24h")
    # getDataFromCoinBase("ETH", "USD", 300)
    df.sort_values(by=["Market_cap"], ascending=False, inplace=True)
    df = df[["Symbol"]]
    df.to_csv("crypto_list.csv", index=False)
    print ("done")
