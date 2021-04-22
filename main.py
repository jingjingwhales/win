from basic_trade_helper import dynamic_buy_and_sell, place_limit_buy, place_limit_sell
from misc import run_in_parallel
from calc_volatility import getFullListOfCryptos

def fun1():
    ticker_list = getFullListOfCryptos(numer_of_crypto_currenty=5000, market_cap_min=1000000000)
    top_ticker = ticker_list["ticker"].tolist()[0]
    hourly_growth_rate = ticker_list[ticker_list["ticker"]==top_ticker]["average_growht_rate"]
    hourly_vol = ticker_list[ticker_list["ticker"]==top_ticker]["average_growht_rate"]
    dynamic_buy_and_sell(ticker='STXUSDT', time=10, total_cost=1000, gap=0.008, precision=4, premium=0.02,
                         basic_premium=0.008, sell_time_out=30, switch_trigger=0.1)
# # # #
run_in_parallel(
    fun1,
)
