from basic_trade_helper import dynamic_buy_and_sell, place_limit_buy, place_limit_sell
from misc import run_in_parallel

def fun1():
    dynamic_buy_and_sell(ticker='STXUSDT', time=10, total_cost=1000, gap=0.008, precision=4, premium=0.02,
                         basic_premium=0.008, sell_time_out=30, switch_trigger=0.1)
# # # #
run_in_parallel(
    fun1,
)
