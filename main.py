from basic_trade_helper import dynamic_buy_and_sell, place_limit_buy, place_limit_sell
from misc import run_in_parallel

def fun1():
    dynamic_buy_and_sell(ticker='DOGEUSDT', time=10, total_cost=1000, premium=0.012, switch_trigger=0.04)


#
# run_in_parallel(
#     # buy_and_sell(ticker='DOGEUSDT', time=1, buy_price=0.0618, total_cost=295, sell_percent=0.006),
#     fun1,
# #dynamic_buy_and_sell(ticker, time, total_cost, premium=0.006, switch_trigger=0.1):
# )

place_limit_sell('WINUSDT', 0.00143, 480)

