from binance.client import Client
api_key = "XXXXXXXXX"
api_secret = "XXXXXXXXXXX"
client = Client(api_key, api_secret)

# # get market depth
# depth = client.get_order_book(symbol='BNBBTC')

# # place a test market buy order, to place an actual order use the create_order function
# order = client.create_test_order(
#     symbol='BNBBTC',
#     side=Client.SIDE_BUY,
#     type=Client.ORDER_TYPE_MARKET,
#     quantity=100)

# get all symbol prices
# prices = client.get_all_tickers()

# # withdraw 100 ETH
# # check docs for assumptions around withdrawals
# from binance.exceptions import BinanceAPIException, BinanceWithdrawException
# try:
#     result = client.withdraw(
#         asset='ETH',
#         address='<eth_address>',
#         amount=100)
# except BinanceAPIException as e:
#     print(e)
# except BinanceWithdrawException as e:
#     print(e)
# else:
#     print("Success")

# # fetch list of withdrawals
# withdraws = client.get_withdraw_history()
#
# # fetch list of ETH withdrawals
# eth_withdraws = client.get_withdraw_history(asset='ETH')
#
# # get a deposit address for BTC
# address = client.get_deposit_address(asset='BTC')
#
# # start aggregated trade websocket for BNBBTC
# def process_message(msg):
#     print("message type: {}".format(msg['e']))
#     print(msg)
#     # do something

from binance.websockets import BinanceSocketManager
# bm = BinanceSocketManager(client)
# bm.start_aggtrade_socket('BNBBTC', process_message)
# bm.start()

# get historical kline data from any date range

# fetch 1 minute klines for the last day up until now
# klines = client.get_historical_klines("DOGEUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
# print(klines)


from time import sleep

ticker = 'DOGEUSDT'
quantity = 1000
buy_price = 0.00613
sell_percent = 0.006


def is_order_filled(side, order_id, quantity):
    order_resp = client.get_order(
      symbol=ticker,
      orderId=order_id)
    if order_resp["status"] == 'FILLED' \
            and order_resp["side"] == side \
            and order_resp["origQty"] == quantity:
        return True
    return False


def place_limit_sell(cost_price):
    while True:
        avg_price = float(client.get_avg_price(symbol=ticker)['price'])
        if avg_price > cost_price*(1+sell_percent):
            print(str(avg_price))
            order = client.order_limit_sell(
              symbol=ticker,
              quantity=quantity,
              price='{0:.5f}'.format(avg_price))
            print(order)
            return order['orderId']
        sleep(10)


def place_limit_buy(target_price):
    while True:
        avg_price = float(client.get_avg_price(symbol=ticker)['price'])
        if avg_price < target_price:
            print(avg_price)
            order = client.order_limit_buy(
              symbol=ticker,
              quantity=quantity,
              price='{0:.5f}'.format(avg_price))
            return order['orderId'], avg_price
        sleep(10)


def is_sell(order_id):
    while True:
        if is_order_filled('SELL', order_id):
            return True
        sleep(10)


def is_buy(order_id):
    while True:
        if is_order_filled('BUY', order_id):
            return True
        sleep(10)

time = 0
while time < 10:
    print('trade #: {0}'.format(time))

    buy_id, cost_price = place_limit_buy(buy_price)
    print('Buying..., set the cost: ', cost_price*quantity)
    sleep(10)

    if is_buy(buy_id, quantity):
        print('Bought')
    else:
        break

    sell_id = place_limit_sell(cost_price)
    print('Selling...., the revenue: ', cost_price*(1+sell_percent)*quantity)
    sleep(10)

    if is_sell(sell_id, quantity):
        print('Sold')
    else:
        break

    time += 1



# # fetch 30 minute klines for the last month of 2017
# klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")
#
# # fetch weekly klines since it listed
# klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")
