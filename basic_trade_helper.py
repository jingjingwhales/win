from binance.client import Client
from time import sleep, time
from datetime import datetime
from time_helper import last_4_hours_milli_time, last_30_mins_milli_time
from requests.exceptions import Timeout  # this handles ReadTimeout or ConnectTimeout

api_key = "Aw5BSQ65dE1fc5knmJyNAqHahWPetNj3p0FIuxLAzXz39iGsFlB9rAzeVaat69FK"
api_secret = "BuKiIxpVuffIm0XEfUYQtNkqEz6zTLjm02X1EU1gRUQEMefX229odKT3Tc28vAIM"
client = Client(api_key, api_secret,  {"verify": True, "timeout": 2000})


def is_order_filled(ticker, side, order_id):
    try:
        order_resp = client.get_order(
            symbol=ticker,
            orderId=order_id)
    except Timeout as e:
        order_resp = is_order_filled(ticker, side, order_id)
    except Exception as e:
        print("Error Message: ", e)
        return False

    if order_resp["status"] == 'FILLED' \
            and order_resp["side"] == side:
        return True
    return False


def place_limit_sell(ticker, target_price, quantity):
    order = client.order_limit_sell(
        symbol=ticker,
        quantity=quantity,
        price=str(target_price))
    return order['orderId']


def place_limit_buy(ticker, target_price, quantity):
    order = client.order_limit_buy(
        symbol=ticker,
        quantity=quantity,
        price=str(target_price))
    return order['orderId'], target_price


def place_cancel_order(ticker, order_id):
    order = client.cancel_order(
        symbol=ticker,
        orderId=order_id)
    print("Cancel order: ", order)
    return


def is_sold(ticker, order_id):
    while True:
        if is_order_filled(ticker, 'SELL', order_id):
            return True
        sleep(10)


def is_bought(ticker, order_id, cancel_buy=False, buy_time_out=300):
    start = time()
    while True:
        if is_order_filled(ticker, 'BUY', order_id):
            return True
        if cancel_buy and time() > start + buy_time_out:
            return False
        sleep(10)

def get_effective_num_decimals(number_in):
    zeros_front = 0
    while number_in < 1:
        zeros_front += 1
        number_in *= 10
    effective_num_decimals = zeros_front + 4 - 1
    return effective_num_decimals

"""
return example:
[
    [
        1499040000000,  # Open time
        "0.01634790",  # Open
        "0.80000000",  # High
        "0.01575800",  # Low
        "0.01577100",  # Close
        "148976.11427815",  # Volume
        1499644799999,  # Close time
        "2434.19055334",  # Quote asset volume
        308,  # Number of trades
        "1756.87402397",  # Taker buy base asset volume
        "28.46694368",  # Taker buy quote asset volume
        "17928899.62484339"  # Can be ignored
    ]
]
"""
def get_historical_price(ticker, interval, start_time):
    klines = client.get_historical_klines(symbol=ticker, interval=interval, start_str=start_time)
    # Sorted by oepn time
    sorted_klines = sorted(klines, key=lambda x: x[0])
    # list of pairs of (high price, low price)
    hl_prices = list(map(lambda x: ([float(x[2]), float(x[3])]), sorted_klines))
    return hl_prices


# return max of high price, min of low price, and average.
def get_last_4_hours_price(ticker):
    hl_prices = get_historical_price(ticker, interval=client.KLINE_INTERVAL_4HOUR, start_time='4 hours ago UTC')
    length = len(hl_prices)
    res = (sum([x[0] for x in hl_prices])/length,
           sum([x[1] for x in hl_prices])/length,
           sum([x[0] + x[1] for x in hl_prices])/length/2)
    print("Last 4 hours prices for ", ticker, ": ", res)
    return res

# return avg of high price, avg of low price, and average of all.
def get_last_30_mins_price(ticker):
    hl_prices = get_historical_price(ticker, interval=client.KLINE_INTERVAL_5MINUTE, start_time='30 minutes ago UTC')
    length = len(hl_prices)
    res = (sum([x[0] for x in hl_prices])/length,
           sum([x[1] for x in hl_prices])/length,
           sum([x[0] + x[1] for x in hl_prices])/length/2)
    print("Last 30 mins prices for ", ticker, ": ", res)
    return res

# return avg of high price, avg of low price, and average of all.
def get_last_1_hour_price(ticker):
    hl_prices = get_historical_price(ticker, interval=client.KLINE_INTERVAL_5MINUTE, start_time='1 hour ago UTC')
    length = len(hl_prices)
    res = (sum([x[0] for x in hl_prices])/length,
           sum([x[1] for x in hl_prices])/length,
           sum([x[0] + x[1] for x in hl_prices])/length/2)
    print("Last 1 hour prices for ", ticker, ": ", res)
    return res


def ref_4_hour_price(ticker, premium):
    # check last 4 hours prices.
    max_4, min_4, avg_4 = get_last_4_hours_price(ticker)
    if max_4 / min_4 < 1 + premium:
        print("Meaningless trading, since in the last 4 hours max price is {0}, min price is {1}, and average price "
              "is {2}.".format(max_4, min_4, avg_4))
        return False
    return True

def buy_and_sell_once(ticker, buy_price, sell_price, quantity, cancel_buy=True):
    buy_id, cost_price = place_limit_buy(ticker, buy_price, quantity)
    print('Buying...{0}, set the cost {1}, at the price {2}, and quantity {3}. '.format(ticker, cost_price * quantity, buy_price, quantity))
    sleep(10)

    if is_bought(ticker, buy_id):
      print('Bought')

    sell_id = place_limit_sell(ticker, sell_price, quantity)
    print('Selling...., the revenue: ', sell_price * quantity)
    sleep(10)

    if is_sold(sell_id, quantity):
      print('Sold')
    return True

# switch_trigger is gap percentage between 30 min avg price and 4 hour avg price.
# For example, switch_trigger is 0.1, if avg_30 > avg_4 * (1 + 0.1), then 30 min price will be used for reference.
# Otherwise 4 hour price will be used for reference.
def dynamic_buy_and_sell(ticker, time, total_cost, premium, switch_trigger, buy_time_out=300):
    n = 0
    while n < time:
        print('trade #: {0}'.format(n), " at timestamp: ", datetime.now())
        # check last 4 hours prices.
        max_4, min_4, avg_4 = get_last_4_hours_price(ticker)

        # check last 1 hour prices.
        max_1, min_1, avg_1 = get_last_1_hour_price(ticker)

        if avg_1 < avg_4*(1+switch_trigger) and avg_1 > avg_4*(1-switch_trigger):
            buy_price = min_1
            print("Trigger 1 hour price ref, buy price is :", buy_price)
            if max_1 / min_1 < 1 + premium:
                print(
                    "Meaningless trading, since in the last 1 hour max price is {0}, min price is {1}, and average price "
                    "is {2}.".format(max_1, min_1, avg_1))
                sleep(300)
                continue
        else:
            buy_price = min_4
            print("Trigger 4 hour price ref, buy price is :", buy_price)
            if max_4 / min_4 < 1 + premium:
                print(
                    "Meaningless trading, since in the last 4 hours max price is {0}, min price is {1}, and average price "
                    "is {2}.".format(max_4, min_4, avg_4))
                sleep(300)
                continue

        number_of_decimals = get_effective_num_decimals(buy_price)
        effective_buy_price = round(buy_price, 5)
        sell_price = round(buy_price*(1+premium), 5)
        quantity = int(total_cost/buy_price)
        buy_id, cost_price = place_limit_buy(ticker, effective_buy_price, quantity)
        print(
            'Buying...{0}, set the cost {1}, at the price {2}, and quantity {3}. '.format(ticker, cost_price * quantity,
                                                                                          effective_buy_price, quantity))
        sleep(10)

        if is_bought(ticker, buy_id, cancel_buy=True, buy_time_out=buy_time_out):
            print('Bought')
        else:
            place_cancel_order(ticker, order_id=buy_id)
            "Failed to buy, go to next loop"
            continue

        quantity = int(quantity*0.998)
        sell_id = place_limit_sell(ticker, sell_price, quantity)
        print('Selling...., the revenue: ', sell_price * quantity)
        sleep(10)

        if is_sold(ticker, sell_id):
            print('Sold')

        n += 1



def static_buy_and_sell(ticker, time, buy_price, total_cost, premium):
    n = 0
    while n < time:
        print('trade #: {0}'.format(time))
        buy_and_sell_once(ticker, buy_price, buy_price*(1 + premium), int(total_cost/buy_price))
        n += 1




# buy_and_sell(ticker='DOGEUSDT', time=1, buy_price=0.00618, total_cost=171, sell_percent=0.006)

# # fetch 30 minute klines for the last month of 2017
# klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")
#
# # fetch weekly klines since it listed
# klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")
