import pandas as pd
import matplotlib.pyplot as plt
K_s = 59500
K_b = 59750
R_s = 203.56
R_b = 32.03

term = 2 / 365

R_s_term = R_s * term
R_b_term = R_b * term

print (R_s_term, R_b_term)

upper_bond = K_s * (1 + R_s_term / 100) + K_b * R_b_term / 100
lower_bond = K_b / (1 + R_s_term / 100 + R_b_term / 100)

S_T = 0

#at time 0, you have a buy contract with strike price = K_b to buy 1 asset
# when price drops a lot, you buy one sell contract with strike price = K_s to sell 1 asset
# where K_s < K_b

#at time T, there are 3 scenarios:

# scenario 1: S_T > K_b > K_s, then buy contract expire, sell contract is excercised. now you have
portfolio_value_T = K_b * (1 + R_b_term / 100) + K_s * (1 + R_s_term / 100)

#scenario 2: K_b >= S_T >= K_s, then both contracts are excercised.
portfolio_value_T = (1 + R_b_term / 100) * S_T + (1 + R_s_term / 100) * K_s

#scenario 3: K_b > K_s > S_T, then buy contract is excercized, sell contract is expired.
portfolio_value_T = (1 + R_b_term / 100) * S_T + (1 + R_s_term / 100) * S_T

#if no trades, the portfolio value is
# portfolio_value_T_idle = S_T + K_b
price_at_termination = [50000 + i * 100 for i in range(300)]
portfolio_values_without_strategy = [val + K_b for val in price_at_termination]
portfolio_values_with_strategy = []
for S_T in price_at_termination:
    value = 0
    if S_T < K_s:
        value = (1 + R_b_term / 100) * S_T + (1 + R_s_term / 100) * S_T
    elif S_T <= K_b:
        value = (1 + R_b_term / 100) * S_T + (1 + R_s_term / 100) * K_s
    else:
        value = K_b * (1 + R_b_term / 100) + K_s * (1 + R_s_term / 100)
    portfolio_values_with_strategy.append(value)

df = pd.DataFrame(data={"asset_price": price_at_termination,
                        "value_without_strategy": portfolio_values_without_strategy,
                        "value_with_strategy": portfolio_values_with_strategy})

df.to_csv("test_strategy.csv", index=False)

df.plot(x='asset_price', kind='line', stacked=False)
plt.title('Area Plot')
plt.xlabel('asset price at termination')
plt.ylabel('portfolio values')
plt.show()