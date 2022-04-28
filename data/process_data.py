import numpy as np

# Hedge funds
hedge_funds = "ret_hedge.csv"
# Mutual funds
mutual_funds = "ret_agg.csv"

np_hedge = np.genfromtxt(hedge_funds, delimiter=',', skip_header=True, usecols=(1, 2))
np_mutual = np.genfromtxt(mutual_funds, delimiter=',', skip_header=True, usecols=(1, 2))

np.save("ret_hedge.npy", np_hedge)
np.save("ret_agg.npy", np_mutual)