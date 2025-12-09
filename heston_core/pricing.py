# heston_core/pricing.py

import numpy as np


def european_option_price_mc(S_T, K, r, T, option_type="call"):
    """
    Monte Carlo estimator for European call/put.
    """
    option_type = option_type.lower()

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    disc_factor = np.exp(-r * T)
    price = disc_factor * payoff.mean()
    std_error = disc_factor * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, std_error


def price_for_strikes(S_T, strikes, r, T, option_type="call"):
    """
    Use one set of terminal prices S_T to price multiple strikes.
    """
    prices = []
    for K in strikes:
        p, _ = european_option_price_mc(S_T, K, r, T, option_type=option_type)
        prices.append(p)
    return np.array(prices)