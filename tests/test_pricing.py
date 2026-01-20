import numpy as np
import pytest

from heston_core.pricing import european_option_price_mc, price_for_strikes


def test_call_payoff_basic():
    S_T = np.array([80.0, 100.0, 120.0])
    K = 100.0
    r = 0.0
    T = 1.0

    price, se = european_option_price_mc(S_T, K, r, T, option_type="call")
    # payoffs = [0, 0, 20], mean=20/3
    assert price == pytest.approx(20.0 / 3.0)
    assert se > 0.0


def test_put_payoff_basic():
    S_T = np.array([80.0, 100.0, 120.0])
    K = 100.0
    r = 0.0
    T = 1.0

    price, se = european_option_price_mc(S_T, K, r, T, option_type="put")
    # payoffs = [20, 0, 0], mean=20/3
    assert price == pytest.approx(20.0 / 3.0)
    assert se > 0.0


def test_discounting_applied_correctly_constant_payoff_call():
    # Make payoff constant across paths: S_T always above K by 10
    S_T = np.full(1000, 110.0)
    K = 100.0
    r = 0.05
    T = 2.0

    price, se = european_option_price_mc(S_T, K, r, T, option_type="call")

    payoff = 10.0
    disc = np.exp(-r * T)

    assert price == pytest.approx(disc * payoff)
    # payoff constant => std error should be 0
    assert se == pytest.approx(0.0)


def test_option_type_case_insensitive():
    S_T = np.array([90.0, 110.0])
    K = 100.0
    r = 0.0
    T = 1.0

    p1, _ = european_option_price_mc(S_T, K, r, T, option_type="CALL")
    p2, _ = european_option_price_mc(S_T, K, r, T, option_type="call")
    assert p1 == pytest.approx(p2)


def test_put_call_parity_same_sample():
    """
    Using the same terminal sample S_T, MC estimators satisfy parity exactly
    up to floating point rounding:

        C - P = exp(-rT) * (E[S_T] - K)
    """
    rng = np.random.default_rng(0)
    S_T = rng.lognormal(mean=0.0, sigma=0.2, size=200_000) * 100.0

    K = 105.0
    r = 0.03
    T = 1.5

    C, _ = european_option_price_mc(S_T, K, r, T, option_type="call")
    P, _ = european_option_price_mc(S_T, K, r, T, option_type="put")

    disc = np.exp(-r * T)
    rhs = disc * (S_T.mean() - K)

    assert (C - P) == pytest.approx(rhs, rel=0, abs=1e-12)


def test_price_for_strikes_matches_single_strike():
    rng = np.random.default_rng(1)
    S_T = rng.normal(loc=100.0, scale=5.0, size=50_000)

    r = 0.01
    T = 1.0
    strikes = np.array([90.0, 100.0, 110.0])

    vec = price_for_strikes(S_T, strikes, r, T, option_type="call")

    singles = np.array([european_option_price_mc(S_T, K, r, T, "call")[0] for K in strikes])
    assert vec.shape == strikes.shape
    assert np.allclose(vec, singles, rtol=0, atol=1e-12)


def test_call_price_decreases_with_strike_for_fixed_sample():
    rng = np.random.default_rng(2)
    S_T = rng.lognormal(mean=0.0, sigma=0.15, size=100_000) * 100.0

    r = 0.0
    T = 1.0
    strikes = np.array([80.0, 100.0, 120.0])

    prices = price_for_strikes(S_T, strikes, r, T, option_type="call")
    assert prices[0] >= prices[1] >= prices[2]


def test_std_error_scales_with_sqrt_n():
    """
    Standard error should shrink ~ 1/sqrt(N). This is stochastic,
    so use a loose tolerance.
    """
    rng = np.random.default_rng(3)
    S_T_big = rng.normal(loc=100.0, scale=20.0, size=200_000)
    S_T_small = S_T_big[:50_000]

    K = 100.0
    r = 0.0
    T = 1.0

    _, se_small = european_option_price_mc(S_T_small, K, r, T, "call")
    _, se_big   = european_option_price_mc(S_T_big,   K, r, T, "call")

    # Expected ratio ~ sqrt(N_big / N_small) = sqrt(200k/50k) = 2
    ratio = se_small / se_big
    assert ratio == pytest.approx(2.0, rel=0.15)  # 15% slack
