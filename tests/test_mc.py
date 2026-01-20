import numpy as np
import pytest

from heston_core.mc import simulate_heston_paths


def test_shapes_and_initial_values():
    S0 = 100.0
    v0 = 0.04
    T = 1.0

    n_paths = 1234
    n_steps = 77

    S, v = simulate_heston_paths(
        S0=S0, v0=v0, r=0.01,
        kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7,
        T=T, n_paths=n_paths, n_steps=n_steps, seed=1
    )

    assert S.shape == (n_steps + 1, n_paths)
    assert v.shape == (n_steps + 1, n_paths)

    assert np.allclose(S[0, :], S0)
    assert np.allclose(v[0, :], v0)


def test_positivity_constraints():
    S0 = 100.0
    v0 = 0.04

    S, v = simulate_heston_paths(
        S0=S0, v0=v0, r=0.02,
        kappa=1.5, theta=0.04, sigma_v=1.0, rho=-0.9,
        T=2.0, n_paths=5000, n_steps=400, seed=2
    )

    # S is exp(...) so should be strictly positive
    assert np.all(S > 0.0)
    # variance is floored at 0
    assert np.all(v >= 0.0)


def test_seed_reproducibility():
    args = dict(
        S0=100.0, v0=0.04, r=0.01,
        kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7,
        T=1.0, n_paths=2000, n_steps=100
    )

    S1, v1 = simulate_heston_paths(**args, seed=123)
    S2, v2 = simulate_heston_paths(**args, seed=123)

    assert np.array_equal(S1, S2)
    assert np.array_equal(v1, v2)


def test_zero_time_returns_initial_state():
    S0 = 100.0
    v0 = 0.04

    S, v = simulate_heston_paths(
        S0=S0, v0=v0, r=0.05,
        kappa=2.0, theta=0.04, sigma_v=0.5, rho=0.3,
        T=0.0, n_paths=1000, n_steps=10, seed=1
    )

    # With T=0, dt=0, exp(0)=1 -> path should be constant
    assert np.allclose(S, S0)
    assert np.allclose(v, v0)


def test_risk_neutral_drift_sanity():
    """
    Under this discretization, the drift is roughly risk-neutral:
    E[S_T] should be close to S0 * exp(rT).
    This is a statistical test: use many paths and a loose tolerance.
    """
    S0 = 100.0
    v0 = 0.04
    r = 0.03
    T = 1.0

    S, _ = simulate_heston_paths(
        S0=S0, v0=v0, r=r,
        kappa=2.0, theta=0.04, sigma_v=0.6, rho=-0.6,
        T=T, n_paths=200_000, n_steps=400, seed=7
    )

    ST_mean = S[-1, :].mean()
    target = S0 * np.exp(r * T)

    # Allow ~1% error (Euler + stochastic error)
    assert ST_mean == pytest.approx(target, rel=0.01)


def test_correlation_wiring_of_increments():
    """
    This checks the intended correlation structure:
    corr(dW1, dW2) ≈ rho.
    We infer increments from S and v using the exact formulas used in the simulator.
    """
    S0 = 100.0
    v0 = 0.04
    r = 0.01
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.5
    rho = -0.75
    T = 1.0
    n_steps = 50
    n_paths = 200_000

    S, v = simulate_heston_paths(
        S0=S0, v0=v0, r=r,
        kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
        T=T, n_paths=n_paths, n_steps=n_steps, seed=11
    )

    dt = T / n_steps

    # use step 1 (t=1) only, v_prev = v[0] = constant v0
    v_prev = np.maximum(v[0, :], 0.0)

    # dW1 from log-return equation:
    # log(S1/S0) = (r - 0.5 v0) dt + sqrt(v0) dW1
    log_ret = np.log(S[1, :] / S[0, :])
    dW1 = (log_ret - (r - 0.5 * v_prev) * dt) / np.sqrt(v_prev)

    # dW2 from variance equation:
    # v1 = v0 + kappa(theta - v0)dt + sigma_v*sqrt(v0)*dW2, floored at 0
    # Use the pre-floor relationship: since v0 is positive and dt small, flooring rarely binds.
    dv = v[1, :] - v_prev
    dW2 = (dv - kappa * (theta - v_prev) * dt) / (sigma_v * np.sqrt(v_prev))

    corr = np.corrcoef(dW1, dW2)[0, 1]
    assert corr == pytest.approx(rho, abs=0.02)