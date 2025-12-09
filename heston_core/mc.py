# heston_core/mc.py

import numpy as np


def simulate_heston_paths(
    S0,
    v0,
    r,
    kappa,
    theta,
    sigma_v,
    rho,
    T,
    n_paths=20_000,
    n_steps=200,
    seed=None,
):
    """
    Simulate Heston paths using Euler discretization.

    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho
    """
    if seed is not None:
        np.random.seed(int(seed))

    dt = T / n_steps
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, n_steps + 1):
        z1 = np.random.randn(n_paths)
        z2 = np.random.randn(n_paths)

        dW1 = np.sqrt(dt) * z1
        dW2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)

        v_prev = np.maximum(v[t - 1, :], 0.0)

        v[t, :] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev) * dW2
        )
        v[t, :] = np.maximum(v[t, :], 0.0)

        S_prev = S[t - 1, :]
        S[t, :] = S_prev * np.exp(
            (r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW1
        )

    return S, v