import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def simulate_heston_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_paths: int = 20_000,
    n_steps: int = 200,
    seed: int = None,
):
    """
    Simulate Heston paths using Euler discretization.

    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, n_steps + 1):
        # Two independent standard normals
        z1 = np.random.randn(n_paths)
        z2 = np.random.randn(n_paths)

        # Correlated Brownian motions
        dW1 = np.sqrt(dt) * z1
        dW2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)

        # Previous step
        v_prev = np.maximum(v[t - 1, :], 0.0)

        # Variance process
        v[t, :] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev) * dW2
        )
        v[t, :] = np.maximum(v[t, :], 0.0)  # enforce non-negativity

        # Asset process
        S_prev = S[t - 1, :]
        S[t, :] = S_prev * np.exp(
            (r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW1
        )

    return S, v


def european_option_price_mc(S_T, K, r, T, option_type="call"):
    if option_type.lower() == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    disc_factor = np.exp(-r * T)
    price = disc_factor * payoff.mean()
    std_error = disc_factor * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return price, std_error


def main():
    st.title("Heston Model – European Option Pricer (Monte Carlo)")

    st.markdown(
        """
This app simulates the **Heston stochastic volatility model** and prices a
European call/put by **Monte Carlo**.

Model under risk-neutral measure:

- dSₜ = r Sₜ dt + √(vₜ) Sₜ dW₁ₜ  
- dvₜ = κ(θ − vₜ) dt + σᵥ √(vₜ) dW₂ₜ  
- corr(dW₁, dW₂) = ρ
"""
    )

    st.sidebar.header("Option & Market Parameters")
    S0 = st.sidebar.number_input("Spot price S₀", value=100.0, min_value=0.01)
    K = st.sidebar.number_input("Strike K", value=100.0, min_value=0.01)
    T = st.sidebar.number_input("Maturity T (years)", value=1.0, min_value=0.01)
    r = st.sidebar.number_input("Risk-free rate r", value=0.02, format="%.4f")

    st.sidebar.header("Heston Parameters")
    v0 = st.sidebar.number_input("Initial variance v₀", value=0.04, min_value=0.0001)
    kappa = st.sidebar.number_input("Mean reversion κ", value=2.0, min_value=0.0001)
    theta = st.sidebar.number_input("Long-run variance θ", value=0.04, min_value=0.0001)
    sigma_v = st.sidebar.number_input(
        "Vol of variance σᵥ", value=0.5, min_value=0.0001
    )
    rho = st.sidebar.slider("Correlation ρ", min_value=-0.99, max_value=0.99, value=-0.7)

    st.sidebar.header("MC Settings")
    n_paths = st.sidebar.number_input(
        "Number of paths", value=20_000, min_value=1000, max_value=200_000, step=1000
    )
    n_steps = st.sidebar.number_input(
        "Time steps", value=200, min_value=10, max_value=1000, step=10
    )
    seed = st.sidebar.number_input(
        "Random seed (0 = none)", value=42, min_value=0, max_value=1_000_000
    )
    if seed == 0:
        seed = None

    option_type = st.radio("Option type", ["Call", "Put"], horizontal=True)

    if st.button("Run Simulation & Price Option"):
        with st.spinner("Simulating Heston paths..."):
            S, v = simulate_heston_paths(
                S0=S0,
                v0=v0,
                r=r,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                T=T,
                n_paths=int(n_paths),
                n_steps=int(n_steps),
                seed=seed,
            )

        S_T = S[-1, :]
        price, se = european_option_price_mc(
            S_T=S_T, K=K, r=r, T=T, option_type=option_type.lower()
        )

        st.subheader("Option Price")
        st.write(
            f"**{option_type} price:** {price:.4f} (standard error ≈ {se:.4f})"
        )

        # Plot a few sample paths
        st.subheader("Sample Price & Variance Paths")
        n_show = min(10, S.shape[1])

        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, T, S.shape[0]), S[:, :n_show])
        ax.set_xlabel("Time")
        ax.set_ylabel("S(t)")
        ax.set_title("Sample underlying paths")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(np.linspace(0, T, v.shape[0]), v[:, :n_show])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("v(t)")
        ax2.set_title("Sample variance paths")
        st.pyplot(fig2)

        st.caption(
            "Monte Carlo estimates converge as you increase the number of paths / steps, at the cost of runtime."
        )


if __name__ == "__main__":
    main()