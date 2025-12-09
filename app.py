import numpy as np
import streamlit as st

from heston_core.mc import simulate_heston_paths
from heston_core.pricing import european_option_price_mc, price_for_strikes
from heston_core.plots import (
    price_paths_figure,
    variance_paths_figure,
    terminal_hist_figure,
    strike_sweep_figure,
)

from heston_core.params import HestonParams
from heston_core.presets import PRESETS, sample_random_heston_params


def sidebar_heston_params() -> HestonParams:
    st.sidebar.header("Heston Parameters")

    # ---- Preset selection ----
    preset_keys = list(PRESETS.keys())
    preset_labels = [PRESETS[k]["label"] for k in preset_keys]
    preset_labels.append("Random (bounded)")

    preset_index = st.sidebar.selectbox(
        "Preset",
        options=list(range(len(preset_labels))),
        format_func=lambda i: preset_labels[i],
    )

    if preset_index == len(preset_labels) - 1:
        # Random
        base_params = sample_random_heston_params()
        st.sidebar.caption(
            "Random parameter set sampled within reasonable bounds for stress testing."
        )
    else:
        key = preset_keys[preset_index]
        base_params = PRESETS[key]["params"]
        st.sidebar.caption(PRESETS[key]["description"])

    # ---- Sliders / number inputs using the preset as defaults ----
    v0 = st.sidebar.number_input(
        "Initial variance v₀",
        value=float(base_params.v0),
        min_value=0.0001,
    )
    kappa = st.sidebar.number_input(
        "Mean reversion κ",
        value=float(base_params.kappa),
        min_value=0.0001,
    )
    theta = st.sidebar.number_input(
        "Long-run variance θ",
        value=float(base_params.theta),
        min_value=0.0001,
    )
    sigma_v = st.sidebar.number_input(
        "Vol of variance σᵥ",
        value=float(base_params.sigma_v),
        min_value=0.0001,
    )
    rho = st.sidebar.slider(
        "Correlation ρ",
        min_value=-0.99,
        max_value=0.99,
        value=float(base_params.rho),
    )
    r = st.sidebar.number_input(
        "Risk-free rate r",
        value=float(base_params.r),
        min_value=-0.05,
        max_value=0.25,
        format="%.4f",
    )

    # (q is unused for now; keep 0.0)
    return HestonParams(
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
        r=r,
        q=0.0,
    )


def main():
    st.set_page_config(
        page_title="Heston Model Monte Carlo",
        layout="wide",
    )

    st.title("Heston Model – European Option Pricer (Monte Carlo)")
    st.markdown(
        """
    This app simulates the **[Heston stochastic volatility model](https://en.wikipedia.org/wiki/Heston_model)**
    and prices a **European option by [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance)**.
    """
    )

    # ----- Sidebar: option & market params -----
    st.sidebar.header("Option & Market Parameters")
    S0 = st.sidebar.number_input("Spot price S₀", value=100.0, min_value=0.01)
    K = st.sidebar.number_input("Strike K", value=100.0, min_value=0.01)
    T = st.sidebar.number_input("Maturity T (years)", value=1.0, min_value=0.01)

    # ----- Sidebar: Heston params (with presets) -----
    params = sidebar_heston_params()

    # ----- Sidebar: Monte Carlo settings -----
    st.sidebar.header("Monte Carlo Settings")
    n_paths = st.sidebar.number_input(
        "Number of paths",
        value=20_000,
        min_value=1_000,
        max_value=200_000,
        step=1_000,
    )
    n_steps = st.sidebar.number_input(
        "Time steps",
        value=200,
        min_value=10,
        max_value=1_000,
        step=10,
    )
    seed = st.sidebar.number_input(
        "Random seed (0 = none)", value=42, min_value=0, max_value=1_000_000
    )
    if seed == 0:
        seed = None

    option_type = st.radio("Option type", ["Call", "Put"], horizontal=True)

    run = st.button("Run Simulation & Price Option")

    if not run:
        st.info("Set parameters in the sidebar and click **Run Simulation & Price Option**.")
        return

    # ----- Simulation -----
    with st.spinner("Simulating Heston paths..."):
        S, v = simulate_heston_paths(
            S0=S0,
            v0=params.v0,
            r=params.r,
            kappa=params.kappa,
            theta=params.theta,
            sigma_v=params.sigma_v,
            rho=params.rho,
            T=T,
            n_paths=int(n_paths),
            n_steps=int(n_steps),
            seed=seed,
        )

    S_T = S[-1, :]

    price, se = european_option_price_mc(
        S_T=S_T, K=K, r=params.r, T=T, option_type=option_type.lower()
    )

    # ----- Top summary metrics -----
    col1, col2, col3 = st.columns(3)
    col1.metric(label=f"{option_type} price", value=f"{price:.4f}")
    col2.metric(label="Std. error (MC)", value=f"{se:.4f}")
    col3.metric(label="Mean terminal price E[S_T]", value=f"{S_T.mean():.4f}")

    st.caption(
        "Monte Carlo estimates converge as you increase the number of paths / steps, at the cost of runtime."
    )

    # ----- Tabs for visuals -----
    tab_paths, tab_var, tab_dist, tab_smile = st.tabs(
        ["Paths", "Variance", "Terminal distribution", "Strike sweep"]
    )

    with tab_paths:
        st.subheader("Sample underlying paths")
        fig = price_paths_figure(S, T, n_show=20)
        st.pyplot(fig)

    with tab_var:
        st.subheader("Sample variance paths")
        fig2 = variance_paths_figure(v, T, n_show=20)
        st.pyplot(fig2)

    with tab_dist:
        st.subheader("Distribution of terminal prices S(T)")
        fig3 = terminal_hist_figure(S_T)
        st.pyplot(fig3)
        st.write(
            f"Mean S(T): **{S_T.mean():.4f}**, "
            f"Std S(T): **{S_T.std(ddof=1):.4f}**"
        )

    with tab_smile:
        st.subheader("Price vs. strike (same simulation)")
        strike_low = st.number_input(
            "Strike min (as multiple of K)",
            value=0.5,
            min_value=0.1,
            max_value=2.0,
            step=0.1,
        )
        strike_high = st.number_input(
            "Strike max (as multiple of K)",
            value=1.5,
            min_value=0.1,
            max_value=3.0,
            step=0.1,
        )
        n_strikes = st.slider(
            "Number of strikes", min_value=5, max_value=30, value=15
        )

        if strike_low >= strike_high:
            st.error("Strike min must be < Strike max.")
        else:
            strikes = np.linspace(K * strike_low, K * strike_high, n_strikes)
            prices_strikes = price_for_strikes(
                S_T, strikes, params.r, T, option_type=option_type.lower()
            )
            fig4 = strike_sweep_figure(strikes, prices_strikes, option_type=option_type)
            st.pyplot(fig4)


if __name__ == "__main__":
    main()