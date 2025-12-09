# heston_core/plots.py

import numpy as np
import matplotlib.pyplot as plt


def price_paths_figure(S, T, n_show=20):
    n_show = min(n_show, S.shape[1])
    time_grid = np.linspace(0, T, S.shape[0])

    fig, ax = plt.subplots()
    ax.plot(time_grid, S[:, :n_show])
    ax.set_xlabel("Time")
    ax.set_ylabel("S(t)")
    ax.set_title(f"{n_show} sample price paths")
    return fig


def variance_paths_figure(v, T, n_show=20):
    n_show = min(n_show, v.shape[1])
    time_grid = np.linspace(0, T, v.shape[0])

    fig, ax = plt.subplots()
    ax.plot(time_grid, v[:, :n_show])
    ax.set_xlabel("Time")
    ax.set_ylabel("v(t)")
    ax.set_title(f"{n_show} sample variance paths")
    return fig


def terminal_hist_figure(S_T):
    fig, ax = plt.subplots()
    ax.hist(S_T, bins=50, density=True)
    ax.set_xlabel("S(T)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of terminal prices")
    return fig


def strike_sweep_figure(strikes, prices, option_type="Call"):
    fig, ax = plt.subplots()
    ax.plot(strikes, prices, marker="o")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option price")
    ax.set_title(f"{option_type} price vs. strike")
    return fig