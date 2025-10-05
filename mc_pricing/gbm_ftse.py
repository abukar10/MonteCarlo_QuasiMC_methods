from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def gbm_terminal_hist(S0: float, r: float, sigma: float, T: float, N: int, nbins: int = 40):
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * Z)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ST, bins=nbins, color="#4c78a8", alpha=0.8)
    ax.set_title("GBM terminal price histogram")
    return ST, fig


def gbm_paths(S0: float, mu: float, sigma: float, T: float, N: int, d: int = 3, nbins: int = 40):
    Dt = T / N
    increments = (mu - 0.5 * sigma * sigma) * Dt + sigma * np.sqrt(Dt) * np.random.randn(N, d)
    Spath = S0 * np.cumprod(np.exp(increments), axis=0)
    Spath = np.vstack([np.full((1, d), S0), Spath])
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(Spath)
    ax1.set_title("GBM sample paths")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(Spath.reshape(-1), bins=nbins, color="#72b7b2")
    ax2.set_title("Histogram of path values")
    return Spath, fig1, fig2


def run(outputs_dir: str):
    S0 = 6814.92
    r = 0.005
    T = 1.0
    sigma = 0.25
    N = 100000
    ST, fig_hist = gbm_terminal_hist(S0, r, sigma, T, N)
    savefig(fig_hist, f"{outputs_dir}/gbm_terminal_hist.png")

    Spath, fig_paths, fig_hist2 = gbm_paths(S0, r, sigma, T, 100000, d=3)
    savefig(fig_paths, f"{outputs_dir}/gbm_paths.png")
    savefig(fig_hist2, f"{outputs_dir}/gbm_paths_hist.png")
    return {"gbm_hist_mean": float(ST.mean()), "gbm_paths_last": Spath[-1, :].tolist()}


