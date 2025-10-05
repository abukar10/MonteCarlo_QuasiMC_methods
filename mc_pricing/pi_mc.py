from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def estimate_pi_basic(n: int = 1000):
    x = np.random.rand(n)
    y = np.random.rand(n)
    inside = x * x + y * y <= 1.0
    c = np.count_nonzero(inside)
    s = n
    p = c / s
    pi_estimate = 4.0 * p
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[~inside], y[~inside], s=8, c="#cc4444", label="outside")
    ax.scatter(x[inside], y[inside], s=8, c="#2a8", label="inside")
    ax.set_title(f"Monte Carlo Pi (n={n}) → {pi_estimate:.5f}")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    return pi_estimate, fig


def estimate_pi_convergence(m: int, n: int):
    data = np.zeros((n, m))
    finals = np.zeros(m)
    for j in range(m):
        x = np.random.rand(n)
        y = np.random.rand(n)
        inside_cum = np.cumsum((x * x + y * y) <= 1.0)
        i_arr = np.arange(1, n + 1)
        data[:, j] = 4.0 * (inside_cum / i_arr)
        finals[j] = data[-1, j]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data, color="k", alpha=0.3)
    ax.set_xlabel("nr of trials")
    ax.set_ylabel("estimate of pi")
    ax.set_title("Convergence of Monte Carlo Pi")
    return finals, fig


def run(outputs_dir: str):
    pi_est, fig1 = estimate_pi_basic(1000)
    savefig(fig1, f"{outputs_dir}/pi_basic.png")

    finals, fig2 = estimate_pi_convergence(m=20, n=1000)
    savefig(fig2, f"{outputs_dir}/pi_convergence.png")
    return {"pi_basic": pi_est, "pi_convergence_last_mean": float(finals.mean())}


