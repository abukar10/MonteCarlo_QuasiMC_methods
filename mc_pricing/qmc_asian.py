from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig

try:
    from scipy.stats import qmc
except Exception:  # Fallback if SciPy version old
    qmc = None


def halton_norm(NSteps: int, NRepl: int):
    if qmc is None:
        # Pseudo fallback
        return np.random.randn(NRepl, NSteps)
    engine = qmc.Halton(d=NSteps, scramble=True)
    engine.fast_forward(1000)
    U = engine.random(NRepl)
    return np.sqrt(2) * erfinv(2 * U - 1)


def sobol_norm(NSteps: int, NRepl: int):
    if qmc is None:
        return np.random.randn(NRepl, NSteps)
    engine = qmc.Sobol(d=NSteps, scramble=True)
    engine.fast_forward(1000)
    U = engine.random(NRepl)
    return np.sqrt(2) * erfinv(2 * U - 1)


def erfinv(x):
    # Use SciPy if available, else polynomial approx
    try:
        from scipy.special import erfinv as sp_erfinv
        return sp_erfinv(x)
    except Exception:
        # Winitzki approximation
        a = 0.147
        ln = np.log(1 - x * x)
        s = (2 / (np.pi * a) + ln / 2)
        return np.sign(x) * np.sqrt(np.sqrt(s * s - ln / a) - s)


def asian_call_price_qmc(S0, K, mu, T, sigma, NSteps, NRepl, mode="halton"):
    dt = T / NSteps
    drift = (mu - 0.5 * sigma * sigma) * dt
    vari = sigma * np.sqrt(dt)
    if mode == "halton":
        z = halton_norm(NSteps, NRepl)
    else:
        z = sobol_norm(NSteps, NRepl)
    increments = drift + vari * z
    log_paths = np.cumsum(np.hstack([np.log(S0) * np.ones((NRepl, 1)), increments]), axis=1)
    SPaths = np.exp(log_paths)
    payoff = np.maximum(0.0, SPaths[:, 1 : NSteps + 1].mean(axis=1) - K)
    price = float(np.mean(np.exp(-mu * T) * payoff))
    return price


def run(outputs_dir: str):
    S0, K, mu, T, sigma, NSteps, NRepl = 100, 90, 0.05, 1.0, 0.2, 16, 10000
    p_halton = asian_call_price_qmc(S0, K, mu, T, sigma, NSteps, NRepl, mode="halton")
    p_sobol = asian_call_price_qmc(S0, K, mu, T, sigma, NSteps, NRepl, mode="sobol")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Halton", "Sobol"]
    vals = [p_halton, p_sobol]
    ax.bar(labels, vals)
    ax.set_title("QMC Asian call prices")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    savefig(fig, f"{outputs_dir}/qmc_asian.png")
    return {"halton": p_halton, "sobol": p_sobol}


