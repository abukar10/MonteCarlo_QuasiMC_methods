from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mc_pricing.utils import savefig


def crude_put_mc(S: float, K: float, r: float, sigma: float, T: float, M: int = 100000):
    Z = np.random.randn(M)
    ST = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    present_vals = disc * payoffs
    mean_val = float(present_vals.mean())
    stderr = float(present_vals.std(ddof=1) / np.sqrt(M))
    ci = [mean_val - 1.96 * stderr, mean_val + 1.96 * stderr]
    return mean_val, stderr, ci


def bs_call_price(s: float, K: float, r: float, q: float, v: float, T: float):
    if T <= 0 or v <= 0:
        return max(s * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(s / K) + (r - q + 0.5 * v * v) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return s * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(s: float, K: float, r: float, q: float, v: float, T: float):
    if T <= 0 or v <= 0:
        return max(K * np.exp(-r * T) - s * np.exp(-q * T), 0.0)
    d1 = (np.log(s / K) + (r - q + 0.5 * v * v) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - s * np.exp(-q * T) * norm.cdf(-d1)


def antithetic_call_mc(S0: float, K: float, r: float, q: float, sigma: float, T: float, N: int = 10000):
    e = np.random.randn(N)
    ST = S0 * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * e)
    ST1 = S0 * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * (-e))
    payoff = 0.5 * (np.maximum(ST - K, 0.0) + np.maximum(ST1 - K, 0.0))
    price = np.exp(-r * T) * float(payoff.mean())
    stderr = float((np.exp(-r * T) * payoff).std(ddof=1) / np.sqrt(N))
    return price, stderr


def hull_example_call(S0=75.0, K=72.0, r=0.03, sigma=0.35, T=0.75, N=10000):
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * Z)
    fcall = np.maximum(ST - K, 0.0)
    disc_val = np.exp(-r * T) * fcall
    mc_call = float(disc_val.mean())
    stderr = float(disc_val.std(ddof=1) / np.sqrt(N))
    ci = [mc_call - 1.96 * stderr, mc_call + 1.96 * stderr]
    return mc_call, stderr, ci


def run(outputs_dir: str):
    # Crude put MC
    put_val, put_se, put_ci = crude_put_mc(S=55, K=50, r=0.05, sigma=0.25, T=3.0, M=100000)

    # Antithetic call MC vs BS
    S0, K, r, q, v, T = 75, 72, 0.03, 0.0, 0.35, 0.75
    bs = bs_call_price(S0, K, r, q, v, T)
    anti_price, anti_se = antithetic_call_mc(S0, K, r, q, v, T, 10000)

    # Hull example MC call with CI
    mc_call, stderr, ci = hull_example_call()

    # Simple bar chart of prices
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Put (MC)", "Call (BS)", "Call (AntiMC)", "Hull Call (MC)"]
    values = [put_val, bs, anti_price, mc_call]
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title("Vanilla options: prices")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    savefig(fig, f"{outputs_dir}/vanilla_prices.png")

    return {
        "put_value": put_val,
        "put_ci": put_ci,
        "bs_call": bs,
        "anti_call": anti_price,
        "anti_stderr": anti_se,
        "hull_call": mc_call,
        "hull_ci": ci,
    }


