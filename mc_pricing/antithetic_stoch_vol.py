from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mc_pricing.utils import savefig


def implied_volatility_call_bs(S: float, K: float, r: float, T: float, price: float, q: float = 0.0):
    # Simple bisection on vol
    def bs_call(sigma):
        if sigma <= 0:
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    lo, hi = 1e-6, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = bs_call(mid)
        if val > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def simulate_antithetic(N: int = 50000, T: float = 0.25, dt: float = 1 / 120, y0: float = 0.04, s0: float = 99.7503):
    N_half = N // 2
    M = int(T / dt)
    disc = np.exp(-0.01 * T)
    c = np.empty(2 * N_half)
    for i in range(N_half):
        y = y0
        y1 = y0
        s = s0
        s1 = s0
        for _ in range(1, M):
            xi = np.random.randn()
            xj = np.random.randn()
            y = y + 2 * (0.04 - y) * dt + 0.2 * y * np.sqrt(dt) * xi
            y = max(y, 0.0)
            s = s * (1 + 0.01 * dt + np.sqrt(y * dt) * (-0.7 * xi + np.sqrt(0.51) * xj))
            s = max(s, 0.0)

            # Antithetic
            y1 = y1 + 2 * (0.04 - y1) * dt + 0.2 * y1 * np.sqrt(dt) * (-xi)
            y1 = max(y1, 0.0)
            s1 = s1 * (1 + 0.01 * dt + np.sqrt(y1 * dt) * (-0.7 * (-xi) + np.sqrt(0.51) * (-xj)))
            s1 = max(s1, 0.0)

        c[i] = disc * max(s - 100.0, 0.0)
        c[i + N_half] = disc * max(s1 - 100.0, 0.0)
    avgc = float(c.mean())
    se = float(np.sqrt(np.sum((c - avgc) ** 2) / (N * (N - 1))))
    return c, avgc, se


def run(outputs_dir: str):
    c, avgc, se = simulate_antithetic()
    N = len(c)
    conv = np.cumsum(c) / np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, N + 1), conv)
    ax.set_title("Convergence: Antithetic Stoch-Vol Call")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Monte Carlo Estimate")
    savefig(fig, f"{outputs_dir}/stoch_vol_antithetic_conv.png")

    iv = implied_volatility_call_bs(99.7503, 100.0, 0.01, 0.25, avgc)
    return {"stoch_vol_call": avgc, "stderr": se, "implied_vol": iv}


