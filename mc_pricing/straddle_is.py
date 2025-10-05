from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def importance_sampling_straddle(N: int = 10000, r: float = 0.02, s0: float = 50.0, vol: float = 0.2, T: float = 1.0):
    a_30 = 2.55
    a_70 = -1.68
    disc = np.exp(-r * T)

    e_70 = np.random.randn(N)
    e_30 = np.random.randn(N)
    sT_call = s0 * np.exp(vol * (e_70 - a_70))
    sT_call = np.maximum(sT_call, 0.0)
    call = disc * np.maximum(sT_call - 70.0, 0.0) * np.exp(a_70 * e_70 - 0.5 * a_70 * a_70)

    sT_put = s0 * np.exp(vol * (e_30 - a_30))
    sT_put = np.maximum(sT_put, 0.0)
    put = disc * np.maximum(30.0 - sT_put, 0.0) * np.exp(a_30 * e_30 - 0.5 * a_30 * a_30)

    strad = call + put
    conv = np.cumsum(strad) / np.arange(1, N + 1)
    price = float(conv[-1])
    se = float(np.sqrt(np.sum((price - strad) ** 2) / (N * (N - 1))))
    return price, se, conv


def run(outputs_dir: str):
    price, se, conv = importance_sampling_straddle()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(conv) + 1), conv)
    ax.set_title("Convergence: Straddle (Importance Sampling)")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Monte Carlo Estimate")
    savefig(fig, f"{outputs_dir}/straddle_is_convergence.png")
    return {"straddle_price": price, "straddle_se": se}


