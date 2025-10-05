from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def ibm_1968_sequence(n: int = 20000):
    a = 65539
    c = 0
    m = 2 ** 31
    x = 1
    xn = np.empty(n, dtype=np.int64)
    for i in range(n):
        x = (a * x + c) % m
        xn[i] = x
    un = xn / m
    return un


def plot_3tuple(un: np.ndarray, outputs_dir: str):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(un[:-2], un[1:-1], un[2:], linestyle="", marker=".", ms=2)
    ax.set_xlabel("u_i")
    ax.set_ylabel("u_{i+1}")
    ax.set_zlabel("u_{i+2}")
    ax.set_title("IBM 1968 RNG 3-tuples")
    return fig


def run(outputs_dir: str):
    un = ibm_1968_sequence(20000)
    fig = plot_3tuple(un, outputs_dir)
    savefig(fig, f"{outputs_dir}/ibm_rng_3tuple.png")
    return {"ibm_rng": "done"}


