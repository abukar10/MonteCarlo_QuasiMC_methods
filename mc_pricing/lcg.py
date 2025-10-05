from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def lcg(a: int, c: int, m: int, seed: int, n: int):
    z = np.empty(n, dtype=np.int64)
    u = np.empty(n, dtype=np.float64)
    state = seed % m
    for i in range(n):
        state = (a * state + c) % m
        z[i] = state
        u[i] = state / m
    return u, z


def lattice_plots(outputs_dir: str):
    m = 2048
    a = 65
    c = 1
    seed = 0
    U, _ = lcg(a, c, m, seed, 2048)

    fig1, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(U[:-1], U[1:], ".", ms=2)
    axes[0].set_title("LCG lattice (a=65,c=1,m=2048)")
    axes[1].plot(U[:511], U[1:512], ".", ms=2)
    axes[1].set_title("LCG lattice (first 512)")
    savefig(fig1, f"{outputs_dir}/lcg_lattice1.png")

    a2 = 1365
    c2 = 1
    U2, _ = lcg(a2, c2, m, seed, 2048)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(U2[:-1], U2[1:], ".", ms=2)
    ax2.set_title("LCG lattice (a=1365,c=1,m=2048)")
    savefig(fig2, f"{outputs_dir}/lcg_lattice2.png")


def run(outputs_dir: str):
    lattice_plots(outputs_dir)
    return {"lcg": "done"}


