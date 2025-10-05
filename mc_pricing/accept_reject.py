from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mc_pricing.utils import savefig


def acceptance_rejection(a: float = 2.0, n: int = 100):
    x = []
    xy = []
    rej = []
    rejy = []
    while len(x) < n:
        y = np.random.rand()
        u = np.random.rand()
        if u <= 2 * y / a:
            x.append(y)
            xy.append(u * a)
        else:
            rej.append(y)
            rejy.append(u * a)
    return np.array(x), np.array(xy), np.array(rej), np.array(rejy)


def run(outputs_dir: str):
    x, xy, rej, rejy = acceptance_rejection(2.0, 300)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(rej, rejy, "o", ms=3)
    ax1.set_title("Rejected samples")
    savefig(fig1, f"{outputs_dir}/accept_reject_rejected.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x, xy, "r*", ms=3)
    ax2.set_title("Accepted samples")
    savefig(fig2, f"{outputs_dir}/accept_reject_accepted.png")
    return {"accept_reject": len(x)}


