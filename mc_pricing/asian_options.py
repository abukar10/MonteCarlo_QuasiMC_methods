from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mc_pricing.utils import savefig


def bs_european_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def geometric_asian_bs(S0, K, r, sigma, T, m):
    sigsqT = sigma * sigma * T * (2 * m + 1) / (6 * m + 6)
    muT = 0.5 * sigsqT + 0.5 * (r - 0.5 * sigma * sigma) * T
    d1 = (np.log(S0 / K) + (muT + 0.5 * sigsqT)) / np.sqrt(sigsqT)
    d2 = d1 - np.sqrt(sigsqT)
    return np.exp(-r * T) * (S0 * np.exp(muT) * norm.cdf(d1) - K * norm.cdf(d2))


def asian_crude_and_vr(S0=100, K=90, sigma=0.2, r=0.05, T=1.0, Dt=0.1, n=150):
    m = int(T / Dt)
    ranvec = np.random.randn(n, m)
    Spath = S0 * np.cumprod(np.exp((r - 0.5 * sigma * sigma) * Dt + sigma * np.sqrt(Dt) * ranvec), axis=1)
    Spath = np.hstack([np.full((n, 1), S0), Spath])
    ESp = S0 * np.cumprod(np.exp(r * Dt) * np.ones((n, m)), axis=1)
    ESp = np.hstack([np.full((n, 1), S0), ESp])

    arithave = Spath.mean(axis=1)
    Parith = np.exp(-r * T) * np.maximum(arithave - K, 0.0)
    Pmean = float(Parith.mean())
    Pstderr = float(Parith.std(ddof=1) / np.sqrt(n))
    confmc = [Pmean - 1.96 * Pstderr, Pmean + 1.96 * Pstderr]

    C_bseu = bs_european_call(S0, K, r, sigma, T)
    europayoff = Spath[:, -1]
    Peuro = np.exp(-r * T) * np.maximum(europayoff - K, 0.0)
    cov_euro = np.cov(Parith, Peuro)
    c_euro = -cov_euro[0, 1] / cov_euro[1, 1]
    E = Parith + c_euro * (Peuro - C_bseu)
    Emean = float(E.mean())
    Estderr = float(E.std(ddof=1) / np.sqrt(n))
    confcv = [Emean - 1.96 * Estderr, Emean + 1.96 * Estderr]

    geo = geometric_asian_bs(S0, K, r, sigma, T, m)
    geoave = np.exp((1.0 / (m + 1)) * np.sum(np.log(Spath), axis=1))
    Pgeo = np.exp(-r * T) * np.maximum(geoave - K, 0.0)
    cov_geo = np.cov(Parith, Pgeo)
    c_geo = -cov_geo[0, 1] / cov_geo[1, 1]
    Z = Parith + c_geo * (Pgeo - geo)
    Zmean = float(Z.mean())
    Zstderr = float(Z.std(ddof=1) / np.sqrt(n))
    confcv_geo = [Zmean - 1.96 * Zstderr, Zmean + 1.96 * Zstderr]

    ranvec_anti = -ranvec
    Spath_anti = S0 * np.cumprod(np.exp((r - 0.5 * sigma * sigma) * Dt + sigma * np.sqrt(Dt) * ranvec_anti), axis=1)
    Spath_anti = np.hstack([np.full((n, 1), S0), Spath_anti])
    Uarithave = Spath_anti.mean(axis=1)
    Aarith = 0.5 * np.exp(-r * T) * (np.maximum(arithave - K, 0.0) + np.maximum(Uarithave - K, 0.0))
    Amean = float(Aarith.mean())
    Astderr = float(Aarith.std(ddof=1) / np.sqrt(n))
    confav = [Amean - 1.96 * Astderr, Amean + 1.96 * Astderr]

    N_vector = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(N_vector, Aarith, label="Antithetic payoffs")
    ax.axhline(geo, color="k", linestyle="--", label="Geo-Asian BS")
    ax.set_title("Antithetic variate for arithmetic Asian call")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Monte Carlo Estimates")
    ax.legend()
    return {
        "crude": (Pmean, Pstderr, confmc),
        "cv_euro": (Emean, Estderr, confcv),
        "cv_geo": (Zmean, Zstderr, confcv_geo),
        "anti": (Amean, Astderr, confav),
        "figure": fig,
    }


def run(outputs_dir: str):
    res = asian_crude_and_vr()
    savefig(res["figure"], f"{outputs_dir}/asian_antithetic.png")
    out = {k: v for k, v in res.items() if k != "figure"}
    return out


