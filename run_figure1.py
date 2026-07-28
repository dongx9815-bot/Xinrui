#!/usr/bin/env python3
"""
run_figure1.py
==============

Regenerates Figure 1 of

    X. Dong, "Ergodic properties and growth-optimal asset allocation in
    stochastic factor markets: invariant measures, limit theorems, and
    Kelly portfolios",

that is, the file ``02_fig_simulation.pdf`` included in the manuscript.

Panel (a) illustrates the strong law of large numbers (Theorem 5.1):
eight independent trajectories of the realized growth rate
``T^{-1} log(V_T/V_0)``, started from widely dispersed initial factor
values, collapse onto the theoretical ergodic rate ``lambda^*``, while
the constant Merton strategy settles on the strictly smaller rate
``lambda(pi_const)``.

Panel (b) illustrates the central limit theorem (Theorem 5.3): the
histogram of the normalised errors at horizon ``T = 100`` across 10 000
independent paths, together with the fitted centred Gaussian density.

Both panels use fixed, independent seeds, so the figure is reproducible
bit for bit. The seed of panel (b) is shared with ``run_verification.py``,
which therefore reports exactly the numbers quoted in Section 8.

Usage
-----
    python run_figure1.py                 # full run, writes 02_fig_simulation.pdf
    python run_figure1.py --quick         # coarse run for a fast smoke test
    python run_figure1.py --out fig.pdf   # custom output file
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ergodic_kelly import (
    PAPER_PARAMETERS,
    clt_sample,
    realized_growth_path,
    simulate_factor,
    simulate_log_wealth,
)

# Configuration of Section 8 -------------------------------------------
DT = 5e-3            # Euler--Maruyama step
T_LONG = 500.0       # horizon of panel (a)
N_PATHS_A = 8        # trajectories in panel (a)
T_CLT = 100.0        # horizon of panel (b)
N_PATHS_B = 10_000   # trajectories in panel (b)
SEED_SLLN = 20240518
SEED_CLT = 20240517


def panel_a(model, ax, dt, horizon, n_paths, seed):
    """Almost-sure convergence of the realized growth rate."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(horizon / dt))
    x0 = np.linspace(-2.0, 2.0, n_paths)          # widely dispersed initial states
    x_path = simulate_factor(model, x0, n_steps, dt, rng)
    log_wealth = simulate_log_wealth(model, x_path, dt, rng)
    times, growth = realized_growth_path(log_wealth, dt)

    # Plot on a grid that is uniform in log T. The simulation itself uses
    # every one of the ~10^5 steps; only the drawing is decimated, which
    # is visually lossless on a logarithmic axis and keeps the vector
    # figure small.
    start = int(round(1.0 / dt)) - 1              # start the plot at T = 1
    idx = np.unique(np.round(
        np.geomspace(start + 1, times.size, num=2000)
    ).astype(int)) - 1
    ax.plot(times[idx], growth[idx], lw=0.8)
    ax.axhline(model.lambda_star, color="k", ls="--", lw=1.2,
               label=r"$\lambda^*$ (theory)")
    ax.axhline(model.lambda_const, color="k", ls=":", lw=1.2,
               label=r"$\lambda(\pi_{\mathrm{const}})$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$T^{-1}\log(V_T/V_0)$")
    ax.set_title("(a) A.s. convergence of the realized growth rate", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)


def panel_b(model, ax, dt, horizon, n_paths, seed):
    """Gaussian fluctuation of the realized growth rate at a finite horizon."""
    normalised = clt_sample(model, n_paths, horizon, dt, seed)

    s_hat = float(normalised.std(ddof=1))
    grid = np.linspace(normalised.min(), normalised.max(), 400)
    density = np.exp(-0.5 * (grid / s_hat) ** 2) / (s_hat * np.sqrt(2.0 * np.pi))

    ax.hist(normalised, bins=45, density=True, color="#a8cbe8",
            edgecolor="white", linewidth=0.3)
    ax.plot(grid, density, color="k", lw=1.4,
            label=r"$\mathcal{N}(0,\hat{s}^2)$, $\hat{s}=%.4f$" % s_hat)
    ax.set_xlabel(r"$\sqrt{T}\,(T^{-1}\log(V_T/V_0)-\lambda^*)$")
    ax.set_ylabel("density")
    ax.set_title(r"(b) CLT at $T=%d$ (%s paths)"
                 % (int(horizon), format(n_paths, ",")), fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
    return normalised


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="02_fig_simulation.pdf",
                        help="output file (default: 02_fig_simulation.pdf)")
    parser.add_argument("--quick", action="store_true",
                        help="coarse settings for a fast smoke test")
    args = parser.parse_args()

    dt = 2e-2 if args.quick else DT
    n_paths_b = 500 if args.quick else N_PATHS_B
    horizon_a = 100.0 if args.quick else T_LONG

    model = PAPER_PARAMETERS

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    panel_a(model, axes[0], dt, horizon_a, N_PATHS_A, SEED_SLLN)
    normalised = panel_b(model, axes[1], dt, T_CLT, n_paths_b, SEED_CLT)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")

    n = normalised.size
    mean, sd = normalised.mean(), normalised.std(ddof=1)
    se_mean = model.s_star / np.sqrt(n)
    se_sd = model.s_star / np.sqrt(2.0 * n)

    print(f"figure written to {args.out}")
    print(f"  theoretical  lambda*  = {model.lambda_star:.4f}")
    print(f"  theoretical  s(pi*)   = {model.s_star:.4f}")
    print(f"  sample mean = {mean:+.4f}  (s.e. {se_mean:.4f};"
          f" {abs(mean) / se_mean:.2f} s.e. from 0)")
    print(f"  sample s.d. = {sd:.4f}  (s.e. {se_sd:.4f};"
          f" {abs(sd - model.s_star) / se_sd:.2f} s.e. from s(pi*))")


if __name__ == "__main__":
    main()
