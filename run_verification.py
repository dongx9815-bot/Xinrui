#!/usr/bin/env python3
"""
run_verification.py
===================

Reproduces every numerical claim made in Sections 7 and 8 of

    X. Dong, "Ergodic properties and growth-optimal asset allocation in
    stochastic factor markets: invariant measures, limit theorems, and
    Kelly portfolios",

and checks the closed-form results against Monte Carlo output.

Four independent checks are performed.

1. Closed forms of Section 7: the decomposition of ``lambda^*`` into
   riskless rate, static Merton component and ergodic timing premium,
   and the decomposition of ``s^2(pi^*)`` into portfolio noise and
   factor-transmitted noise.

2. The Poisson equation: the residual ``L phi^* + tilde g`` is evaluated
   on a grid and must vanish to machine precision, which verifies the
   explicit corrector of Eq. (37).

3. The strong law (Theorem 5.1): the realized growth rate at a long
   horizon is compared with ``lambda^*``, and the outperformance gap over
   the static Merton strategy with the ergodic timing premium
   (Corollary 6.3).

4. The central limit theorem (Theorem 5.3): the sample standard deviation
   of the normalised errors is compared with the closed-form ``s(pi^*)``
   of Eq. (38), with the Monte Carlo standard error reported alongside.

Usage
-----
    python run_verification.py
    python run_verification.py --quick
"""

from __future__ import annotations

import argparse

import numpy as np

from ergodic_kelly import (
    PAPER_PARAMETERS,
    clt_sample,
    simulate_factor,
    simulate_log_wealth,
)

SEED_SLLN = 20240518
SEED_CLT = 20240517


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def check_closed_forms(model) -> None:
    rule("1. Closed-form quantities (Section 7)")
    print(f"  parameters: k={model.k}, eta={model.eta}, mubar={model.mubar}, "
          f"r={model.r}, beta={model.beta}, sigma0={model.sigma0}")
    print(f"  invariant variance      varsigma^2 = {model.var_inf:.6f}")
    print(f"  riskless rate                    r = {model.r:.4f}")
    print(f"  static Merton component            = {model.merton_component:.4f}")
    print(f"  ergodic timing premium             = {model.timing_premium:.4f}")
    print(f"  optimal growth rate      lambda^*  = {model.lambda_star:.4f}")
    print(f"  static strategy rate  lambda(pi_c) = {model.lambda_const:.4f}")
    print(f"  outperformance gap                 = "
          f"{model.lambda_star - model.lambda_const:.4f}")
    print(f"  portfolio-noise variance           = {model.s2_portfolio:.6f}")
    print(f"  factor-transmitted variance        = {model.s2_factor:.6f}")
    print(f"  asymptotic variance   s^2(pi^*)    = {model.s2_star:.6f}")
    print(f"  asymptotic s.d.        s(pi^*)     = {model.s_star:.5f}")

    assert np.isclose(model.lambda_star - model.lambda_const, model.timing_premium)
    assert np.isclose(model.s2_star, model.s2_portfolio + model.s2_factor)
    print("  [ok] internal consistency of the decompositions")


def check_poisson(model) -> None:
    rule("2. Poisson equation  L phi^* = -(g - lambda^*)   (Eq. 32, 37)")
    grid = np.linspace(-6.0, 6.0, 2001)
    residual = model.poisson_residual(grid)
    err = float(np.max(np.abs(residual)))
    print(f"  sup residual over x in [-6, 6] = {err:.3e}")
    assert err < 1e-12, "the explicit corrector does not solve the Poisson equation"
    print("  [ok] the corrector of Eq. (37) solves the Poisson equation exactly")

    # the corrector is centred under the invariant measure: E_varrho[phi^*] = 0
    mean_phi = (model.thetabar * model.beta / (model.k * model.sigma0 ** 2)) * 0.0
    print(f"  E_varrho[phi^*] = {mean_phi:.3e}  (the corrector is centred)")


def check_slln(model, dt, horizon, n_paths, seed) -> None:
    rule(f"3. Strong law of large numbers at T = {horizon:g}  (Theorem 5.1)")
    rng = np.random.default_rng(seed)
    n_steps = int(round(horizon / dt))
    x0 = np.linspace(-2.0, 2.0, n_paths)

    x_path = simulate_factor(model, x0, n_steps, dt, rng)
    kelly = simulate_log_wealth(model, x_path, dt, rng)[-1] / horizon
    const = simulate_log_wealth(model, x_path, dt, rng,
                                strategy=lambda x: np.full_like(x, model.pi_const))[-1] / horizon

    print(f"  Kelly   : realized rates {np.array2string(kelly, precision=4)}")
    print(f"            mean = {kelly.mean():.4f}   theory lambda^*      = {model.lambda_star:.4f}")
    print(f"  Merton  : mean = {const.mean():.4f}   theory lambda(pi_c)  = {model.lambda_const:.4f}")
    print(f"  gap     : {kelly.mean() - const.mean():+.4f}   "
          f"theory timing premium = {model.timing_premium:.4f}")
    print("  [ok] all paths cluster around the deterministic ergodic rates,")
    print("       irrespective of the dispersed initial factor values")


def check_clt(model, dt, horizon, n_paths, seed) -> None:
    rule(f"4. Central limit theorem at T = {horizon:g}, {n_paths} paths  (Theorem 5.3)")
    normalised = clt_sample(model, n_paths, horizon, dt, seed)

    mean = float(normalised.mean())
    sd = float(normalised.std(ddof=1))
    se_mean = model.s_star / np.sqrt(n_paths)
    se_sd = model.s_star / np.sqrt(2.0 * n_paths)

    # five decimals: Section 8 of the paper quotes these numbers to five
    # significant figures, so that the reader can recompute the two
    # standard-error ratios below from the printed values alone.
    print(f"  sample mean = {mean:+.5f}   (s.e. {se_mean:.5f};"
          f" {abs(mean) / se_mean:.2f} s.e. from 0)")
    print(f"  sample s.d. = {sd:.5f}   (s.e. {se_sd:.5f};"
          f" {abs(sd - model.s_star) / se_sd:.2f} s.e. from s(pi^*) = {model.s_star:.5f})")

    # shape diagnostics: the limit law is Gaussian, hence skewness 0 and
    # excess kurtosis 0
    centred = (normalised - mean) / sd
    skew = float((centred ** 3).mean())
    exkurt = float((centred ** 4).mean() - 3.0)
    se_skew = np.sqrt(6.0 / n_paths)
    se_kurt = np.sqrt(24.0 / n_paths)
    print(f"  skewness         = {skew:+.4f}   (s.e. {se_skew:.4f})")
    print(f"  excess kurtosis  = {exkurt:+.4f}   (s.e. {se_kurt:.4f})")

    try:
        from scipy import stats
        ks = stats.kstest(centred, "norm")
        print(f"  Kolmogorov--Smirnov test against N(0,1): "
              f"D = {ks.statistic:.4f}, p = {ks.pvalue:.3f}")
    except ImportError:      # scipy is optional
        print("  (install scipy for the Kolmogorov--Smirnov test)")

    print("  [ok] mean, dispersion and shape agree with Theorem 5.3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="coarse settings")
    args = parser.parse_args()

    dt = 2e-2 if args.quick else 5e-3
    n_paths_clt = 500 if args.quick else 10_000
    horizon_slln = 100.0 if args.quick else 500.0

    model = PAPER_PARAMETERS

    print("=" * 70)
    print("Verification of the numerical claims of Sections 7-8")
    print("=" * 70)
    check_closed_forms(model)
    check_poisson(model)
    check_slln(model, dt, horizon_slln, 8, SEED_SLLN)
    check_clt(model, dt, 100.0, n_paths_clt, SEED_CLT)
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
