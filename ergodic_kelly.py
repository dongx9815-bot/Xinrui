"""
ergodic_kelly.py
================

Model, closed-form quantities and Monte Carlo kernels for

    X. Dong and E. V. Bulinskaya, "Ergodic properties and growth-optimal
    asset allocation in stochastic factor markets: invariant measures,
    limit theorems, and Kelly portfolios".

The module implements the Ornstein--Uhlenbeck specialisation of Section 7
of the paper:

    factor      dX_t = -k X_t dt + eta dW^1_t
    risky asset dS_t = S_t[ (mubar + beta X_t) dt + sigma0 dW^2_t ]
    bank        dB_t = r B_t dt

with W^1 and W^2 independent, so that d = n = 1 and m = 2.

Every symbol below carries the same name as in the manuscript, with the
following ASCII transliterations:

    k        mean-reversion speed of the factor          (Section 7)
    eta      factor volatility
    mubar    constant part of the risky drift
    beta     factor loading of the risky drift
    sigma0   volatility of the risky asset
    r        riskless rate
    thetabar mubar - r                                   (\bar\theta)
    var_inf  eta^2 / (2k), the invariant variance        (\varsigma^2)

Authors: Xinrui Dong <sinzhui.dun@math.msu.ru>
         Ekaterina V. Bulinskaya <bulinskaya@yandex.ru>
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Model",
    "PAPER_PARAMETERS",
    "simulate_factor",
    "simulate_log_wealth",
    "simulate_terminal_growth",
    "realized_growth_path",
]


# ----------------------------------------------------------------------
# Model and closed-form quantities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """Ornstein--Uhlenbeck factor market of Section 7.

    All closed-form quantities of Sections 6 and 7 are exposed as
    properties, so that the Monte Carlo output can be compared against
    them without re-deriving anything.
    """

    k: float = 2.0        # factor mean-reversion speed
    eta: float = 0.5      # factor volatility
    mubar: float = 0.08   # constant part of the risky drift
    beta: float = 0.3     # factor loading of the risky drift
    sigma0: float = 0.25  # volatility of the risky asset
    r: float = 0.02       # riskless rate

    # -- basic derived quantities --------------------------------------
    @property
    def thetabar(self) -> float:
        r"""\bar\theta = \bar\mu - r."""
        return self.mubar - self.r

    @property
    def var_inf(self) -> float:
        r"""Invariant variance \varsigma^2 = \eta^2 / (2k) of the factor."""
        return self.eta ** 2 / (2.0 * self.k)

    # -- optimal strategy and growth rates ------------------------------
    def kelly(self, x):
        r"""Kelly feedback \pi^*(x) = (\bar\mu + \beta x - r)/\sigma_0^2, Eq. (34)."""
        return (self.thetabar + self.beta * np.asarray(x)) / self.sigma0 ** 2

    @property
    def pi_const(self) -> float:
        r"""Static Merton weight \pi \equiv \bar\theta/\sigma_0^2."""
        return self.thetabar / self.sigma0 ** 2

    def growth_function(self, x, strategy=None):
        r"""Instantaneous growth function g_\pi(x) of Eq. (10).

        With ``strategy=None`` the Kelly rule is used, for which
        g_{\pi^*}(x) = r + |\theta(x)|^2 / 2.
        """
        x = np.asarray(x, dtype=float)
        if strategy is None:
            return self.r + (self.thetabar + self.beta * x) ** 2 / (2.0 * self.sigma0 ** 2)
        pi = np.asarray(strategy(x), dtype=float) if callable(strategy) else np.asarray(strategy, dtype=float)
        excess = self.thetabar + self.beta * x
        return self.r + pi * excess - 0.5 * pi ** 2 * self.sigma0 ** 2

    @property
    def merton_component(self) -> float:
        r"""Static Merton term \bar\theta^2 / (2\sigma_0^2)."""
        return self.thetabar ** 2 / (2.0 * self.sigma0 ** 2)

    @property
    def timing_premium(self) -> float:
        r"""Ergodic timing premium \beta^2\varsigma^2/(2\sigma_0^2) = \beta^2\eta^2/(4k\sigma_0^2)."""
        return self.beta ** 2 * self.var_inf / (2.0 * self.sigma0 ** 2)

    @property
    def lambda_star(self) -> float:
        r"""Optimal ergodic growth rate \lambda^*, Eq. (35)."""
        return self.r + self.merton_component + self.timing_premium

    @property
    def lambda_const(self) -> float:
        r"""Ergodic growth rate of the static Merton strategy."""
        return self.r + self.merton_component

    # -- Poisson corrector and asymptotic variance -----------------------
    def corrector(self, x):
        r"""Solution \phi^* of the Poisson equation (32), given in Eq. (37)."""
        x = np.asarray(x, dtype=float)
        a1 = self.thetabar * self.beta / (self.k * self.sigma0 ** 2)
        a2 = self.beta ** 2 / (4.0 * self.k * self.sigma0 ** 2)
        return a1 * x + a2 * (x ** 2 - self.var_inf)

    def corrector_derivative(self, x):
        r"""(\phi^*)'(x), Eq. (37)."""
        x = np.asarray(x, dtype=float)
        return (self.thetabar * self.beta / (self.k * self.sigma0 ** 2)
                + self.beta ** 2 * x / (2.0 * self.k * self.sigma0 ** 2))

    @property
    def s2_portfolio(self) -> float:
        r"""Direct portfolio-noise part of s^2(\pi^*): \varrho(|\theta|^2)."""
        return (self.thetabar ** 2 + self.beta ** 2 * self.var_inf) / self.sigma0 ** 2

    @property
    def s2_factor(self) -> float:
        r"""Factor-transmitted part of s^2(\pi^*), Eq. (38)."""
        pref = self.eta ** 2 * self.beta ** 2 / (self.k ** 2 * self.sigma0 ** 4)
        return pref * (self.thetabar ** 2 + self.beta ** 2 * self.var_inf / 4.0)

    @property
    def s2_star(self) -> float:
        r"""Asymptotic variance s^2(\pi^*) of the realized Kelly growth rate, Eq. (38)."""
        return self.s2_portfolio + self.s2_factor

    @property
    def s_star(self) -> float:
        r"""Asymptotic standard deviation s(\pi^*)."""
        return float(np.sqrt(self.s2_star))

    # -- self-consistency check ------------------------------------------
    def poisson_residual(self, x):
        r"""Residual of the Poisson equation, :math:`\mathcal L\phi^* + \tilde g`.

        Should vanish identically; used by ``run_verification.py`` as a
        symbolic-free check of Eq. (37).
        """
        x = np.asarray(x, dtype=float)
        phi_p = self.corrector_derivative(x)
        phi_pp = self.beta ** 2 / (2.0 * self.k * self.sigma0 ** 2)  # constant second derivative
        gen_phi = -self.k * x * phi_p + 0.5 * self.eta ** 2 * phi_pp
        g_tilde = self.growth_function(x) - self.lambda_star
        return gen_phi + g_tilde


PAPER_PARAMETERS = Model()
"""Parameter set used in Section 8 of the paper."""


# ----------------------------------------------------------------------
# Monte Carlo kernels
# ----------------------------------------------------------------------
def simulate_factor(model, x0, n_steps, dt, rng, exact=True):
    """Simulate the OU factor on a uniform grid.

    Parameters
    ----------
    model : Model
    x0 : float or array of shape (n_paths,)
        Initial factor value(s).
    n_steps : int
        Number of time steps.
    dt : float
        Step size.
    rng : numpy.random.Generator
    exact : bool, default True
        If ``True`` (the default), use the exact Gaussian transition of
        the OU process; if ``False``, use the Euler--Maruyama scheme. The
        paper samples the factor exactly, so that the only discretisation
        left is that of the time integrals in the log-wealth. Setting
        ``exact=False`` reproduces the Euler factor and quantifies the
        bias it introduces (0.57 Monte Carlo standard errors at
        ``dt = 5e-3``).

    Returns
    -------
    ndarray of shape (n_steps + 1, n_paths)
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n_paths = x0.size
    out = np.empty((n_steps + 1, n_paths))
    out[0] = x0
    x = x0.copy()

    if exact:
        decay = np.exp(-model.k * dt)
        scale = model.eta * np.sqrt((1.0 - decay ** 2) / (2.0 * model.k))
        for i in range(1, n_steps + 1):
            x = decay * x + scale * rng.standard_normal(n_paths)
            out[i] = x
    else:
        sqdt = np.sqrt(dt)
        for i in range(1, n_steps + 1):
            x = x - model.k * x * dt + model.eta * sqdt * rng.standard_normal(n_paths)
            out[i] = x
    return out


def simulate_log_wealth(model, x_path, dt, rng, strategy=None):
    r"""Log-wealth increments along a given factor path.

    Implements the decomposition (9) of the paper,

        log(V_T / V_0) = \int_0^T g_\pi(X_t) dt + \int_0^T \pi \sigma dW,

    discretised by Euler--Maruyama. The asset noise ``W^2`` is drawn
    independently of the factor noise ``W^1`` already contained in
    ``x_path``, in accordance with Section 7.

    Parameters
    ----------
    model : Model
    x_path : ndarray of shape (n_steps + 1, n_paths)
        Output of :func:`simulate_factor`.
    dt : float
    rng : numpy.random.Generator
    strategy : callable or None
        Feedback strategy ``x -> pi(x)``. ``None`` selects the Kelly rule.

    Returns
    -------
    ndarray of shape (n_steps + 1, n_paths)
        Cumulative ``log(V_t / V_0)``, starting at 0.
    """
    n_steps = x_path.shape[0] - 1
    n_paths = x_path.shape[1]
    x_left = x_path[:-1]                       # left endpoints (Ito convention)

    pi = model.kelly(x_left) if strategy is None else np.asarray(strategy(x_left), dtype=float)
    pi = np.broadcast_to(pi, x_left.shape)

    drift = model.growth_function(x_left, strategy) * dt
    diffusion = pi * model.sigma0 * np.sqrt(dt) * rng.standard_normal((n_steps, n_paths))

    log_wealth = np.empty((n_steps + 1, n_paths))
    log_wealth[0] = 0.0
    np.cumsum(drift + diffusion, axis=0, out=log_wealth[1:])
    return log_wealth


def simulate_terminal_growth(model, x0, horizon, dt, rng, strategy=None, exact=True):
    r"""Terminal realized growth rate :math:`T^{-1}\log(V_T/V_0)`.

    Streaming counterpart of :func:`simulate_factor` followed by
    :func:`simulate_log_wealth`: the factor and the log-wealth are
    advanced jointly and no path is retained, so the memory footprint is
    ``O(n_paths)`` rather than ``O(n_steps * n_paths)``. This is what
    makes the central limit theorem experiment feasible with a large
    number of paths.

    Parameters
    ----------
    model : Model
    x0 : array of shape (n_paths,)
        Initial factor values.
    horizon : float
        Terminal time ``T``.
    dt : float
        Step size of the time grid.
    rng : numpy.random.Generator
    strategy : callable or None
        Feedback strategy ``x -> pi(x)``. ``None`` selects the Kelly rule.
    exact : bool, default True
        Sample the factor from its exact Gaussian transition (default);
        set to ``False`` to revert the factor to Euler--Maruyama.

    Returns
    -------
    ndarray of shape (n_paths,)
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_paths = x.size
    n_steps = int(round(horizon / dt))
    sqdt = np.sqrt(dt)

    if exact:
        decay = np.exp(-model.k * dt)
        scale = model.eta * np.sqrt((1.0 - decay ** 2) / (2.0 * model.k))

    log_wealth = np.zeros(n_paths)
    for _ in range(n_steps):
        pi = model.kelly(x) if strategy is None else np.asarray(strategy(x), dtype=float)
        pi = np.broadcast_to(pi, x.shape)
        log_wealth += (model.growth_function(x, strategy) * dt
                       + pi * model.sigma0 * sqdt * rng.standard_normal(n_paths))
        if exact:
            x = decay * x + scale * rng.standard_normal(n_paths)
        else:
            x = x - model.k * x * dt + model.eta * sqdt * rng.standard_normal(n_paths)

    return log_wealth / horizon


def clt_sample(model, n_paths, horizon, dt, seed, exact=True):
    r"""Normalised errors :math:`\sqrt{T}(T^{-1}\log(V_T/V_0)-\lambda^*)`.

    The factor is started from its invariant measure
    :math:`\varrho=\mathcal N(0,\varsigma^2)`. The generator is created
    from ``seed`` inside this function, so that every script calling it
    with the same ``seed`` obtains bit-for-bit identical output.

    Returns
    -------
    ndarray of shape (n_paths,)
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, np.sqrt(model.var_inf), size=n_paths)
    growth = simulate_terminal_growth(model, x0, horizon, dt, rng, exact=exact)
    return np.sqrt(horizon) * (growth - model.lambda_star)


def realized_growth_path(log_wealth, dt):
    r"""Running realized growth rate :math:`t^{-1}\log(V_t/V_0)`.

    exact : bool, default True
        Sample the factor from its exact Gaussian transition (default);
        set to ``False`` to revert the factor to Euler--Maruyama.

    Returns
    -------
    times : ndarray of shape (n_steps,)
    growth : ndarray of shape (n_steps, n_paths)
        The initial instant ``t = 0`` is dropped, the ratio being
        undefined there.
    """
    n_steps = log_wealth.shape[0] - 1
    times = np.arange(1, n_steps + 1) * dt
    return times, log_wealth[1:] / times[:, None]
