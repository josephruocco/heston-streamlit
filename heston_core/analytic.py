# heston_core/analytic.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

try:
    from scipy.integrate import quad
except ImportError as e:
    raise ImportError("scipy is required for analytic Heston pricing (scipy.integrate.quad).") from e


@dataclass(frozen=True)
class HestonParams:
    kappa: float   # mean reversion
    theta: float   # long-run variance
    sigma: float   # vol of vol
    rho: float     # correlation
    v0: float      # initial variance


def _heston_cf(
    u: complex,
    *,
    S0: float,
    r: float,
    T: float,
    params: HestonParams,
    K: float,
    trap: bool = True,
    j: int = 1,
) -> complex:
    """
    Characteristic function of log(S_T) under Heston, evaluated at u.
    We implement the 'Little Heston Trap' variant by default (trap=True),
    which is numerically more stable.

    j = 1 or 2 selects the P1/P2 integrand convention.
    """
    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0

    x0 = np.log(S0)
    a = kappa * theta

    # As in Heston: b differs for P1 vs P2
    if j == 1:
        b = kappa - rho * sigma
    elif j == 2:
        b = kappa
    else:
        raise ValueError("j must be 1 or 2")

    iu = 1j * u

    # d = sqrt((rho*sigma*iu - b)^2 + sigma^2*(iu + u^2))
    d = np.sqrt((rho * sigma * iu - b) ** 2 + (sigma ** 2) * (iu + u ** 2))

    # g = (b - rho*sigma*iu - d) / (b - rho*sigma*iu + d)
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    # Little Heston Trap: use 1/g to avoid blowups
    if trap:
        G = 1.0 / g
        exp_neg_dT = np.exp(-d * T)
        C = (r * iu * T) + (a / (sigma ** 2)) * (
            (b - rho * sigma * iu - d) * T - 2.0 * np.log((1.0 - G * exp_neg_dT) / (1.0 - G))
        )
        D = ((b - rho * sigma * iu - d) / (sigma ** 2)) * ((1.0 - exp_neg_dT) / (1.0 - G * exp_neg_dT))
    else:
        exp_dT = np.exp(d * T)
        C = (r * iu * T) + (a / (sigma ** 2)) * (
            (b - rho * sigma * iu + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = ((b - rho * sigma * iu + d) / (sigma ** 2)) * ((1.0 - exp_dT) / (1.0 - g * exp_dT))

    # CF of log(S_T): exp(C + D*v0 + i*u*x0)
    return np.exp(C + D * v0 + iu * x0)


def _Pj(
    j: int,
    *,
    S0: float,
    K: float,
    r: float,
    T: float,
    params: HestonParams,
    trap: bool = True,
    integ_upper: float = 200.0,
) -> float:
    """
    P1/P2 probability via numerical integration:
      Pj = 1/2 + 1/pi * ∫_0^∞ Re( e^{-i u ln K} * f_j(u) / (i u) ) du
    where f_j is the characteristic function with appropriate b (j=1,2).
    """
    lnK = np.log(K)

    def integrand(u: float) -> float:
        u_c = u + 0j
        cf = _heston_cf(u_c, S0=S0, r=r, T=T, params=params, K=K, trap=trap, j=j)
        numer = np.exp(-1j * u_c * lnK) * cf
        denom = 1j * u_c
        val = numer / denom
        return float(np.real(val))

    # Integrate from 0 to upper; upper ~ 100-300 often enough
    # quad handles oscillatory integrals decently with enough limit.
    integral, _ = quad(integrand, 0.0, float(integ_upper), limit=250)
    return 0.5 + (1.0 / np.pi) * integral


def heston_price_analytic(
    *,
    S0: float,
    K: float,
    r: float,
    T: float,
    params: HestonParams,
    option: Literal["call", "put"] = "call",
    trap: bool = True,
    integ_upper: float = 200.0,
) -> float:
    """
    Semi-closed Heston European option price using numerical integration.

    Call: C = S0*P1 - K*exp(-rT)*P2
    Put:  P via put-call parity
    """
    P1 = _Pj(1, S0=S0, K=K, r=r, T=T, params=params, trap=trap, integ_upper=integ_upper)
    P2 = _Pj(2, S0=S0, K=K, r=r, T=T, params=params, trap=trap, integ_upper=integ_upper)

    discK = K * np.exp(-r * T)
    call = S0 * P1 - discK * P2

    if option == "call":
        return float(call)
    elif option == "put":
        # put-call parity: P = C - S0 + K e^{-rT}
        put = call - S0 + discK
        return float(put)
    else:
        raise ValueError("option must be 'call' or 'put'")