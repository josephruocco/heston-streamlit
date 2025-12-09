# heston_core/presets.py

from typing import Dict
import random

from .params import HestonParams


# ---- Fixed presets ----

# 1) Equity index style (SPX-like skew)
EQUITY_SKEW = HestonParams(
    kappa=2.0,
    theta=0.04,
    sigma_v=0.60,
    rho=-0.70,
    v0=0.04,
    r=0.02,
    q=0.00,
)

# 2) FX-style regime (weaker skew, lower vol-of-vol, faster mean reversion)
FX_STYLE = HestonParams(
    kappa=3.0,
    theta=0.02,
    sigma_v=0.30,
    rho=-0.20,
    v0=0.02,
    r=0.02,
    q=0.00,
)

# 3) Near Black–Scholes (small vol-of-vol, almost flat smile)
NEAR_BS = HestonParams(
    kappa=1.5,
    theta=0.04,
    sigma_v=0.15,
    rho=-0.05,
    v0=0.04,
    r=0.02,
    q=0.00,
)


PRESETS: Dict[str, Dict[str, object]] = {
    "equity_skew": {
        "label": "Equity skew (SPX-like)",
        "description": (
            "Strong negative skew and relatively high vol-of-vol. "
            "Typical for equity index options with crash risk priced in."
        ),
        "params": EQUITY_SKEW,
    },
    "fx_style": {
        "label": "FX-style",
        "description": (
            "Weaker skew, moderate vol-of-vol, faster mean reversion. "
            "Illustrative of FX / macro underlyings."
        ),
        "params": FX_STYLE,
    },
    "near_bs": {
        "label": "Near Black–Scholes",
        "description": (
            "Small vol-of-vol and mild correlation. "
            "Smile is almost flat; good for sanity checks."
        ),
        "params": NEAR_BS,
    },
}


# ---- Random preset within reasonable bounds ----

def sample_random_heston_params(seed: int = None) -> HestonParams:
    """
    Sample a random but reasonable Heston parameter set.

    Bounds avoid crazy unusable regimes while still exploring variety.
    """
    if seed is not None:
        random.seed(seed)

    kappa = random.uniform(0.5, 5.0)
    theta = random.uniform(0.01, 0.09)
    sigma_v = random.uniform(0.10, 1.00)
    rho = random.uniform(-0.90, 0.0)   # mostly negative skew (equity-like)
    v0 = random.uniform(0.01, 0.09)
    r = 0.02
    q = 0.00

    return HestonParams(
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
        r=r,
        q=q,
    )