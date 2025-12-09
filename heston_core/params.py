# heston_core/params.py

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class HestonParams:
    """
    Container for Heston model parameters under the risk-neutral measure.
    Names are chosen to match your existing mc/pricing functions.
    """
    kappa: float      # mean reversion speed
    theta: float      # long-run variance level
    sigma_v: float    # vol of variance
    rho: float        # correlation between asset & variance Brownian motions
    v0: float         # initial variance
    r: float = 0.02   # risk-free rate
    q: float = 0.0    # dividend yield (not used yet, but handy later)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (handy for Streamlit display, JSON, etc.)."""
        return asdict(self)