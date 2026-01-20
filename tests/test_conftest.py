import pytest

@pytest.fixture
def base_params():
    # Stable “normal” parameters
    return dict(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        q=0.0,          # if you don't use dividend yield, keep it but ignore it in your wrapper
        kappa=2.0,
        theta=0.04,
        sigma=0.50,     # vol-of-vol
        rho=-0.70,
        v0=0.04,
        option_type="call",
    )