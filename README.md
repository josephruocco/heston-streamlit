# Heston Model Monte Carlo Option Pricer (Streamlit)

An interactive Streamlit application for simulating the Heston stochastic volatility model and
pricing European options via Monte Carlo simulation.

---

## Overview

This project allows users to:

- Simulate asset price and variance paths under the Heston model
- Price European call and put options using Monte Carlo
- Visualize price paths, variance paths, terminal price distributions, and option prices across strikes
- Interactively explore model parameters through a Streamlit interface

Background references:
- https://en.wikipedia.org/wiki/Heston_model
- https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance
- https://en.wikipedia.org/wiki/European_option

---

## Model

Under the risk-neutral measure, the Heston model is given by:


dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S

dv_t = kappa (theta − v_t) dt + sigma_v sqrt(v_t) dW_t^v

with correlation \( \rho \) between the Brownian motions.

European option prices are estimated via discounted expected payoffs.

---

## Project Structure

```text
heston-streamlit/
├── app.py
└── heston_core/
    ├── mc.py        # Monte Carlo simulation
    ├── pricing.py  # Pricing utilities
    └── plots.py    # Plot construction
```

## Installation and Usage

```bash
Copy code
pip install streamlit numpy matplotlib
streamlit run app.py
```

### Notes
Monte Carlo estimates converge as the number of paths increases

Variance non-negativity is enforced at each timestep

The codebase is structured for easy extension to calibration or American options
 
