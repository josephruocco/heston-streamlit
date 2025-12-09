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

<img width="278" height="36" alt="Screenshot 2025-12-08 at 7 36 38 PM" src="https://github.com/user-attachments/assets/feac772e-9332-4b8c-bbc2-3117cb5d7c9e" />
<img width="1363" height="387" alt="Screenshot 2025-12-07 at 10 09 50 PM" src="https://github.com/user-attachments/assets/a6fd0934-f749-4a74-bacc-ae2e22e5bfe3" />

 
with correlation rho between the Brownian motions.

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
 
