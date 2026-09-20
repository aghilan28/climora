# CLIMORA AI Risk Classification Methodology Documentation

## Overview
`src/risk/` provides a dual-methodology climate risk classification engine. Both methodologies are fully data-derived, leak-free, and deterministic.

## Dual Risk Methodologies (`src/risk/methodology.py`)

### Method A — Distribution-Based (Empirical Quantiles)
Risk thresholds are computed from the empirical training-set distribution of temperature anomalies ($1880–1999$):
- **Low Risk**: Anomaly $\le 50\text{th percentile}$ ($+0.0700\text{ °C}$)
- **Moderate Risk**: $50\text{th percentile} < \text{Anomaly} \le 75\text{th percentile}$ ($+0.1600\text{ °C}$)
- **High Risk**: $75\text{th percentile} < \text{Anomaly} \le 90\text{th percentile}$ ($+0.2510\text{ °C}$)
- **Critical Risk**: Anomaly $> 90\text{th percentile}$ ($+0.2510\text{ °C}$)

### Method B — Literature-Anchored (+1.5°C / +2.0°C Policy Thresholds)
Anchored to the IPCC Paris Agreement climate policy targets relative to the $1850–1900$ pre-industrial baseline:
- GISTEMP baseline ($1951–1980$) offset relative to pre-industrial is $+0.0300\text{ °C}$ (stored in `config/risk.yaml`).
- Pre-industrial equivalent anomaly = $\text{GISTEMP Anomaly} + 0.0300\text{ °C}$.
- **Low Risk**: Pre-industrial anomaly $< 0.8\text{ °C}$
- **Moderate Risk**: $0.8\text{ °C} \le \text{Pre-industrial anomaly} < 1.5\text{ °C}$
- **High Risk**: $1.5\text{ °C} \le \text{Pre-industrial anomaly} < 2.0\text{ °C}$
- **Critical Risk**: Pre-industrial anomaly $\ge 2.0\text{ °C}$

## Risk Score Computation (0–100 Scale)
The continuous risk score combines standard anomaly departure with a 12-month rolling exceedance persistence term:
$$\text{Risk Score} = \min\left(100, \max\left(0, \frac{\text{Anomaly} - \mu_{\text{train}}}{\sigma_{\text{train}}} \times 25 + 50 + \text{Persistence Bonus}\right)\right)$$

## Mandatory Disclaimer
This risk index is an analytical index derived from dataset metrics and models. It is not an operational climate safety certification.
