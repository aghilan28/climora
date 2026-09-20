# CLIMORA AI Anomaly Detection Documentation

## Overview
`src/anomaly/isolation_forest.py` implements multivariate climate anomaly detection using an Isolation Forest algorithm (`sklearn.ensemble.IsolationForest`).

## Configuration & Hyperparameters
- Default contamination: `0.05` ($5\%$ expected anomaly rate)
- Seed: `42` for exact reproducibility
- Sensitivity analysis evaluated across contamination levels $\in \{0.01, 0.05, 0.10\}$

## Outputs & Metrics
For each observation $i$:
1. `is_anomaly` (bool): `True` if flagged as an anomaly ($label = -1$).
2. `anomaly_score` (float): Raw decision function score.
3. `normalized_score` (float): Min-max scaled score in $[0.0, 1.0]$ where higher values indicate higher anomalousness.
4. `anomaly_rank` (int): Rank ordering of observations from most to least anomalous.

## Preserving Extremes vs Physical Invalidation
- **Physically Impossible Outliers**: Dropped during cleaning stage (e.g. temperature $< -90\text{ °C}$ or $> 60\text{ °C}$).
- **Extreme Climate Events**: Retained in feature matrix and flagged by Isolation Forest as genuine anomalies (e.g. historical heatwave events).
