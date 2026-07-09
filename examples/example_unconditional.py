"""Example: unconditional Granger-causality spectrum on simulated data.

Simulates a bivariate VAR(1) process where y Granger-causes x
(but not the other way around), then estimates the unconditional
Granger-causality spectrum in both directions.
"""

import numpy as np

from grangerspy import granger_unconditional


def simulate(T=300, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        x[t] = 0.3 * x[t - 1] + 0.5 * y[t - 1] + rng.standard_normal()
    return x, y


if __name__ == "__main__":
    x, y = simulate()

    result = granger_unconditional(x, y, maxlag=5, ic="aic", trend="c", plot=True)

    print(f"Sample size:        {result['n']}")
    print(f"Selected lag order: {result['lag_order']}")
    print(f"Mean GC y -> x:     {result['causality_y_to_x'].mean():.4f}")
    print(f"Mean GC x -> y:     {result['causality_x_to_y'].mean():.4f}")
