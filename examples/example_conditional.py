"""Example: conditional Granger-causality spectrum on simulated data.

Simulates a trivariate VAR(1) process where y causes x directly and
z drives both series, then estimates the Granger-causality spectrum
of y on x conditional on z.
"""

import numpy as np

from grangerspy import granger_conditional


def simulate(T=300, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.6 * z[t - 1] + rng.standard_normal()
        y[t] = 0.4 * y[t - 1] + 0.3 * z[t - 1] + rng.standard_normal()
        x[t] = 0.3 * x[t - 1] + 0.5 * y[t - 1] + 0.3 * z[t - 1] + rng.standard_normal()
    return x, y, z


if __name__ == "__main__":
    x, y, z = simulate()

    result = granger_conditional(x, y, z, maxlag=4, ic="bic", trend="c", plot=True)

    print(f"Sample size:               {result['n']}")
    print(f"Lag order (x, z) model:    {result['lag_order_xz']}")
    print(f"Lag order (x, y, z) model: {result['lag_order_xyz']}")
    print(f"Mean GC y -> x | z:        {result['causality_y_to_x_on_z'].mean():.4f}")
