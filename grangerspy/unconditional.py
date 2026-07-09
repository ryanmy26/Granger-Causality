"""Unconditional Granger-causality spectrum in the frequency domain."""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from ._utils import to_series, validate_inputs, frequency_grid, transfer_polynomial


def granger_unconditional(x, y, maxlag=1, ic="aic", trend="c", plot=False):
    """Estimate the unconditional Granger-causality spectrum between x and y.

    A bivariate VAR model is fitted to (x, y). The Granger-causality
    spectrum is computed on the frequency grid (0, 0.5] in both
    directions (y -> x and x -> y).

    Parameters
    ----------
    x, y : array-like or pandas.Series
        The two time series (same length).
    maxlag : int, default 1
        Maximum VAR lag order to consider.
    ic : {'aic', 'fpe', 'hqic', 'bic', None}, default 'aic'
        Information criterion used to select the VAR order (up to
        ``maxlag``). If None, ``maxlag`` is used directly.
    trend : {'c', 'ct', 'ctt', 'n'}, default 'c'
        Deterministic trend term passed to statsmodels VAR:
        constant, constant+trend, constant+linear+quadratic trend, or none.
    plot : bool, default False
        If True, plot the two causality spectra.

    Returns
    -------
    dict with keys:
        'frequency'          : frequency grid, cycles per unit time
        'causality_y_to_x'   : Granger-causality spectrum of y on x
        'causality_x_to_y'   : Granger-causality spectrum of x on y
        'n'                  : sample size
        'lag_order'          : selected VAR lag order
        'roots'              : roots of the VAR characteristic polynomial
        'params'             : estimated VAR coefficients
    """
    x = to_series(x, "x")
    y = to_series(y, "y")
    validate_inputs([x, y], maxlag)

    data = pd.concat([x, y], axis=1)
    results = VAR(data).fit(maxlags=maxlag, ic=ic, trend=trend)

    n = len(data)
    freqs = frequency_grid(n)

    # Residual covariance matrix
    Sigma = results.resid.cov().values

    # Normalizing (rotation) matrices for the intrinsic spectra
    P_x = np.array([[1.0, 0.0], [-Sigma[0, 1] / Sigma[0, 0], 1.0]])
    P_y = np.array([[1.0, -Sigma[1, 0] / Sigma[1, 1]], [0.0, 1.0]])

    I2 = np.eye(2)
    gc_y_to_x = np.zeros(len(freqs))
    gc_x_to_y = np.zeros(len(freqs))

    for l, f in enumerate(freqs):
        A = transfer_polynomial(results.coefs, f)
        M = I2 - A                      # I - A(f)
        H = np.linalg.inv(M)            # transfer function
        H_x = np.linalg.inv(P_x @ M)    # rotated transfer function (x eq.)
        H_y = np.linalg.inv(P_y @ M)    # rotated transfer function (y eq.)

        # Spectrum of x: intrinsic part + causal part from y
        sx_intr = 0.25 * H_x[0, 0] * Sigma[0, 0] * np.conj(H_x[0, 0])
        sx_caus = 0.25 * H[0, 1] * Sigma[1, 1] * np.conj(H[0, 1])
        gc_y_to_x[l] = np.log(abs(sx_intr + sx_caus) / abs(sx_intr))

        # Spectrum of y: intrinsic part + causal part from x
        sy_intr = 0.25 * H_y[1, 1] * Sigma[1, 1] * np.conj(H_y[1, 1])
        sy_caus = 0.25 * H[1, 0] * Sigma[0, 0] * np.conj(H[1, 0])
        gc_x_to_y[l] = np.log(abs(sy_intr + sy_caus) / abs(sy_intr))

    result = {
        "frequency": freqs,
        "causality_y_to_x": gc_y_to_x,
        "causality_x_to_y": gc_x_to_y,
        "n": n,
        "lag_order": results.k_ar,
        "roots": results.roots,
        "params": results.params,
    }

    if plot:
        _plot(freqs, gc_y_to_x, gc_x_to_y)

    return result


def _plot(freqs, gc_y_to_x, gc_x_to_y):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(freqs, gc_y_to_x, linewidth=1, color="grey")
    axes[0].set_title("Unconditional Granger-causality: y → x")
    axes[0].set_ylabel("GC spectrum")
    axes[1].plot(freqs, gc_x_to_y, linewidth=1, color="black")
    axes[1].set_title("Unconditional Granger-causality: x → y")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("GC spectrum")
    fig.tight_layout()
    plt.show()
