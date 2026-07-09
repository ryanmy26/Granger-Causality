"""Conditional Granger-causality spectrum in the frequency domain."""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from ._utils import to_series, validate_inputs, frequency_grid, transfer_polynomial


def granger_conditional(x, y, z, maxlag=4, ic="aic", trend="c", plot=False):
    """Estimate the Granger-causality spectrum of y on x, conditional on z.

    Two VAR models are fitted: a bivariate model on (x, z) and a
    trivariate model on (x, y, z). The conditional Granger-causality
    spectrum y -> x | z is computed on the frequency grid (0, 0.5].

    Parameters
    ----------
    x, y, z : array-like or pandas.Series
        The three time series (same length). The causality measured is
        from ``y`` to ``x``, conditioning on ``z``.
    maxlag : int, default 4
        Maximum VAR lag order to consider.
    ic : {'aic', 'fpe', 'hqic', 'bic', None}, default 'aic'
        Information criterion used to select the VAR order (up to
        ``maxlag``). If None, ``maxlag`` is used directly.
    trend : {'c', 'ct', 'ctt', 'n'}, default 'c'
        Deterministic trend term passed to statsmodels VAR.
    plot : bool, default False
        If True, plot the conditional causality spectrum.

    Returns
    -------
    dict with keys:
        'frequency'               : frequency grid, cycles per unit time
        'causality_y_to_x_on_z'   : conditional Granger-causality spectrum
        'n'                       : sample size
        'lag_order_xz'            : selected lag order of the (x, z) VAR
        'lag_order_xyz'           : selected lag order of the (x, y, z) VAR
        'roots_xz', 'roots_xyz'   : VAR characteristic roots
        'params_xz', 'params_xyz' : estimated VAR coefficients
    """
    x = to_series(x, "x")
    y = to_series(y, "y")
    z = to_series(z, "z")
    validate_inputs([x, y, z], maxlag)

    # Bivariate model on (x, z)
    data1 = pd.concat([x, z], axis=1)
    results1 = VAR(data1).fit(maxlags=maxlag, ic=ic, trend=trend)

    # Trivariate model on (x, y, z)
    data2 = pd.concat([x, y, z], axis=1)
    results2 = VAR(data2).fit(maxlags=maxlag, ic=ic, trend=trend)

    n = len(data1)
    freqs = frequency_grid(n)

    # Residual covariances
    Sigma1 = results1.resid.cov().values
    Sigma2 = results2.resid.cov().values

    # Rotation matrix for the bivariate model
    P1 = np.array([[1.0, 0.0], [-Sigma1[0, 1] / Sigma1[0, 0], 1.0]])

    # Rotation matrix for the trivariate model (block orthogonalization)
    P2_a = np.array([
        [1.0, 0.0, 0.0],
        [-Sigma2[1, 0] / Sigma2[0, 0], 1.0, 0.0],
        [-Sigma2[2, 0] / Sigma2[0, 0], 0.0, 1.0],
    ])
    s11_c = Sigma2[1, 1] - Sigma2[1, 0] / Sigma2[0, 0] * Sigma2[0, 1]
    s21_c = Sigma2[2, 1] - Sigma2[2, 0] / Sigma2[0, 0] * Sigma2[0, 1]
    P2_b = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -s21_c / s11_c, 1.0],
    ])
    P2 = P2_a @ P2_b

    I2, I3 = np.eye(2), np.eye(3)
    gc = np.zeros(len(freqs))

    for l, f in enumerate(freqs):
        # Rotated transfer function of the bivariate (x, z) model
        A1 = transfer_polynomial(results1.coefs, f)
        H1 = np.linalg.inv(P1 @ (I2 - A1))

        # Rotated transfer function of the trivariate (x, y, z) model
        A2 = transfer_polynomial(results2.coefs, f)
        H2 = np.linalg.inv(P2 @ (I3 - A2))

        # Embed the bivariate transfer function into a 3x3 matrix
        G = np.zeros((3, 3), dtype=complex)
        G[0, 0] = H1[0, 0]
        G[0, 2] = H1[0, 1]
        G[2, 0] = H1[1, 0]
        G[2, 2] = H1[1, 1]
        G[1, 1] = 1.0

        Q = np.linalg.solve(G, H2)

        s_intr = Q[0, 0] * Sigma2[0, 0] * np.conj(Q[0, 0])
        s_y = Q[0, 1] * Sigma2[1, 1] * np.conj(Q[0, 1])
        s_z = Q[0, 2] * Sigma2[2, 2] * np.conj(Q[0, 2])
        gc[l] = np.log(abs(s_intr + s_y + s_z) / abs(s_intr))

    result = {
        "frequency": freqs,
        "causality_y_to_x_on_z": gc,
        "n": n,
        "lag_order_xz": results1.k_ar,
        "lag_order_xyz": results2.k_ar,
        "roots_xz": results1.roots,
        "roots_xyz": results2.roots,
        "params_xz": results1.params,
        "params_xyz": results2.params,
    }

    if plot:
        _plot(freqs, gc)

    return result


def _plot(freqs, gc):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, gc, linewidth=1, color="grey")
    plt.title("Conditional Granger-causality: y → x | z")
    plt.xlabel("Frequency")
    plt.ylabel("GC spectrum")
    plt.tight_layout()
    plt.show()
