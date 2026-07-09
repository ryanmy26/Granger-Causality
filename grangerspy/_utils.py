"""Shared helpers for grangerspy."""

import numpy as np
import pandas as pd


def to_series(data, name):
    """Convert array-like input to a named pandas Series."""
    if isinstance(data, pd.Series):
        s = data.reset_index(drop=True)
    else:
        s = pd.Series(np.asarray(data).ravel())
    s.name = name
    return s


def validate_inputs(series, maxlag):
    """Validate that series have equal length > 1 and maxlag is feasible."""
    n = len(series[0])
    if n <= 1:
        raise ValueError("Time series must contain more than one observation.")
    if any(len(s) != n for s in series):
        raise ValueError("All input time series must have the same length.")
    if maxlag > n - 1:
        raise ValueError(
            f"maxlag ({maxlag}) must be smaller than the series length ({n})."
        )


def frequency_grid(n):
    """Frequency grid on (0, 0.5], with step 1/n (as in the R package grangers)."""
    return np.arange(1.0 / n, 0.5 + 0.001, 1.0 / n)


def transfer_polynomial(coefs, freq):
    """Evaluate A(f) = sum_k A_k * exp(-2*pi*i*k*f) for a fitted VAR.

    Parameters
    ----------
    coefs : ndarray of shape (k_ar, neqs, neqs)
        VAR lag coefficient matrices (``results.coefs`` from statsmodels).
    freq : float
        Frequency in cycles per unit time.

    Returns
    -------
    ndarray of shape (neqs, neqs), complex
    """
    k_ar, neqs, _ = coefs.shape
    A = np.zeros((neqs, neqs), dtype=complex)
    for k in range(k_ar):
        A += coefs[k] * np.exp(-2j * np.pi * (k + 1) * freq)
    return A
