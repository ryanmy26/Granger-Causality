"""grangerspy: Granger-causality spectrum estimation in the frequency domain.

Python routines for estimating the unconditional and conditional
Granger-causality spectrum, paralleling the R package `grangers`.

Reference:
    Farnè, M., Yang, M. (2024). Comparing how Python and R estimate
    Granger-causality in the frequency domain.
    https://link.springer.com/chapter/10.1007/978-3-031-53717-2_20
"""

from .unconditional import granger_unconditional
from .conditional import granger_conditional

__version__ = "0.1.0"
__all__ = ["granger_unconditional", "granger_conditional"]
