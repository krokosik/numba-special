"""numba-special: Numba-compatible bridges for scipy.special with complex support.

Importing this package registers Numba overloads so that supported
:mod:`scipy.special` functions -- starting with :func:`scipy.special.erfcx`
-- become callable from ``numba.njit`` code, including for complex
arguments (which the original numba-scipy could not handle due to C99
complex-by-value ABI limitations under ctypes).
"""

from . import _overloads as _overloads  # noqa: F401 (registers overloads)

__all__ = ["erfcx"]


def erfcx(x):
    """Numba-overloadable alias for :func:`scipy.special.erfcx`.

    Outside of ``numba.njit`` this simply forwards to scipy; inside njit it
    dispatches to the compiled C bridge for both real and complex inputs.
    """
    return _overloads.erfcx(x)
