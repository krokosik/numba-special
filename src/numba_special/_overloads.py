"""Numba overloads for :func:`scipy.special.erfcx`.

This is the proof-of-concept registration. It wires
:func:`scipy.special.erfcx` to the compiled C bridge in
:mod:`numba_special._special_ext` so it can be called from ``numba.njit``
code for both real and complex arguments.

Two things make this work:

* The bridge exposes plain C entry points (``ns_erfcx_r`` for real inputs,
  ``ns_erfcx_c`` for complex inputs) that take/return only ``double`` scalars
  and ``double *`` out-pointers. No C99 ``double complex`` ever crosses the
  ctypes boundary -- that's what sidesteps the complex-by-value ABI hazard
  that made the original numba-scipy approach unable to support complex
  numbers.
* The bridge functions are looked up as attributes of the extension's own
  ``ctypes.CDLL`` with ``argtypes``/``restype`` set, and :mod:`numpy` arrays
  are passed via their ``.ctypes`` handle for the ``double *`` out-params.
  This is the *only* ctypes calling convention Numba lowers natively in
  nopython mode; the capsule/CFUNCTYPE-cast forms fall back to a broken
  GIL-bound callback path in modern Numba.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numba
import numpy as np
import scipy.special

from . import _special_ext

__all__ = ["erfcx"]

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


def _load_bridge() -> ctypes.CDLL:
    """Open the compiled extension as a regular ``ctypes.CDLL`` so its
    exported C entry points can be bound with ``argtypes``/``restype`` --
    the form Numba lowers natively from nopython mode.
    """
    so_path = Path(_special_ext.__file__)
    lib = ctypes.CDLL(str(so_path))

    lib.ns_erfcx_r.argtypes = [ctypes.c_double]
    lib.ns_erfcx_r.restype = ctypes.c_double

    lib.ns_erfcx_c.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.ns_erfcx_c.restype = None

    # Resolve and bind the underlying scipy kernel addresses once, up front.
    _special_ext._ensure_kernel_addresses_bound()
    return lib


_BRIDGE = _load_bridge()

# Bind the exported C entry points as module-level function-pointer globals
# (with argtypes/restype set). This is the exact form Numba's ctypes support
# lowers natively from nopython mode: a typed ctypes function object is
# typeof-able, whereas the enclosing ``ctypes.CDLL`` object is not.
_NS_ERFCX_R = _BRIDGE.ns_erfcx_r
_NS_ERFCX_R.argtypes = [ctypes.c_double]
_NS_ERFCX_R.restype = ctypes.c_double

_NS_ERFCX_C = _BRIDGE.ns_erfcx_c
_NS_ERFCX_C.argtypes = [ctypes.c_double, ctypes.c_double, _DOUBLE_PTR, _DOUBLE_PTR]
_NS_ERFCX_C.restype = None


@numba.extending.overload(scipy.special.erfcx)
def _overload_erfcx(x):
    """Register a Numba implementation of ``scipy.special.erfcx``.

    Dispatches on the *numeric* nature of ``x``:

    * real floating-point input -> call ``ns_erfcx_r`` (scalar return);
    * complex input -> call ``ns_erfcx_c`` with two length-1 ``float64``
      scratch arrays used as the real/imaginary out-pointers, then reassemble
      a Python ``complex``.

    Only scalar arguments are supported, matching the per-element ufunc
    contract used inside ``numba.njit`` loops; array broadcasting is left to
    the caller/ufunc machinery as future work.
    """
    if isinstance(x, numba.types.Complex):

        def impl(x):
            ore = np.empty(1, dtype=np.float64)
            oim = np.empty(1, dtype=np.float64)
            _NS_ERFCX_C(x.real, x.imag, ore.ctypes, oim.ctypes)
            return complex(ore[0], oim[0])

        return impl

    if isinstance(x, numba.types.Float):

        def impl(x):
            return _NS_ERFCX_R(x)

        return impl

    return None


# Eagerly trigger overload registration on import of the package so that
# ``import numba_special`` is sufficient to make ``scipy.special.erfcx``
# usable from ``numba.njit``.
erfcx = scipy.special.erfcx
