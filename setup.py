"""Setuptools build script.

Defines the single Cython extension that bridges into the scipy.special
C kernels: it resolves the addresses of the ``scipy.special.cython_special``
Cython capsules at import time and exposes plain scalar / pointer C entry
points that Numba can call through a normal ``ctypes.CDLL`` attribute (the
only ctypes calling convention that Numba lowers natively, in nopython mode).
"""
import os

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - build-time only
    raise SystemExit(
        "Cython is required to build numba-special: pip install Cython"
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension(
        name="numba_specialz._special_ext",
        sources=["src/numba_specialz/_special_ext.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)