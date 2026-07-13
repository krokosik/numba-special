"""Proof-of-concept tests: scipy.special.erfcx callable from numba.njit.

These verify the compiled C bridge + Numba overload against scipy's reference
implementation for both real and complex scalar arguments, and confirm that
erfcx can be used inside njit kernels mixing real and complex evaluation.
"""

import numpy as np
import pytest
import scipy.special as ss

import numba
import numba_specialz  # noqa: F401  (import registers the overloads)


@numba.njit
def _erfcx_real(x):
    return ss.erfcx(x)


@numba.njit
def _erfcx_complex(z):
    return ss.erfcx(z)


REAL_CASES = [0.0, 0.5, 1.0, -2.0, 10.0, 50.0, -25.0, 1e-3, -1e-3]
COMPLEX_CASES = [
    0.5 + 0.3j,
    1.0 - 1.0j,
    -2.0 + 3.0j,
    0.0 + 0.0j,
    1.5 + 0.0j,
    0.0 - 0.5j,
    5.0 + 5.0j,
    -3.0 - 4.0j,
    1e3 - 1e3j,
    3.14159 + 2.71828j,
    -1e2 + 1e2j,
]


@pytest.mark.parametrize("x", REAL_CASES)
def test_erfcx_real_matches_scipy(x):
    got = _erfcx_real(x)
    ref = ss.erfcx(x)
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-14), (x, got, ref)


@pytest.mark.parametrize("z", COMPLEX_CASES)
def test_erfcx_complex_matches_scipy(z):
    got = _erfcx_complex(z)
    ref = ss.erfcx(z)
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-14), (z, got, ref)


def test_erfcx_inside_mixed_njit_loop():
    @numba.njit
    def kernel(re, im):
        s = 0.0 + 0.0j
        for i in range(re.shape[0]):
            s += ss.erfcx(re[i]) * 0.5
            s += ss.erfcx(complex(re[i], im[i])) * 0.5
        return s

    re = np.array([0.0, 0.5, 1.0, 2.0])
    im = np.array([0.3, -0.5, 1.0, -2.0])

    got = kernel(re, im)
    ref = sum(
        ss.erfcx(r) * 0.5 + ss.erfcx(complex(r, im_i)) * 0.5
        for r, im_i in zip(re.tolist(), im.tolist())
    )
    assert np.isclose(got, ref, rtol=1e-12, atol=1e-14)
