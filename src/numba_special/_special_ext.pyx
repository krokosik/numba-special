# cython: language_level=3
# distutils: language = c
"""
C-level bridge into the ``scipy.special.cython_special`` kernels.

scipy ships its special-function kernels as Cython *capsules*
(``scipy.special.cython_special.__pyx_capi__``). The C signatures of those
capsules take and return C99 ``double complex`` *by value*. Python's
:mod:`ctypes` (and therefore Numba's ctypes-based foreign-function support)
cannot represent C99 complex values across the language boundary reliably:
on most ABIs passing/returning ``double complex`` by value is mapped to a
two-scalar struct whose register/stack placement disagrees between ctypes
and the platform C ABI, which corrupts memory or segfaults.

This module side-steps that entirely. It resolves the raw C addresses of the
scipy kernels at import time (via
:func:`numba.extending.get_cython_function_address`) and invokes them through
*typed C function pointers*, so the only place a ``double complex`` ever
appears by value is inside compiled C code, where the ABI is consistent. The
public entry points exposed to Numba use only plain scalars and raw
``double *`` pointers -- the calling convention Numba can lower natively in
nopython mode via a normal ``ctypes.CDLL`` attribute.

The exported C entry points (``ns_erfcx_r``, ``ns_erfcx_c``,
``ns_bind_erfcx``) are given default visibility so that a plain
``ctypes.CDLL(...).ns_erfcx_r`` lookup with ``argtypes``/``restype`` set
yields the function-pointer form Numba lowers natively -- the only ctypes
calling convention Numba supports without falling back to (broken) object
mode.

Currently this is a proof of concept for ``scipy.special.erfcx``; the same
machinery generalises to the whole special family.
"""

cdef extern from *:
    """
    #include <complex.h>

    /*
     * Cython lowers Python ``double complex`` to C99 ``double _Complex`` on
     * any C99-capable toolchain, which is exactly the type scipy's own
     * Cython kernels use internally, so calling scipy's capsules through
     * these typed function pointers reproduces scipy's own internal by-value
     * ABI exactly -- no struct/complex reinterpretation across a boundary.
     */

    /* Resolved at import time from the scipy capsules (see ns_bind_erfcx). */
    static double (*ns_erfcx_real_fp)(double x, int skip_dispatch) = NULL;
    static double complex (*ns_erfcx_cplx_fp)(double complex z, int skip_dispatch) = NULL;

    /* Bind the raw kernel addresses. Called once from Python at import. */
    __attribute__((visibility("default")))
    void ns_bind_erfcx(size_t real_fp, size_t cplx_fp) {
        ns_erfcx_real_fp = (double (*)(double, int)) real_fp;
        ns_erfcx_cplx_fp = (double complex (*)(double complex, int)) cplx_fp;
    }

    /* Real specialisation of erfcx: scalar in, scalar out. */
    __attribute__((visibility("default")))
    double ns_erfcx_r(double x) {
        return ns_erfcx_real_fp(x, 0);
    }

    /*
     * Complex specialisation of erfcx: real/imaginary in via plain doubles,
     * results out via raw ``double *`` pointers. The ``double complex`` is
     * constructed and consumed entirely in C, so no complex value ever
     * crosses the ctypes boundary -- only doubles and double pointers.
     */
    __attribute__((visibility("default")))
    void ns_erfcx_c(double re, double im, double *ore, double *oim) {
        double complex z = CMPLX(re, im);
        double complex r = ns_erfcx_cplx_fp(z, 0);
        *ore = creal(r);
        *oim = cimag(r);
    }
    """
    # Cython-visible declarations of the exported C entry points.
    void ns_bind_erfcx(size_t real_fp, size_t cplx_fp) noexcept nogil
    double ns_erfcx_r(double x) noexcept nogil
    void ns_erfcx_c(double re, double im, double *ore, double *oim) noexcept nogil


def _ensure_kernel_addresses_bound():
    """Idempotent import-time hook that resolves and binds the scipy capsule
    addresses for the real and complex specialisations of ``erfcx``.

    Uses the ``__pyx_capi__`` capsule names that scipy exposes for the fused
    real (``__pyx_fuse_1erfcx``) and complex (``__pyx_fuse_0erfcx``)
    specialisations.
    """
    from numba.extending import get_cython_function_address

    cdef size_t addr_real = <size_t> get_cython_function_address(
        'scipy.special.cython_special', '__pyx_fuse_1erfcx',
    )
    cdef size_t addr_cplx = <size_t> get_cython_function_address(
        'scipy.special.cython_special', '__pyx_fuse_0erfcx',
    )
    if addr_real == 0 or addr_cplx == 0:
        raise RuntimeError(
            "could not resolve scipy.special erfcx kernel addresses"
        )
    ns_bind_erfcx(addr_real, addr_cplx)