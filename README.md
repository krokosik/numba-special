# numba-specialz

> [!CAUTION]
> This project is WIP and only has a proof of concept. It is also written by GLM 5.2 with human oversight.

Numba-compatible bridges for `scipy.special`, with **complex-number support**.

`numba.njit` code cannot call `scipy.special` functions directly — they are
NumPy ufuncs backed by Cython capsules, which Numba's nopython mode does not
know how to dispatch. The legacy
[`numba-scipy`](https://github.com/numba/numba-scipy) package bridged that gap
for real arguments, but it is unmaintained, targets only older Python/NumPy
releases, and — critically — **cannot handle complex arguments**.

This package provides a fresh bridge that supports both real and complex
inputs, starting with `scipy.special.erfcx` as a proof of concept and built
to generalise across the special family.

## Why a new bridge is needed

`numba-scipy` resolved each scipy special-function kernel's C address with
`numba.extending.get_cython_function_address`, cast it to a `ctypes.CFUNCTYPE`
function pointer, and called it directly from inside `numba.njit`. That works
for kernels whose arguments and return values are plain `double`. It
**breaks** for kernels that take or return C99 `double complex` *by value*:

- `ctypes` has no native `double complex` type, so the by-value complex has
  to be approximated as a `{double re; double im;}` struct.
- On most ABIs (verified on Linux/x86-64) passing that struct by value uses a
  different register/stack placement than the C99 `_Complex` the kernel
  actually expects. The call returns garbage or segfaults.

This is not a ctypes bug that will be fixed; it is an ABI mismatch between
two different notions of "complex by value". Any bridge that calls the
capsule directly from Python-space ctypes inherits it.

## How this package works

`numba-special` keeps the by-value `double complex` call **entirely inside
compiled C**, where the ABI is consistent, and only exposes plain scalars
and raw pointers across the ctypes boundary.

It is a two-file mechanism:

1. **`src/numba_specialz/_special_ext.pyx`** — a tiny Cython extension. At
   import time it resolves the raw C addresses of scipy's
   `scipy.special.cython_special` capsules (e.g. `__pyx_fuse_0erfcx` for the
   complex specialisation, `__pyx_fuse_1erfcx` for the real one) and stores
   them as *typed C function pointers* whose signature matches scipy's
   internal one exactly. It then exposes plain C entry points with default
   visibility:

   - `ns_erfcx_r(double x) -> double` — real specialisation, scalar in/out.
   - `ns_erfcx_c(double re, double im, double *ore, double *oim)` — complex
     specialisation. The `re`/`im` pair is packed into a C99
     `double complex` *inside* the C shim, the scipy kernel is invoked
     through the typed pointer, and the result's real/imaginary parts are
     written out through `double *` pointers. No `double complex` ever
     crosses the ctypes boundary — only `double` and `double *`.

2. **`src/numba_specialz/_overloads.py`** — opens the compiled `.so` as a
   `ctypes.CDLL`, binds `ns_erfcx_r` / `ns_erfcx_c` as module-level
   function-pointer globals with `argtypes`/`restype` set, and registers a
   `numba.extending.overload(scipy.special.erfcx)` that dispatches on the
   argument type:

   - real `Float` input → call `ns_erfcx_r` and return the scalar;
   - `Complex` input → allocate two length-1 `float64` scratch arrays, call
     `ns_erfcx_c` passing them via `.ctypes`, and reassemble a `complex`.

Importing `numba_specialz` is enough to activate the overload — afterwards
`scipy.special.erfcx` Just Works inside `numba.njit`, for both real and
complex arguments.

### Why the `ctypes.CDLL`-attribute calling convention

Numba's ctypes support only lowers one calling convention natively in
nopython mode: a typed function object obtained as an *attribute of a
`ctypes.CDLL`* with `argtypes`/`restype` set. The capsule/`CFUNCTYPE`-cast
form that `numba-scipy` relied on falls back to a broken GIL-bound callback
path in modern Numba. So the bridge functions are deliberately exposed as
ordinary exported symbols of the extension's `.so` and bound the way Numba
can actually lower — and the bound function pointers are kept as module-level
globals (the typed `ctypes` function object is `typeof`-able; the enclosing
`CDLL` is not).

## Usage

```python
import numba
import scipy.special as ss

import numba_specialz  # registers the overloads on import


@numba.njit
def f(z):
    return ss.erfcx(z)


print(f(0.5))             # real
print(f(0.5 + 0.3j))      # complex — unsupported by numba-scipy
```

## Installation

The package includes a Cython extension that must be compiled in place.

```bash
uv sync
uv pip install setuptools wheel Cython          # build deps, not kept by uv sync
uv pip install -e . --no-build-isolation        # compile _special_ext.so in place
```

`uv sync` alone is not sufficient: it provisions `[build-system].requires`
in an isolated build environment that is not reused locally, so the build
dependencies must be installed into the venv explicitly and the editable
install run without build isolation.

After editing `src/numba_specialz/_special_ext.pyx`, re-run the
`uv pip install -e . --no-build-isolation` line to recompile.

## Testing & linting

The project uses [pytest](https://docs.pytest.org/) for tests and
[ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
.venv/bin/python -m pytest tests/ -q                          # run the suite
.venv/bin/python -m pytest tests/test_erfcx.py -q             # erfcx only
.venv/bin/python -m pytest tests/test_erfcx.py::test_erfcx_real_matches_scipy -q   # one test
.venv/bin/ruff check src tests setup.py                       # lint
.venv/bin/ruff format --check src tests                       # format check (run `ruff format` to fix)
```

Tests compare the bridged implementation against `scipy.special` directly, for
both real and complex scalar arguments and inside mixed real/complex
`numba.njit` loops.

## Status

Proof of concept: `scipy.special.erfcx` is fully bridged for real and complex
scalar arguments. The machinery (capsule-address resolution, typed C shims,
`ctypes.CDLL`-attribute overloads) is designed to generalise across the
`scipy.special` family; see `AGENTS.md` for the per-function recipe.

## Requirements

- Python ≥ 3.13
- `numba` ≥ 0.66
- `scipy` ≥ 1.18

Capsule names such as `__pyx_fuse_0erfcx` are scipy-private and can change
across scipy releases; if a kernel address resolves to `0` after a scipy
upgrade, check `scipy.special.cython_special.__pyx_capi__` for the new
mangled name.