# AGENTS.md

Notes for OpenCode sessions working in this repo. Read before changing the Cython bridge or Numba overloads.

## Build & install (non-obvious)

The Cython extension must be compiled in place before tests or imports work.
`uv sync` alone is NOT enough — it strips the build backend deps from the venv
because they live in `[build-system].requires`, which uv provisions in an
isolated build env that isn't reused locally.

```bash
uv sync
uv pip install setuptools wheel Cython          # build deps into the venv
uv pip install -e . --no-build-isolation        # compile _special_ext.so in place
```

After editing `src/numba_special/_special_ext.pyx`, re-run the
`uv pip install -e . --no-build-isolation` line to recompile.

`src/numba_special/_special_ext.{c,so,h}` are gitignored generated artifacts
— edit the `.pyx`, never the `.c`.

## Tests & lint

```bash
.venv/bin/python -m pytest tests/ -q               # full suite
.venv/bin/python -m pytest tests/test_erfcx.py::test_erfcx_real_matches_scipy -q   # single test
.venv/bin/ruff check src tests setup.py            # lint
.venv/bin/ruff format --check src tests            # format check (run `ruff format` to fix)
```

`pytest` will fail with `ImportError` if the `.so` was not rebuilt after a
`.pyx` change or a fresh checkout.

## Architecture (not obvious from filenames)

Goal: make `scipy.special.*` functions callable from `numba.njit`, including
for complex arguments — which the legacy `numba-scipy` package could not do.

Two-file mechanism:

- `src/numba_special/_special_ext.pyx` — Cython bridge. At import it resolves
  the raw C addresses of scipy's `cython_special` capsules (e.g.
  `__pyx_fuse_0erfcx`, `__pyx_fuse_1erfcx`) via
  `numba.extending.get_cython_function_address`, stores them as *typed C
  function pointers*, and exposes plain C entry points
  (`ns_erfcx_r`, `ns_erfcx_c`, `ns_bind_erfcx`) with default visibility.
- `src/numba_special/_overloads.py` — opens the extension's `.so` as a
  `ctypes.CDLL`, binds the entry points as module-level function-pointer
  globals with `argtypes`/`restype`, and registers
  `numba.extending.overload(scipy.special.erfcx)`.

Why the indirection (critical — do not "simplify"):

- scipy's complex kernels take/return C99 `double complex` **by value**.
  ctypes cannot represent that across the boundary; casting a capsule
  address via `ctypes.CFUNCTYPE` and calling it directly segfaults/misreads
  results on Linux (verified). The `.pyx` keeps the by-value complex call
  entirely in compiled C, where the ABI is consistent, and only passes plain
  `double` scalars and `double *` out-pointers across to Python.
- The C entry points are looked up as **attributes of a `ctypes.CDLL`**
  (with `argtypes`/`restype` set), not via `CFUNCTYPE`-cast function objects.
  This is the *only* ctypes calling convention Numba lowers natively in
  nopython mode; the capsule/CFUNCTYPE form falls back to a broken GIL-bound
  callback path in modern Numba. A typed ctypes function object is
  `typeof`-able; the enclosing `ctypes.CDLL` object is not, so keep the bound
  function pointers as module-level globals and reference those in the
  `overload` impl, not the `CDLL` itself.
- Complex out-values are written through `double *` via length-1 `np.empty`
  scratch arrays passed as `arr.ctypes`. Pass the arrays, not
  `arr.ctypes.data` (int) — a raw int is not a typed pointer for numba.

## Adding a new scipy.special function

1. In `_special_ext.pyx`: resolve its capsule name(s), add a typed function
   pointer, an `ns_<name>_r` / `ns_<name>_c` pair of `__attribute__((visibility("default")))`
   C shims, and extend `_ensure_kernel_addresses_bound`.
2. Recompile: `uv pip install -e . --no-build-isolation`.
3. In `_overloads.py`: bind via `ctypes.CDLL`, set `argtypes`/`restype`,
   and add an `@numba.extending.overload(scipy.special.<name>)` dispatcher.

## Numba overload gotchas

- The `impl` function's parameter name **must match** the typing function's
  parameter name (numba rejects mismatched names with "Typing and
  implementation arguments differ"). Use the same name in both signatures.
- Use `np.empty`/`np.zeros` for scratch out-arrays (length-1, `float64`),
  never Python `ctypes` array objects — the Python `ctypes` array isn't
  typeof-able by numba.
- Dispatch on `numba.types.Float` / `numba.types.Complex`, not on concrete
  `numba.types.float64`.

## Dependency note

`numba` and `scipy` versions are pinned in `pyproject.toml`; the capsule
names (`__pyx_fuse_*`) are scipy-private and can change across scipy
releases. If a kernel address resolves to 0 after a scipy upgrade, check
`scipy.special.cython_special.__pyx_capi__` for the new fused-specialisation
mangled name.