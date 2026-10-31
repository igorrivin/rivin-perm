# CUDA source tools

CUDA support in this repository is source-only. The supported Tier-3 sources and selected
Python wrappers are available in a repository checkout and are intentionally
included in the source distribution for reproducibility. They are not listed
as installed Python modules, are not included in wheels, are not built by
`pip install`, and are not selected automatically by the packaged CPU APIs.
The research-only Tier-5 CUDA utilities in the inventory below are excluded
from both source and wheel distributions.

These are supported source tools, separate from installed wheel APIs.

## Tool inventory

| Source and wrapper | Purpose | Status |
|---|---|---|
| `perm_binary_cuda.cu`, `perm_binary_cuda.py` | Modular and exact zero-one permanents using the binary preprocessing, bounds, and CRT plan | Tier 3; focused wrapper and device tests |
| `perm_mod3_cuda.cu`, `perm_modp_cuda.cu`, `perm_cuda.py` | Characteristic-three, general modular, and Schur-permanent CUDA kernels | Tier 3; focused wrapper, modular-device, and Schur-device tests |
| `perm_parallel_cuda.cu`, `perm_parallel.py` | Floating-point real and complex permanent of one matrix | Tier 3; wrapper tests and opt-in device tests |
| `perm_enum_cuda.cu` | Finite-field matrix enumeration | Tier 3 research utility with opt-in device tests |
| `perm_proj_cuda.cu` | Prime-field projective-multiset enumeration | Tier 3 research utility with opt-in device tests |

The root CUDA sources have different input and arithmetic contracts. A
successful build of one is not evidence that another supports the same order,
modulus, coefficient range, or matrix class. Consult its source header,
wrapper, focused tests, and any adjacent assessment before a large run.

## Manual binary-backend build

For the zero-one backend, keep the `lib` prefix. Naming the plain C ABI
library `perm_binary_cuda.so` can shadow the Python wrapper during import.

```bash
nvcc -shared -Xcompiler -fPIC -O3 -std=c++17 -arch=native \
    -o libperm_binary_cuda.so perm_binary_cuda.cu
```

The wrapper expects the library beside `perm_binary_cuda.py` in a source
checkout:

```python
from perm_binary_cuda import binary_permanent_cuda

binary_permanent_cuda([[1, 1], [1, 1]])  # 2
```

`perm_parallel.py` can compile its companion source on first use. Set
`PERM_CUDA_ARCH` to the desired nvcc architecture when `native` or its default
multi-architecture selection is unsuitable.

## Device tests

On October 30, 2026, a Colab A100-SXM4-40GB running CUDA 13.0.88 completed
a selected correctness gate: **47 passed, zero failures or skips**, with four
expensive regressions deliberately deselected. The run compiled fresh libraries
and covered all five device-test families below. The run used source snapshot `a8357e814bf753ef535a56e348d29868079de2a4`;
its original logs are retained by the maintainer. The deliberately excluded
regressions were the order-40 binary grid and order-33, 41, and 47 Schur cases.
This check does not establish their running time or validate H100 hardware.

The normal test suite does not require a CUDA toolkit or device. Device tests
compile fresh libraries in pytest temporary directories and are opt-in:

```bash
PERM_RUN_CUDA_TESTS=1 \
PERM_CUDA_TEST_ARCH=native \
PERM_CUDA_ARCH=native \
    python -m pytest -q \
    tests/test_cuda_device_binary.py \
    tests/test_cuda_device_modular.py \
    tests/test_cuda_device_parallel.py \
    tests/test_cuda_device_schur_glynn.py \
    tests/test_cuda_device_utilities.py
```

The ordinary Schur-device module keeps expensive frontier checks behind
additional gates:

```bash
# Includes the order-41 long regression.
PERM_RUN_CUDA_TESTS=1 \
PERM_RUN_CUDA_LONG_TESTS=1 \
    python -m pytest -q tests/test_cuda_device_schur_glynn.py

# Includes the much longer order-47 frontier regression.
PERM_RUN_CUDA_TESTS=1 \
PERM_RUN_CUDA_FRONTIER_TESTS=1 \
    python -m pytest -q tests/test_cuda_device_schur_glynn.py
```

Use `PERM_CUDA_TEST_ARCH=sm_90` instead of `native` when testing a GH200 build
through the fixtures. `perm_parallel.py` uses the separate
`PERM_CUDA_ARCH` variable for its own on-demand build.

## Large-run discipline

Before an expensive run:

1. build for an explicit architecture and record the compiler version;
2. run the small exact pins and bounded calibration mode provided by the
   source;
3. inspect registers, stack, local memory, and spills;
4. verify the certified modulus product or numerical range for the exact
   instance;
5. retain stdout, stderr, commands, elapsed times, source and binary hashes,
   and an independent certificate; and
6. keep large or frontier test gates out of routine CI.

Validated output for one frozen matrix does not promote a Tier-4 POC into a
general backend. Automatic dispatch additionally requires a stable
matrix-valued ABI, arbitrary supported inputs, centered reconstruction where
coefficients may be signed, concurrency tests, and independent holdouts.

## Packaging boundary

`MANIFEST.in` includes the manual wrappers and `*.cu` sources so they survive
an sdist. `setup.py` intentionally omits those wrappers from `py_modules` and
does not define CUDA extensions. Consequently:

- a checkout or unpacked sdist can build the tools manually;
- an installed wheel exposes only the packaged CPU/Numba APIs and any OpenMP
  extensions built into that wheel; and
- installing `perm` never requires nvcc or a CUDA runtime.
