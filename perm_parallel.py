"""
Python wrapper for perm_parallel_cuda – parallel Ryser permanent for a
single large matrix (n up to 56+).

Auto-compiles the CUDA shared library on first use.

Usage:
    from perm_parallel import permanent_gpu
    result = permanent_gpu(matrix)              # real or complex
    result = permanent_gpu(matrix, real=True)    # force real path
"""

import ctypes
import os
import subprocess
import sys
import tempfile
import numpy as np
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CU_SRC = _DIR / "perm_parallel_cuda.cu"
_SO_PATH = _DIR / "perm_parallel_cuda.so"

_lib = None


def _compile():
    """Compile the CUDA shared library if needed."""
    if not _CU_SRC.exists():
        raise RuntimeError(
            f"CUDA source {_CU_SRC} is not installed; build the optional CUDA "
            "backend from a source checkout"
        )
    if _SO_PATH.exists() and _SO_PATH.stat().st_mtime >= _CU_SRC.stat().st_mtime:
        return  # up to date
    print(f"Compiling {_CU_SRC.name} → {_SO_PATH.name} ...", file=sys.stderr)
    cuda_arch = os.environ.get("PERM_CUDA_ARCH", "all-major")
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{_SO_PATH.stem}-", suffix=".so", dir=_DIR
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-O3",
        f"-arch={cuda_arch}",
        "-o", str(tmp_path), str(_CU_SRC),
    ]
    try:
        subprocess.check_call(cmd)
        os.replace(tmp_path, _SO_PATH)
    finally:
        tmp_path.unlink(missing_ok=True)
    print("Done.", file=sys.stderr)


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    _compile()
    _lib = ctypes.CDLL(str(_SO_PATH))

    # int permanent_parallel_real(const double *matrix_re, int n, double *result_out)
    _lib.permanent_parallel_real.restype = ctypes.c_int
    _lib.permanent_parallel_real.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]

    # int permanent_parallel_complex(const double *re, const double *im,
    #                                int n, double *res_re, double *res_im)
    _lib.permanent_parallel_complex.restype = ctypes.c_int
    _lib.permanent_parallel_complex.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    # int get_gpu_info(int *num_sms, int *max_threads_per_sm)
    _lib.get_gpu_info.restype = ctypes.c_int
    _lib.get_gpu_info.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    return _lib


def gpu_info():
    """Return (num_sms, max_threads_per_sm)."""
    lib = _get_lib()
    sms = ctypes.c_int()
    thr = ctypes.c_int()
    rc = lib.get_gpu_info(ctypes.byref(sms), ctypes.byref(thr))
    if rc != 0:
        raise RuntimeError(f"CUDA device query failed (rc={rc})")
    return sms.value, thr.value


def permanent_gpu(matrix, real=None):
    """
    Compute the permanent of a single n×n matrix on GPU.

    Parameters
    ----------
    matrix : array_like
        n×n real or complex matrix.
    real : bool or None
        If True, use real kernel. If False, use complex kernel.
        If None (default), auto-detect from dtype.

    Returns
    -------
    float or complex
        The permanent.
    """
    matrix = np.asarray(matrix, order='C')
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {matrix.shape}")
    n = matrix.shape[0]

    is_complex = np.issubdtype(matrix.dtype, np.complexfloating)
    if real is None:
        real = not is_complex
    if real and is_complex:
        raise ValueError("Cannot use real kernel on complex matrix")

    if n == 0:
        return 1.0 if real else 1.0 + 0.0j

    lib = _get_lib()

    if real:
        mat = np.ascontiguousarray(matrix, dtype=np.float64)
        result = ctypes.c_double()
        rc = lib.permanent_parallel_real(
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
            ctypes.byref(result),
        )
        if rc != 0:
            raise RuntimeError(f"CUDA kernel failed (rc={rc})")
        return result.value
    else:
        if not is_complex:
            matrix = matrix.astype(np.complex128)
        mat_re = np.ascontiguousarray(matrix.real, dtype=np.float64)
        mat_im = np.ascontiguousarray(matrix.imag, dtype=np.float64)
        res_re = ctypes.c_double()
        res_im = ctypes.c_double()
        rc = lib.permanent_parallel_complex(
            mat_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mat_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
            ctypes.byref(res_re),
            ctypes.byref(res_im),
        )
        if rc != 0:
            raise RuntimeError(f"CUDA kernel failed (rc={rc})")
        return complex(res_re.value, res_im.value)
