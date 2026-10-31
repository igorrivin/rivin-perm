"""Check the installed public API from outside a checkout, without pytest.

Run with ``python -I /path/to/check_installed.py`` after installing a wheel.
"""

import argparse
import importlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path


MODULES = (
    "perm", "perm_binary", "perm_binary_applications", "perm_float",
    "perm_mod3", "perm_numba", "perm_polynomial",
)
EXTENSIONS = ("perm_c_ext", "perm_binary_ext", "perm_mod3_omp", "perm_polynomial_ext")


def definition(matrix):
    return sum(math.prod(matrix[i][j] for i, j in enumerate(columns))
               for columns in itertools.permutations(range(len(matrix))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-native", action="store_true")
    mode.add_argument("--require-portable", action="store_true")
    parser.add_argument("--require-numba", action="store_true")
    args = parser.parse_args()
    distribution = metadata.distribution("rivin-perm")
    modules = {name: importlib.import_module(name) for name in MODULES}
    for name, module in modules.items():
        installed = Path(distribution.locate_file(name + ".py")).resolve()
        if Path(module.__file__).resolve() != installed:
            raise RuntimeError(f"{name} was imported from a checkout: {module.__file__}")

    import numpy as np

    perm = modules["perm"]
    binary = modules["perm_binary"]
    polynomial = modules["perm_polynomial"]
    examples = ([], [[-7]], [[1, 2], [3, 4]],
                [[2, -3, 0], [4, 1, 5], [-1, 2, 3]])
    for matrix in examples:
        assert perm.matrix_permanent(matrix, parallel=False) == definition(matrix)
    assert perm.matrix_permanent([[1 << 70, 0], [0, -(1 << 70)]]) == -(1 << 140)
    assert perm.permanent(5) == -5
    u, v = list(range(1, 25)), [(-1)**j * (j + 1) for j in range(24)]
    matrix = [[a * b for b in v] for a in u]
    assert perm.matrix_permanent(matrix, method="low_rank", parallel=False) == (
        math.factorial(24) * math.prod(u) * math.prod(v))
    assert binary.binary_permanent([[1] * 7 for _ in range(7)]) == math.factorial(7)
    assert modules["perm_binary_applications"].grid_dimer_count((2, 2)) == 2
    identity, off_diagonal = [[1, 0], [0, 1]], [[0, 1], [1, 0]]
    assert tuple(polynomial.affine_permanent(identity, off_diagonal)) == (1, 0, 1)
    assert tuple(polynomial.principal_permanent_polynomial([[1, 2], [3, 4]])) == (1, 5, 10)
    assert tuple(polynomial.weighted_rook_polynomial([[1, 2], [3, 4]])) == (1, 10, 10)
    array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert modules["perm_mod3"].permanent_mod3_fast(array) == 1
    actual = modules["perm_float"].permanent_ryser_float([[1+1j, 2], [3, 4-1j]])
    assert abs(actual - (11+3j)) < 1e-12

    native = []
    for name in EXTENSIONS:
        try:
            importlib.import_module(name)
        except ImportError:
            if args.require_native:
                raise
        else:
            native.append(name)
    if args.require_portable:
        assert not native, f"portable wheel contains extensions: {native}"
    if args.require_native:
        assert perm.get_backend() == "c"
        assert polynomial.polynomial_backend_capabilities()["native_extension_available"]
        assert importlib.import_module("perm_c_ext").permanent_mod_p([[1, 2], [3, 4]], 101, 1) == 10
        assert importlib.import_module("perm_binary_ext").binary_permanent_mod_p([[1, 1], [1, 1]], 101, 1) == 2
        assert importlib.import_module("perm_mod3_omp").permanent_mod3(array, 1) == 1
        assert tuple(polynomial.affine_permanent(identity, off_diagonal, backend="native", parallel=False)) == (1, 0, 1)
    if args.require_numba:
        numba = modules["perm_numba"]
        assert numba.HAS_NUMBA
        assert numba.ryser_permanent_mod_p_numba(array, 101) == 10
    print(json.dumps({"distribution": distribution.metadata["Name"],
                      "version": distribution.version, "numpy": np.__version__,
                      "backend": perm.get_backend(), "native": native, "status": "PASS"}))


if __name__ == "__main__":
    main()
