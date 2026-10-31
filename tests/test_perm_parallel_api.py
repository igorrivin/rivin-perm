import numpy as np
import pytest

import perm_parallel


@pytest.mark.parametrize(
    "matrix",
    [
        np.arange(6).reshape(2, 3),
        np.arange(6).reshape(3, 2),
        np.arange(3),
        np.array(1.0),
    ],
)
def test_permanent_gpu_rejects_non_square_inputs_without_loading_cuda(
    monkeypatch, matrix
):
    monkeypatch.setattr(
        perm_parallel,
        "_get_lib",
        lambda: pytest.fail("CUDA should not be loaded for invalid input"),
    )

    with pytest.raises(ValueError, match="square 2-D matrix"):
        perm_parallel.permanent_gpu(matrix)


def test_permanent_gpu_handles_empty_matrix_without_loading_cuda(monkeypatch):
    monkeypatch.setattr(
        perm_parallel,
        "_get_lib",
        lambda: pytest.fail("CUDA should not be loaded for the empty permanent"),
    )

    assert perm_parallel.permanent_gpu(np.empty((0, 0))) == 1.0
    assert perm_parallel.permanent_gpu(np.empty((0, 0)), real=False) == 1.0 + 0.0j


def test_permanent_gpu_rejects_complex_input_for_real_kernel(monkeypatch):
    monkeypatch.setattr(
        perm_parallel,
        "_get_lib",
        lambda: pytest.fail("CUDA should not be loaded for invalid input"),
    )

    with pytest.raises(ValueError, match="real kernel"):
        perm_parallel.permanent_gpu(np.eye(2, dtype=np.complex128), real=True)
