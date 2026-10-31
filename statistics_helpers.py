"""Small, research-only helpers for permanent distribution studies.

KS distances here are descriptive discrepancies on held-out observations, not
composite-null goodness-of-fit p-values or evidence for a limiting law.
"""

import json
import os
from math import comb
from operator import index
from pathlib import Path
import tempfile

import numpy as np


def haar_permanent_mean_square(n):
    """E|per(U)|² for a full Haar U(n) matrix (Nezami, Eq. 97).

    The empty permanent is one. This is not the dilute Haar-submatrix or
    variance-1/n independent-Ginibre normalization.
    """
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("matrix order must be a nonnegative integer")
    try:
        n = index(n)
    except TypeError as exc:
        raise ValueError("matrix order must be a nonnegative integer") from exc
    if n < 0:
        raise ValueError("matrix order must be a nonnegative integer")
    return 1.0 if n == 0 else 1 / comb(2 * n - 1, n)


def split_fit_evaluation(values, *, seed=42, max_evaluation=500):
    """Return disjoint training and shared evaluation samples, deterministically.

    Hold out one quarter of the observations, capped to bound expensive stable
    CDF calls. All compared models must fit the returned training array and
    evaluate on the same returned evaluation array.
    """
    values = np.asarray(values)
    if (values.ndim != 1 or np.iscomplexobj(values) or len(values) < 8
            or not np.all(np.isfinite(values))):
        raise ValueError("need at least eight finite real observations")
    if isinstance(max_evaluation, (bool, np.bool_)):
        raise ValueError("max_evaluation must be a positive integer")
    try:
        limit = index(max_evaluation)
    except TypeError as exc:
        raise ValueError("max_evaluation must be a positive integer") from exc
    if limit < 1:
        raise ValueError("max_evaluation must be a positive integer")
    order = np.random.default_rng(seed).permutation(len(values))
    count = min(limit, len(values) // 4)
    return values[order[count:]].copy(), values[order[:count]].copy()


def ks_distance(values, cdf):
    """Two-sided empirical/CDF distance; deliberately returns no p-value."""
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError("KS distance requires real observations")
    values = np.sort(values.astype(float))
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("need a nonempty finite one-dimensional sample")
    probabilities = np.asarray(cdf(values), dtype=float)
    if (probabilities.shape != values.shape
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("CDF returned invalid probabilities")
    count = len(values)
    upper = np.arange(1, count + 1) / count
    lower = np.arange(count) / count
    return float(max(np.max(upper - probabilities), np.max(probabilities - lower)))


def sample_key(ensemble, n):
    return f"{ensemble}__n{n}"


def save_permanent_samples(path, samples, metadata):
    """Retain raw permanents and JSON provenance in a pickle-free NPZ file."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("sample archive must have an .npz suffix")
    if not isinstance(metadata, dict) or not samples:
        raise ValueError("sample archive needs metadata and at least one sample")
    provenance = np.asarray(json.dumps(metadata, allow_nan=False))
    validated = {}
    for key, values in samples.items():
        values = np.asarray(values)
        if (not isinstance(key, str) or key == "__metadata__"
                or values.ndim != 1 or len(values) == 0
                or values.dtype.kind not in "fciu"
                or not np.all(np.isfinite(values))):
            raise ValueError("sample arrays must contain finite numeric permanents")
        validated[key] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash during a checkpoint must not destroy the previous archive.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **validated, __metadata__=provenance)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_permanent_samples(path):
    """Read archives produced by save_permanent_samples without loading pickle."""
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError("sample archive is missing provenance metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        samples = {}
        for key in archive.files:
            if key == "__metadata__":
                continue
            values = archive[key]
            if (values.ndim != 1 or len(values) == 0 or values.dtype.kind not in "fciu"
                    or not np.all(np.isfinite(values))):
                raise ValueError(f"invalid permanent sample array: {key}")
            samples[key] = values.copy()
    if not isinstance(metadata, dict) or not samples:
        raise ValueError("sample archive needs metadata and at least one sample")
    return samples, metadata
