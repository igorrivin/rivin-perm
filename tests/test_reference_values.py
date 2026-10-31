"""Regression tests for the published Schur-matrix permanent values."""

from pathlib import Path
import re

import pytest

from perm import permanent


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# OEIS A003112 b-file: https://oeis.org/A003112/b003112.txt
# The OEIS index k represents the odd matrix size n = 2*k + 1.
OEIS_A003112 = {
    1: 1,
    3: -3,
    5: -5,
    7: -105,
    9: 81,
    11: 6765,
    13: 175747,
    15: 30375,
    17: 25219857,
    19: 142901109,
    21: 4548104883,
    23: -31152650265,
    25: -5198937484375,
    27: 65230244418933,
    29: -1300425712598285,
    31: 126691467546591,
    33: 868088125376401545,
    35: -15139017417029296875,
}

PROJECT_EXTENSIONS = {
    37: -339696021338069660245,
    39: -29260070859488572505025,
    41: 1617562409111471411255401,
    43: -23811413912733485902431027,
    45: -470483360490107435783203125,
    47: -52876272639843868105982022705,
}


def reference_file_values():
    text = (REPOSITORY_ROOT / "schur_permanents.txt").read_text(encoding="utf-8")
    return {
        int(size): int(value)
        for size, value in re.findall(r"^n=(\d+):\s+(-?\d+)\s*$", text, re.MULTILINE)
    }


def test_reference_file_matches_oeis():
    values = reference_file_values()
    assert {size: values[size] for size in OEIS_A003112} == OEIS_A003112


def test_reference_file_preserves_project_extensions():
    values = reference_file_values()
    assert {size: values[size] for size in PROJECT_EXTENSIONS} == PROJECT_EXTENSIONS


@pytest.mark.parametrize("size, expected", list(OEIS_A003112.items())[:6])
def test_small_schur_permanents(size, expected):
    assert permanent(size, parallel=False) == expected
