# Building and validating rivin-perm

The distribution name is `rivin-perm`; public imports remain `perm`,
`perm_binary`, `perm_polynomial`, and the other documented modules. The initial
release targets CPython 3.10–3.14. NumPy 1.23.5 or later in the 1.x/2.x series
is required; Numba is optional. CUDA is a separately built source tool.

## Distribution policy

The portable `py3-none-any` wheel installs on supported operating systems
without a compiler, OpenMP, Numba, or CUDA. Linux x86-64 accelerator wheels are
built separately with cibuildwheel: all four OpenMP extensions must compile,
the repaired wheel must import them, and installed API checks must pass.
macOS and Windows use the portable wheel for the initial release. A local
source build can opt into another compatible OpenMP toolchain.

`PERM_BUILD_OPENMP` controls source builds:

| Value | Behavior |
|---|---|
| `auto` (default) | Attempt optional OpenMP extensions on Linux/FreeBSD; install the portable implementation if compilation fails |
| `0` | Do not compile extensions; produce a portable wheel |
| `1` | Attempt optional GCC-style OpenMP compilation on any platform |
| `required` | Require every extension to compile; fail the build otherwise |

Build isolation uses NumPy 2.x headers and targets the NumPy 1.23 C API. This
allows the same native wheel to work with supported NumPy 1.x and 2.x runtimes;
the release checks must exercise both. See
[NumPy's downstream build guidance](https://numpy.org/doc/stable/dev/depending_on_numpy.html#numpy-2-abi-handling).

`MANIFEST.in` starts by removing default/VCS/cached file lists, then includes
only reviewed public files. It excludes research-only CUDA sources as well as
private evaluation trees. The archive step also filters filesystem metadata
that macOS can synthesize after manifest processing. Add new public source tools and tests explicitly.
Build from a clean directory; never upload artifacts left by an earlier build.

## Portable release candidate

Use a fresh environment with `build`, `twine`, NumPy, and pytest installed:

```bash
PERM_BUILD_OPENMP=0 python -m build --outdir /tmp/rivin-perm-candidate
python -m twine check /tmp/rivin-perm-candidate/*
python tools/check_distribution.py /tmp/rivin-perm-candidate --require-portable
```

The build frontend makes the wheel from its newly built sdist. The installation
check creates two fresh environments outside the checkout, installs the wheel
in one and rebuilds/installs the sdist in the other, runs `pip check`, verifies
that imports belong to the installed distribution, and checks exact, low-rank,
binary, polynomial, floating-point, and application results. It needs network
access to obtain declared dependencies. Add `--require-numba` to test the extra,
or `--numpy-version 1.23.5` on Python 3.10/3.11 to test the oldest runtime.

For a native candidate, build with `PERM_BUILD_OPENMP=required` and use
`--require-native` during installation validation. A compiler error must fail
this release gate, even though ordinary source installations permit fallback.

`.github/workflows/distributions.yml` defines portable checks on Linux, macOS,
and Windows, an optional-compiler failure check, Numba and old-NumPy checks,
and Linux native wheel builds. Workflow configuration is not evidence that
the remote jobs have run. Candidate wheels must pass that matrix before an
official cross-platform release.

## Publication

These commands prepare and validate artifacts; they do not publish them.
This public repository contains the reviewed export with fresh Git history.
Run the CI matrix and review the repaired wheel artifacts before publication.
The manual `Publish PyPI` workflow accepts an exact successful CI run and
commit, verifies all 17 jobs, downloads its seven immutable distribution
files, checks their metadata and common Python payloads, and publishes those
files without rebuilding. It uses the `pypi` environment and a project-specific
PyPI trusted publisher for `igorrivin/rivin-perm`, workflow `publish.yml`.
