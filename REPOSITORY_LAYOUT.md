# Repository layout and support levels

This is the public source repository for `rivin-perm`. It has fresh history
and contains the reviewed library, tests, source tools, and documentation.

| Component | Source repository / sdist | Installed wheel |
|---|---|---|
| `perm`, `perm_binary`, applications, polynomial, floating-point, mod-3 and Numba modules | Included | Included |
| Four C/OpenMP extension sources | Included; optional source build | Included only in a native wheel |
| Numba backend | Optional extra | Optional extra |
| Six supported CUDA sources and three wrappers | Manual build and use | Excluded |
| Benchmarks and source-tool tests listed in `MANIFEST.in` | Included | Excluded |

The installed API is the stable library interface. Optional accelerators must
agree with the portable implementation; their absence must not break it.
Source tools have individual build commands and input contracts and are not
automatically selected by installed CPU APIs.

`MANIFEST.in` is a positive allowlist. Add new public files and tests explicitly.
Generated binaries, local results, and unreviewed research material are not
distribution inputs. See [PACKAGING.md](PACKAGING.md), [CUDA.md](CUDA.md), and
[CONTRIBUTING.md](CONTRIBUTING.md) for release and validation procedures.
