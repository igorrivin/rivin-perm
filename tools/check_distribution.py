"""Install a built wheel and rebuild its sdist in fresh, separate pip environments.

This release check downloads declared build/runtime dependencies. It never
installs into the caller's environment and never publishes anything.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


def run(command, cwd, environment):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True, timeout=600)


def check(artifact, directory, environment, options, numpy_version):
    venv.EnvBuilder(with_pip=True).create(directory)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    target = str(artifact) + ("[numba]" if options.require_numba else "")
    requirement = [f"numpy=={numpy_version}"] if numpy_version else []
    run([python, "-m", "pip", "install", "--no-cache-dir", target, *requirement], directory, environment)
    run([python, "-m", "pip", "check"], directory, environment)
    flags = [name for name in ("--require-native", "--require-portable", "--require-numba")
             if getattr(options, name[2:].replace("-", "_"))]
    run([python, "-I", Path(__file__).with_name("check_installed.py"), *flags], directory, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--require-native", action="store_true")
    modes.add_argument("--require-portable", action="store_true")
    parser.add_argument("--require-numba", action="store_true")
    parser.add_argument("--numpy-version", help="Test an older runtime against the same built wheel")
    args = parser.parse_args()
    wheels = list(args.dist.resolve().glob("*.whl"))
    sources = list(args.dist.resolve().glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error("directory must contain exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert not any(part.startswith("._") or part == ".DS_Store"
                       for name in names for part in Path(name).parts), "macOS metadata in wheel"
        if args.require_portable:
            assert wheels[0].name.endswith("-py3-none-any.whl"), "incorrect portable wheel tag"
            assert not any(name.endswith((".so", ".pyd", ".dylib", ".dll")) for name in names)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    if args.require_portable:
        environment["PERM_BUILD_OPENMP"] = "0"
    elif args.require_native:
        environment["PERM_BUILD_OPENMP"] = "required"
    with tempfile.TemporaryDirectory(prefix="rivin-perm-install-") as temporary:
        root = Path(temporary)
        check(wheels[0], root / "wheel", environment, args, args.numpy_version)
        check(sources[0], root / "sdist", environment, args, args.numpy_version)
    print("PASS: wheel and sdist installations checked outside the checkout")


if __name__ == "__main__":
    main()
