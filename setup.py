"""Build configuration for the ``perm`` Python modules.

The OpenMP extensions are optional accelerators.  Linux source builds try to
compile them by default; other platforms install the portable Python/Numba
implementations instead.  Set ``PERM_BUILD_OPENMP=0`` to request a pure-Python
build, ``PERM_BUILD_OPENMP=1`` to try the GCC/Clang OpenMP flags explicitly,
or ``PERM_BUILD_OPENMP=required`` to fail if any accelerator cannot be built.
"""

from pathlib import Path
import os
import sys
import tarfile
import warnings

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist


class ReviewedSdist(sdist):
    """Exclude macOS sidecars created after MANIFEST.in has been processed."""

    def make_archive(self, base_name, format, root_dir=None, base_dir=None,
                     owner=None, group=None):
        if format != "gztar":
            return super().make_archive(base_name, format, root_dir, base_dir, owner, group)
        destination = Path(base_name + ".tar.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)

        def public_metadata(member):
            if any(part.startswith("._") or part == ".DS_Store"
                   for part in Path(member.name).parts):
                return None
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            return member

        with tarfile.open(destination, "w:gz") as archive:
            archive.add(Path(root_dir or ".") / base_dir,
                        arcname=base_dir, filter=public_metadata)
        return str(destination)


class OptionalExtensionWheel(bdist_wheel):
    """Tag a wheel as pure when every optional extension failed to build."""

    def run(self):
        # wheel decides its compatibility tag before setuptools normally tries
        # optional extensions. Build first so the tag reflects what was
        # actually produced, then let the standard command package that build.
        if not self.skip_build:
            self.run_command("build")

        build_ext = self.get_finalized_command("build_ext")
        built_extensions = [
            output for output in build_ext.get_outputs() if Path(output).is_file()
        ]
        self.root_is_pure = not built_extensions
        self.skip_build = True
        super().run()


def openmp_extensions():
    """Return optional OpenMP extensions for supported compiler platforms."""
    setting = os.environ.get("PERM_BUILD_OPENMP", "auto").strip().lower()
    if setting not in {"auto", "0", "1", "required"}:
        raise RuntimeError("PERM_BUILD_OPENMP must be one of: auto, 0, 1, required")
    if setting == "0":
        return []

    gcc_style_platform = sys.platform.startswith(("linux", "freebsd"))
    if setting == "auto" and not gcc_style_platform:
        warnings.warn(
            "OpenMP extensions are not built automatically on this platform; "
            "the portable Python/Numba backends will be installed.",
            stacklevel=2,
        )
        return []

    # Deliberately omit -march=native: published wheels must run on CPUs other
    # than the build host.  The extensions are optional, so a compiler without
    # OpenMP support does not make the package unusable.
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]
    optional = setting != "required"
    import numpy as np

    return [
        Extension(
            "perm_c_ext",
            sources=["perm_c_omp.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=optional,
        ),
        Extension(
            "perm_mod3_omp",
            sources=["perm_mod3_omp.c"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_TARGET_VERSION", "NPY_1_23_API_VERSION")],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=optional,
        ),
        Extension(
            "perm_binary_ext",
            sources=["perm_binary_omp.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=optional,
        ),
        Extension(
            "perm_polynomial_ext",
            sources=["perm_polynomial_omp.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=optional,
        ),
    ]


setup(
    ext_modules=openmp_extensions(),
    cmdclass={"bdist_wheel": OptionalExtensionWheel, "sdist": ReviewedSdist},
    zip_safe=False,
)
