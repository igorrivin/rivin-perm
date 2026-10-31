"""Source-package boundaries and test collection, without native/GPU builds."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TREES = (
    "notes/evals", "notes/experiments", "notes/verification", "notes/client",
    "notes/session-logs", "outputs", "papers", ".agents", ".codex",
)
CHECKOUT_ONLY_TESTS = (
    "test_entry_derivative_cofactor_projection_oracle.py",
    "test_f9_count_verification.py",
    "test_finite_field_rank_verification.py",
    "test_krawtchouk_crt_cuda_oracle.py",
    "test_krawtchouk_power2_prototype.py",
    "test_polynomial_interpolation_campaign.py",
    "test_polynomial_record_verification.py",
    "test_principal_sign_projection_prototype.py",
    "test_publication_queue.py",
    "test_repository_structure.py",
    "test_scalar_holdout_campaign.py",
    "test_schur_record_verification.py",
    "test_schur_release.py",
    "test_schur_release_v2.py",
    "test_schur47_public_audit.py",
    "test_schur_ryser_audit.py",
    "test_schur_ryser_crt.py",
    "test_schur_crt_queue.py",
)
RESEARCH_CUDA = ("perm_complex_cuda.cu", "perm_proj_fq_cuda.cu", "sample_perm_n4_cuda.cu")


def test_manifest_removes_private_paths_from_cached_file_lists():
    pytest.importorskip("setuptools")
    from setuptools._distutils.filelist import FileList

    candidates = [f"{tree}/private-probe.txt" for tree in EXCLUDED_TREES]
    candidates += [f"tests/{name}" for name in CHECKOUT_ONLY_TESTS]
    candidates += list(RESEARCH_CUDA)
    candidates += ["private-future-script.py", "tests/test_future_private_oracle.py", ".git/config"]
    candidates += ["rivin_perm.egg-info/._PKG-INFO", "rivin_perm.egg-info/._SOURCES.txt"]
    candidates += ["perm_modp_cuda.so", "nested/old-library.dylib", "notes/benchmarks/schur_compare.py"]
    files = FileList()
    files.set_allfiles(candidates)
    files.files = candidates.copy()  # Simulate a stale egg-info/SOURCES.txt.
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        if line and not line.startswith("#"):
            files.process_template_line(line)
    assert set(files.files) == {"notes/benchmarks/schur_compare.py"}


@pytest.fixture(scope="module")
def source_distribution(tmp_path_factory):
    pytest.importorskip("setuptools")
    pytest.importorskip("numpy")
    temporary = tmp_path_factory.mktemp("sdist-boundary")
    stage = temporary / "stage"
    stage.mkdir()

    # Copy only public package inputs and tests. Checkout-only test bodies are
    # replaced by synthetic probes; private evaluator files are never copied.
    names = {"MANIFEST.in", "setup.py", "pyproject.toml", "LICENSE", "README.md"}
    names.update(name + ".py" for name in (
        "perm", "perm_binary", "perm_binary_applications", "perm_float", "perm_mod3",
        "perm_numba", "perm_polynomial"))
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "include":
            for name in fields[1:]:
                assert not any(character in name for character in "*?[]"), name
                names.add(name)
    for name in sorted(names):
        source, target = ROOT / name, stage / name
        assert source.is_file(), f"explicit package input missing: {name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in (ROOT / "tests").glob("test_*.py"):
        target = stage / "tests" / source.name
        if source.name in CHECKOUT_ONLY_TESTS:
            target.write_text("PRIVATE_SOURCE_TEST_PROBE\n")
        else:
            shutil.copyfile(source, target)
    for tree in EXCLUDED_TREES:
        target = stage / tree / "private-probe.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("PRIVATE_SOURCE_DATA_PROBE\n")

    environment = dict(os.environ, PERM_BUILD_OPENMP="0", PERM_RUN_CUDA_TESTS="0",
                       PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    built = subprocess.run(
        [sys.executable, "-B", "-c",
         "from pathlib import Path\n"
         "from setuptools.command.sdist import sdist\n"
         "original = sdist.make_release_tree\n"
         "def make_tree(self, base_dir, files):\n"
         "    original(self, base_dir, files)\n"
         "    Path(base_dir, '._generated-sidecar').write_text('PRIVATE_METADATA_PROBE')\n"
         "sdist.make_release_tree = make_tree\n"
         "from setuptools.build_meta import build_sdist\n"
         "import sys\n"
         "build_sdist(sys.argv[1])",
         str(temporary / "dist")],
        cwd=stage, env=environment, text=True, capture_output=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    archives = list((temporary / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    extracted = temporary / "extracted"
    extracted.mkdir()
    with tarfile.open(archives[0]) as archive:
        members = archive.getmembers()
        prefix = members[0].name.split("/")[0] + "/"
        paths = {member.name[len(prefix):] for member in members if member.isfile()}
        assert all(member.isdir() or member.isfile() for member in members)
        assert all(".." not in Path(member.name).parts and not member.name.startswith("/") for member in members)
        # This is our freshly built local archive, already checked above.
        for member in members:
            if member.isfile():
                target = extracted / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    return extracted / prefix.rstrip("/"), paths, environment


def test_built_sdist_contains_helpers_and_excludes_private_artifacts(source_distribution):
    directory, paths, _ = source_distribution
    for path in (
        "statistics_helpers.py", "cycle_geodesic.py", "perm_modp_cuda.cu",
        "notes/benchmarks/schur_compare.py", "notes/benchmarks/schur_compare_wrapper.cu",
        "notes/benchmarks/check_schur_orbits.py", "tests/test_schur_benchmark.py",
        "notes/benchmarks/SCHUR.md",
        "tests/test_source_distribution.py",
        "benchmark_low_rank.py", "tests/test_low_rank_dispatch.py",
        "showcase_low_rank.py", "LOW_RANK_SHOWCASES.md", "tests/test_low_rank_showcases.py",
    ):
        assert path in paths
    assert not any(path.startswith(tree + "/") for path in paths for tree in EXCLUDED_TREES)
    assert not any("tests/" + name in paths for name in CHECKOUT_ONLY_TESTS)
    assert not any(name in paths for name in RESEARCH_CUDA)
    assert not any(part.startswith("._") or part == ".DS_Store"
                   for path in paths for part in Path(path).parts)
    assert not any(Path(path).suffix in {".so", ".pyd", ".dll", ".dylib", ".o", ".a", ".pyc"} for path in paths)
    assert not any(b"PRIVATE_SOURCE_DATA_PROBE" in (directory / path).read_bytes()
                   for path in paths if path.endswith(".txt"))


def test_built_sdist_metadata_identifies_the_public_distribution(source_distribution):
    from email.parser import Parser

    directory, paths, _ = source_distribution
    metadata = Parser().parsestr((directory / "PKG-INFO").read_text())
    assert metadata["Name"] == "rivin-perm"
    assert metadata["Requires-Python"] == ">=3.10"
    assert metadata["License-Expression"] == "MIT"
    assert "numpy<3,>=1.23.5" in metadata.get_all("Requires-Dist")
    assert "tools/check_installed.py" in paths
    assert "tools/check_distribution.py" in paths


def test_built_sdist_test_suite_collects_without_private_checkout(source_distribution):
    directory, _, environment = source_distribution
    collected = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q"],
        cwd=directory, env=environment, capture_output=True, text=True, timeout=60)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    if sys.version_info >= (3, 10):
        assert "test_schur_benchmark.py::" in collected.stdout
    assert "test_research_statistics.py::test_full_haar_mean" in collected.stdout


def test_built_sdist_public_helpers_run_without_private_checkout(source_distribution):
    directory, _, environment = source_distribution
    checked = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_schur_benchmark.py",
         "tests/test_research_statistics.py", "tests/test_cycle_geodesic.py",
         "tests/test_low_rank_dispatch.py", "tests/test_low_rank_showcases.py"],
        cwd=directory, env=environment, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stdout + checked.stderr
