"""Collect upload inputs from a successful, explicitly selected public CI run."""

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


REPOSITORY = "igorrivin/rivin-perm"
MODULES = tuple(name + ".py" for name in (
    "perm", "perm_binary", "perm_binary_applications", "perm_float",
    "perm_mod3", "perm_numba", "perm_polynomial"))


def gh(*arguments):
    return subprocess.check_output(["gh", *arguments, "--repo", REPOSITORY], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=Path("release-dist"))
    args = parser.parse_args()
    if not args.run_id.isdecimal() or not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("provide a numeric CI run id and a complete commit SHA")
    if os.environ.get("GITHUB_REF", "refs/heads/main") != "refs/heads/main":
        raise RuntimeError("publish only from main")
    if os.environ.get("GITHUB_REPOSITORY", REPOSITORY) != REPOSITORY:
        raise RuntimeError("unexpected repository")
    run = json.loads(gh("run", "view", args.run_id, "--json",
                        "status,conclusion,event,headBranch,headSha,workflowName,jobs,url"))
    if not (run["status"] == "completed" and run["conclusion"] == "success"
            and run["workflowName"] == "Distributions" and run["event"] == "push"
            and run["headBranch"] == "main" and run["headSha"] == args.commit
            and len(run["jobs"]) == 17
            and all(job["conclusion"] == "success" for job in run["jobs"])):
        raise RuntimeError("CI run is not a fully passing main-branch distribution run for this commit")

    args.output.mkdir(exist_ok=False)
    manifest = {"repository": REPOSITORY, "commit": args.commit,
                "run_id": args.run_id, "run_url": run["url"], "artifacts": []}
    with tempfile.TemporaryDirectory(prefix="rivin-release-") as temporary:
        root = Path(temporary)
        for name in ("portable-dist", "linux-native-wheels"):
            gh("run", "download", args.run_id, "--name", name, "--dir", str(root / name))
        wheels = [p for p in root.rglob("*.whl") if not p.name.startswith("._")]
        sources = [p for p in root.rglob("*.tar.gz") if not p.name.startswith("._")]
        if len(wheels) != 6 or len(sources) != 1:
            raise RuntimeError("expected one portable wheel, five Linux wheels, and one sdist")
        versions, payloads = set(), []
        for path in wheels:
            with zipfile.ZipFile(path) as wheel:
                names = wheel.namelist()
                metadata = Parser().parsestr(wheel.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode())
                if metadata["Name"] != "rivin-perm":
                    raise RuntimeError("unexpected package")
                versions.add(metadata["Version"])
                payloads.append({name:hashlib.sha256(wheel.read(name)).hexdigest() for name in MODULES})
                if any(any(part in ("evals", "experiments", "verification", "outputs", "papers", ".git", "__pycache__")
                           or part.startswith("._") for part in Path(name).parts) for name in names):
                    raise RuntimeError("unexpected wheel content")
        if len(versions) != 1 or any(payload != payloads[0] for payload in payloads[1:]):
            raise RuntimeError("wheel versions or Python modules differ")
        version = next(iter(versions))
        with tarfile.open(sources[0]) as archive:
            members = archive.getmembers()
            if any(not (m.isfile() or m.isdir()) or Path(m.name).is_absolute()
                   or ".." in Path(m.name).parts for m in members):
                raise RuntimeError("unexpected sdist member")
            prefix = f"rivin_perm-{version}/"
            metadata = Parser().parsestr(archive.extractfile(prefix + "PKG-INFO").read().decode())
            if metadata["Name"] != "rivin-perm" or metadata["Version"] != version:
                raise RuntimeError("sdist metadata differs")
            for name in MODULES:
                if hashlib.sha256(archive.extractfile(prefix + name).read()).hexdigest() != payloads[0][name]:
                    raise RuntimeError("sdist and wheel Python modules differ")
        for path in sorted(wheels + sources):
            if not path.name.startswith(f"rivin_perm-{version}-") and path.name != f"rivin_perm-{version}.tar.gz":
                raise RuntimeError("unexpected artifact filename")
            target = args.output / path.name
            shutil.copyfile(path, target)
            manifest["artifacts"].append({"file":path.name, "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    subprocess.run([sys.executable, "-m", "twine", "check", *map(str, args.output.iterdir())], check=True)
    Path("release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"version":version, "artifacts":len(manifest["artifacts"]), "status":"READY"}))


if __name__ == "__main__":
    main()
