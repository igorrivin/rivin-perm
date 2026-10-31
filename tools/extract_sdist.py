"""Extract our built source archive on every supported Python patch version."""

import argparse
from pathlib import Path, PurePosixPath
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (not (member.isdir() or member.isfile()) or path.is_absolute()
                    or ".." in path.parts or "\\" in member.name or ":" in member.name):
                raise ValueError(f"unexpected source archive member: {member.name}")
        args.destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = args.destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.extractfile(member).read())


if __name__ == "__main__":
    main()
