"""Copy matching files from a directory tree into one flat directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def flatten_files(input_dir: Path, output_dir: Path, extension: str = ".tif") -> int:
    """Copy files recursively and add numeric suffixes for name collisions."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    extension = extension if extension.startswith(".") else f".{extension}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resolved = output_dir.resolve()
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == extension.lower()
        and path.parent.resolve() != output_resolved
    ]

    copied = 0
    for source in sorted(files):
        destination = output_dir / source.name
        counter = 1
        while destination.exists():
            destination = output_dir / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        shutil.copy2(source, destination)
        copied += 1

    print(f"Copied {copied} {extension} file(s) to {output_dir}")
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--extension", default=".tif", help="File extension to copy (default: .tif)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flatten_files(args.input_dir, args.output_dir, args.extension)
