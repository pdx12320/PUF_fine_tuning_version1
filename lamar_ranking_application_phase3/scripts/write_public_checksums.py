#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "provenance/PUBLIC_CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing generated checksum manifest.",
    )
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        raise FileExistsError(OUTPUT)
    paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT
    ]
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT)}\n"
        for path in sorted(paths)
    ]
    OUTPUT.write_text("".join(lines))
    print(f"WROTE {OUTPUT} ({len(lines)} files)")


if __name__ == "__main__":
    main()
