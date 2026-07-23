#!/usr/bin/env python3
"""Validate the lightweight public archive without scientific recomputation."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_BLOB = 100 * 1024 * 1024
FORBIDDEN = tuple(
    token.encode()
    for token in (
        "/run/media/" + "y" + "dx",
        "/data/" + "y" + "dx",
        "/Users/" + "pat" + "rick",
        "y" + "dx@",
        "10.20." + "52.75",
        "ZHANG" + "LAB",
        "yang" + "dengxiang",
    )
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def repository_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        )
    )


def check_json(files: list[Path]) -> None:
    for path in files:
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as handle:
                json.load(handle)


def check_gzip(files: list[Path]) -> None:
    for path in files:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass


def check_size_and_paths(files: list[Path]) -> None:
    failures: list[str] = []
    for path in files:
        if path.stat().st_size >= MAX_GITHUB_BLOB:
            failures.append(f"GitHub-size violation: {path.relative_to(ROOT)}")
        data = path.read_bytes()
        for token in FORBIDDEN:
            if token in data:
                failures.append(
                    f"machine-specific token {token!r}: {path.relative_to(ROOT)}"
                )
    if failures:
        raise RuntimeError("\n".join(failures))


def check_markdown_links(files: list[Path]) -> None:
    failures: list[str] = []
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.strip()
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
                or "${" in target
            ):
                continue
            target = target.split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"broken link in {path.relative_to(ROOT)}: {target}"
                )
    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> int:
    files = repository_files()
    check_json(files)
    check_gzip(files)
    check_size_and_paths(files)
    check_markdown_links(files)
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_checked": len(files),
                "largest_bytes": max(path.stat().st_size for path in files),
                "json_files": sum(path.suffix == ".json" for path in files),
                "gzip_files": sum(path.suffix == ".gz" for path in files),
                "markdown_files": sum(path.suffix.lower() == ".md" for path in files),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
