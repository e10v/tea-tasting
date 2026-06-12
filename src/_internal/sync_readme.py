"""Sync docs/index.md with README.md."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


README_PATH = Path("README.md")
INDEX_PATH = Path("docs/index.md")


def main() -> int:
    args = parse_args()

    if args.check:
        if is_synced(README_PATH, INDEX_PATH):
            sys.stdout.write("README.md and docs/index.md are in sync.\n")
            return 0
        sys.stdout.write(
            "README.md and docs/index.md are out of sync. "
            "Run `uv run python src/_internal/sync_readme.py`.\n",
        )
        return 1

    if sync_readme(README_PATH, INDEX_PATH):
        sys.stdout.write("Updated docs/index.md from README.md.\n")
    else:
        sys.stdout.write("docs/index.md is already up to date.\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy README.md to docs/index.md.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if README.md and docs/index.md are different.",
    )
    return parser.parse_args()


def is_synced(readme_path: Path, index_path: Path) -> bool:
    if not index_path.exists():
        return False
    return readme_path.read_text(encoding="utf-8") == index_path.read_text(
        encoding="utf-8",
    )


def sync_readme(readme_path: Path, index_path: Path) -> bool:
    readme_text = readme_path.read_text(encoding="utf-8")
    if index_path.exists() and readme_text == index_path.read_text(encoding="utf-8"):
        return False

    index_path.write_text(readme_text, encoding="utf-8")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
