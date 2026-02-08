"""Sync docs/index.md with README.md."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


README_PATH = Path("README.md")
INDEX_PATH = Path("docs/index.md")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Copy README.md to docs/index.md.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if README.md and docs/index.md are different.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    """Read a text file as UTF-8.

    Args:
        path: File path.

    Returns:
        File content.
    """
    return path.read_text(encoding="utf-8")


def is_synced(readme_path: Path, index_path: Path) -> bool:
    """Return whether the files have the same content.

    Args:
        readme_path: Path to README.md.
        index_path: Path to docs/index.md.

    Returns:
        True if files are equal, otherwise False.
    """
    if not index_path.exists():
        return False
    return read_text(readme_path) == read_text(index_path)


def sync_readme(readme_path: Path, index_path: Path) -> bool:
    """Copy README content into docs/index.md.

    Args:
        readme_path: Path to README.md.
        index_path: Path to docs/index.md.

    Returns:
        True if docs/index.md was updated, otherwise False.
    """
    readme_text = read_text(readme_path)
    if index_path.exists() and readme_text == read_text(index_path):
        return False

    index_path.write_text(readme_text, encoding="utf-8")
    return True


def main() -> int:
    """Run the CLI.

    Returns:
        Process exit code.
    """
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


if __name__ == "__main__":
    raise SystemExit(main())
