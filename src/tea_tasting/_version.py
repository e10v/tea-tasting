"""Package version."""

from __future__ import annotations

import importlib.metadata
import importlib.resources


try:
    __version__ = importlib.metadata.version(__package__ or "tea-tasting")
except importlib.metadata.PackageNotFoundError:
    __version__ = (
        importlib.resources.files("tea_tasting")
        .joinpath("_version.txt")
        .read_text()
        .strip()
    )
