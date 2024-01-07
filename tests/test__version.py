from __future__ import annotations

import importlib
import importlib.metadata
import unittest.mock

import tea_tasting._version


def test_version():
    assert isinstance(tea_tasting._version.__version__, str)

    with (
        unittest.mock.patch(
            "tea_tasting._version.importlib.metadata.version") as version,
        unittest.mock.patch("tea_tasting._version.importlib.resources.files") as files,
    ):
        (
            files.return_value
            .joinpath.return_value
            .read_text.return_value
            .strip.return_value
        ) = "version"

        version.side_effect = importlib.metadata.PackageNotFoundError("Not found")
        importlib.reload(tea_tasting._version)
        assert isinstance(tea_tasting._version.__version__, str)
