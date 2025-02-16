"""Markdown extension that strips doctest artifacts."""
# ruff: noqa: N802
from __future__ import annotations

import re

import markdown
import markdown.extensions
import markdown.preprocessors


class StripDoctestArtifactsPreprocessor(markdown.preprocessors.Preprocessor):
    """A preprocessor that removes doctest artifacts."""

    def run(self, lines: list[str]) -> list[str]:
        """Run the preprocessor."""
        return [_strip(line) for line in lines]


_doctest_pattern = re.compile(r"\s+# doctest:.*")
def _strip(line: str) -> str:
    return _doctest_pattern.sub("", line.replace("<BLANKLINE>", ""))


class StripDoctestArtifactsExtension(markdown.extensions.Extension):
    """An extension that registers the preprocessor."""

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        """Register the preprocessor."""
        md.preprocessors.register(
            StripDoctestArtifactsPreprocessor(md),
            "blankline_strip_preprocessor",
            175,
        )


def makeExtension(**kwargs: dict[str, object]) -> StripDoctestArtifactsExtension:
    """A factory function for the extension, required by Python-Markdown."""
    return StripDoctestArtifactsExtension(**kwargs)
