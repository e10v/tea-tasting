"""Markdown extension that strips doctest artifacts."""
# ruff: noqa: N802
from __future__ import annotations

import re

import markdown
import markdown.extensions
import markdown.preprocessors


_doctest_pattern = re.compile(r"<BLANKLINE>|\s+# doctest:.*")
class StripDoctestArtifactsPreprocessor(markdown.preprocessors.Preprocessor):
    """A preprocessor that removes doctest artifacts."""
    def run(self, lines: list[str]) -> list[str]:
        """Run the preprocessor."""
        return [_doctest_pattern.sub("", line) for line in lines]

class StripDoctestArtifactsExtension(markdown.extensions.Extension):
    """An extension that registers the preprocessor."""
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        """Register the preprocessor."""
        md.preprocessors.register(
            StripDoctestArtifactsPreprocessor(md),
            "doctest_strip_preprocessor",
            175,
        )

def makeExtension(**kwargs: dict[str, object]) -> StripDoctestArtifactsExtension:
    """A factory function for the extension, required by Python-Markdown."""
    return StripDoctestArtifactsExtension(**kwargs)
