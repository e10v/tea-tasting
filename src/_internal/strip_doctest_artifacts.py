"""Markdown extension that strips doctest artifacts."""
# ruff: noqa: N802
from __future__ import annotations

import re

import markdown
import markdown.extensions
import markdown.preprocessors


RE_DOCTEST = re.compile(r"<BLANKLINE>|\s+# doctest:.*")

class StripDoctestArtifactsPreprocessor(markdown.preprocessors.Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        return [RE_DOCTEST.sub("", line) for line in lines]

class StripDoctestArtifactsExtension(markdown.extensions.Extension):
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.preprocessors.register(
            StripDoctestArtifactsPreprocessor(md),
            "strip_doctest_artifacts",
            175,
        )

def makeExtension(**kwargs: dict[str, object]) -> StripDoctestArtifactsExtension:
    return StripDoctestArtifactsExtension(**kwargs)
