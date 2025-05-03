"""Markdown extension that adds target="_blank" and rel="noopener" to external links."""
# ruff: noqa: N802
from __future__ import annotations

from typing import TYPE_CHECKING
import urllib.parse

import markdown
import markdown.extensions
import markdown.treeprocessors


if TYPE_CHECKING:
    import xml.etree.ElementTree as ET


class ExternalLinksTreeprocessor(markdown.treeprocessors.Treeprocessor):
    def run(self, root: ET.Element) -> None:
        for a in root.iter("a"):
            url = urllib.parse.urlparse(a.get("href", ""))
            if (
                url.scheme in {"http", "https"} and
                url.hostname is not None and
                not url.hostname.startswith(("tea-tasting.e10v.me", "127.0.0.1"))
            ):
                a.set("target", "_blank")
                a.set("rel", "noopener")

class ExternalLinksExtension(markdown.extensions.Extension):
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.treeprocessors.register(
            ExternalLinksTreeprocessor(md),
            "external_links",
            -1000,
        )

def makeExtension(**kwargs: dict[str, object]) -> ExternalLinksExtension:
    return ExternalLinksExtension(**kwargs)
