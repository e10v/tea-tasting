"""Convert guides to examples as marimo notebooks."""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import re
import textwrap

import marimo._ast.cell
import marimo._convert.utils


GUIDES: dict[str, tuple[str, ...]] = {
    "user-guide": ("polars",),
    "data-backends": ("ibis-framework[duckdb]", "polars"),
    "power-analysis": (),
    "multiple-testing": ("polars",),
    "custom-metrics": (),
    "simulated-experiments": ("polars",),
}

HIDE_CODE = marimo._ast.cell.CellConfig(hide_code=True)
SHOW_CODE = marimo._ast.cell.CellConfig(hide_code=False)

RE_LINK = re.compile(r"\[([^\]]+)\]\((?!#)([^)]+)\)")
RE_DOCTEST = re.compile(r"\s+# doctest:.*")


def convert_guide(name: str, deps: tuple[str, ...]) -> None:
    with open(f"docs/{name}.md") as f:
        guide_text = f.read()

    sources = []
    cell_configs = []
    for text in guide_text.split("```pycon"):
        if len(sources) == 0:
            md = text
        else:
            end_of_code = text.find("```")
            md = text[end_of_code + 3:]
            sources.append(convert_code(text[:end_of_code]))
            cell_configs.append(SHOW_CODE)

        sources.append(marimo._convert.utils.markdown_to_marimo(convert_md(md)))
        cell_configs.append(HIDE_CODE)

    sources.append("import marimo as mo")
    cell_configs.append(HIDE_CODE)

    code = marimo._convert.utils.generate_from_sources(
        sources=sources,
        cell_configs=cell_configs,
        header_comments=create_header_comments(deps),
    )
    with open(f"examples/{name}.py", "w") as f:
        f.write(code)


def convert_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        if line == ">>> import tqdm":
            pass
        elif line.startswith((">>>", "...")):
            lines.append(RE_DOCTEST.sub("", line[4:]))
        elif line == "":
            lines.append("")
    return "\n".join(lines).strip().replace("tqdm.tqdm", "mo.status.progress_bar")


def convert_md(md: str) -> str:
    return (
        RE_LINK.sub(update_link, md.strip())
        .replace(
            "[tqdm](https://github.com/tqdm/tqdm)",
            "[marimo](https://github.com/marimo-team/marimo)",
        )
        .replace(" tqdm", " marimo")
    )


def update_link(match: re.Match[str]) -> str:
    label = match.group(1)
    url = match.group(2).replace(".md", "/")
    root = "" if url.startswith("http") else "https://tea-tasting.e10v.me/"
    return f"[{label}]({root}{url})"


def create_header_comments(deps: tuple[str, ...]) -> str:
    dependencies = "\n".join(
        f'#     "{dep}",'
        for dep in sorted((*deps, "marimo", "tea-tasting"))
    )
    return textwrap.dedent("""
        # /// script
        # requires-python = ">=3.10"
        # dependencies = [
        {dependencies}
        # ]
        # [tool.marimo.display]
        # cell_output = "below"
        # ///
    """).format(dependencies=dependencies)


if __name__ == "__main__":
    for name, deps in GUIDES.items():
        convert_guide(name, deps)
