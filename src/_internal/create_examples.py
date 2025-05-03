"""Convert guides to examples as marimo notebooks."""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import re
import textwrap

import marimo._ast.cell
import marimo._convert.utils


GUIDES = {
    "user-guide": ("polars",),
    "data-backends": ("ibis-framework[duckdb]", "polars"),
    "power-analysis": (),
    "multiple-testing": ("polars",),
    "custom-metrics": (),
    "simulated-experiments": ("polars",),
}


def convert_guide(name: str, deps: tuple[str, ...]) -> None:
    with open(f"docs/{name}.md") as f:
        guide_text = f.read()

    sources = []
    cell_configs = []
    hide_code = marimo._ast.cell.CellConfig(hide_code=True)
    show_code = marimo._ast.cell.CellConfig(hide_code=False)
    for text in guide_text.split("```pycon"):
        if len(sources) == 0:
            md = text
        else:
            end_of_code = text.find("```")
            sources.append(convert_code(text[:end_of_code].strip()))
            cell_configs.append(show_code)
            md = text[end_of_code + 3:]

        sources.append(marimo._convert.utils.markdown_to_marimo(
            re_link.sub(update_link, md.strip()),
        ))
        cell_configs.append(hide_code)

    sources.append("import marimo as mo")
    cell_configs.append(hide_code)

    dependencies = "\n".join(
        f'#     "{dep}",'
        for dep in sorted((*deps, "marimo", "tea-tasting"))
    )
    header_comments = textwrap.dedent("""
        # /// script
        # requires-python = ">=3.10"
        # dependencies = [
        {dependencies}
        # ]
        # ///
    """).format(dependencies=dependencies)

    code = marimo._convert.utils.generate_from_sources(
        sources=sources,
        cell_configs=cell_configs,
        header_comments=header_comments,
    )
    with open(f"examples/{name}.py", "w") as f:
        f.write(code)


re_link = re.compile(r"\[([^\]]+)\]\((?!#)([^)]+)\)")

def update_link(match: re.Match[str]) -> str:
    label = match.group(1)
    url = match.group(2).replace(".md", "/")
    root = "" if url.startswith("http") else "https://tea-tasting.e10v.me/"
    return f"[{label}]({root}{url})"


re_doctest = re.compile(r"\s+# doctest:.*")

def convert_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        if line.startswith((">>> ", "... ")):
            lines.append(re_doctest.sub("", line[4:]))
        elif line.startswith("<BLANKLINE>") or line == "":
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    for name, deps in GUIDES.items():
        convert_guide(name, deps)
