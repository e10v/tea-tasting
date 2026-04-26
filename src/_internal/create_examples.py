"""Convert guides to examples as marimo notebooks."""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import textwrap

import marimo._ast.cell
import marimo._ast.codegen
import marimo._ast.names
import marimo._convert.common


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
RE_GENERATED_WITH = re.compile(r'^__generated_with = "[^"]+"$', re.MULTILINE)


def convert_guide(name: str, deps: tuple[str, ...]) -> str:
    guide_text = Path(f"docs/{name}.md").read_text()

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

        sources.append(marimo._convert.common.markdown_to_marimo(convert_md(md)))
        cell_configs.append(HIDE_CODE)

    sources.append("import marimo as mo")
    cell_configs.append(HIDE_CODE)

    return marimo._ast.codegen.generate_filecontents(
        sources,
        [marimo._ast.names.DEFAULT_CELL_NAME for _ in sources],
        cell_configs,
        config=None,
        header_comments=create_header_comments(deps),
    )


def write_examples() -> None:
    for name, deps in GUIDES.items():
        Path(f"examples/{name}.py").write_text(convert_guide(name, deps))


def check_examples() -> int:
    out_of_sync = [
        f"examples/{name}.py"
        for name, deps in GUIDES.items()
        if normalize_example_for_check(Path(f"examples/{name}.py").read_text()) !=
            normalize_example_for_check(convert_guide(name, deps))
    ]
    if not out_of_sync:
        return 0

    sys.stderr.write("Examples are out of sync. Run:\n")
    sys.stderr.write("  uv run src/_internal/create_examples.py\n")
    sys.stderr.write("\nOut-of-sync files:\n")
    for path in out_of_sync:
        sys.stderr.write(f"  {path}\n")
    return 1


def normalize_example_for_check(code: str) -> str:
    return RE_GENERATED_WITH.sub('__generated_with = "<ignored>"', code)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert guides to marimo notebook examples.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether generated examples are in sync with guides.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(check_examples() if args.check else write_examples())
