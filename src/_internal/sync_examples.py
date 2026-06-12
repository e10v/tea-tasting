"""Sync examples with guides as marimo notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import textwrap
import tomllib

import marimo._ast.cell
import marimo._ast.codegen
import marimo._ast.names
import marimo._convert.common


DOCS_DIR = Path("docs")
EXAMPLES_DIR = Path("examples")
PYPROJECT_PATH = Path("pyproject.toml")
SITE_URL = "https://tea-tasting.e10v.me/"

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


def main() -> int:
    args = parse_args()

    if args.check:
        out_of_sync = get_out_of_sync_examples()
        if not out_of_sync:
            sys.stdout.write("Examples are in sync with guides.\n")
            return 0

        sys.stdout.write(
            "Examples are out of sync with guides. "
            "Run `uv run python src/_internal/sync_examples.py`.\n",
        )
        sys.stdout.write("\nOut-of-sync files:\n")
        for path in out_of_sync:
            sys.stdout.write(f"  {path}\n")
        return 1

    if sync_examples():
        sys.stdout.write("Updated examples from guides.\n")
    else:
        sys.stdout.write("Examples are already up to date.\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync guides to marimo notebook examples.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether generated examples are in sync with guides.",
    )
    return parser.parse_args()


def get_out_of_sync_examples() -> list[Path]:
    out_of_sync = []
    for name, deps in GUIDES.items():
        path = EXAMPLES_DIR / f"{name}.py"
        content = "" if not path.exists() else path.read_text(encoding="utf-8")
        if normalize_example_for_check(content) != normalize_example_for_check(
            convert_guide(name, deps),
        ):
            out_of_sync.append(path)
    return out_of_sync


def sync_examples() -> bool:
    updated = False
    for name, deps in GUIDES.items():
        path = EXAMPLES_DIR / f"{name}.py"
        content = convert_guide(name, deps)
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue

        path.write_text(content, encoding="utf-8")
        updated = True
    return updated


def normalize_example_for_check(code: str) -> str:
    return RE_GENERATED_WITH.sub('__generated_with = "<ignored>"', code)


def convert_guide(name: str, deps: tuple[str, ...]) -> str:
    guide_text = (DOCS_DIR / f"{name}.md").read_text(encoding="utf-8")

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
    root = "" if url.startswith("http") else SITE_URL
    return f"[{label}]({root}{url})"


def create_header_comments(deps: tuple[str, ...]) -> str:
    dependencies = "\n".join(
        f'#     "{dep}",'
        for dep in sorted((*deps, "marimo", "tea-tasting"))
    )
    requires_python = get_requires_python()
    return textwrap.dedent("""
        # /// script
        # requires-python = "{requires_python}"
        # dependencies = [
        {dependencies}
        # ]
        # [tool.marimo.display]
        # cell_output = "below"
        # ///
    """).format(dependencies=dependencies, requires_python=requires_python)


def get_requires_python() -> str:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return pyproject["project"]["requires-python"]


if __name__ == "__main__":
    raise SystemExit(main())
