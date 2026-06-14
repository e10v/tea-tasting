from __future__ import annotations

import builtins
import textwrap
from typing import TYPE_CHECKING
import unittest.mock

import pandas as pd
import pandas.testing
import polars as pl
import polars.testing
import pyarrow as pa
import pytest

import tea_tasting.utils.display


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal


def test_format_num() -> None:
    assert tea_tasting.utils.display.format_num(1.2345) == "1.23"
    assert tea_tasting.utils.display.format_num(0.9999) == "1.00"
    assert tea_tasting.utils.display.format_num(1.2345, sig=2) == "1.2"
    assert tea_tasting.utils.display.format_num(0.12345, pct=True) == "12.3%"
    assert tea_tasting.utils.display.format_num(None) == "-"
    assert tea_tasting.utils.display.format_num(float("nan")) == "-"
    assert tea_tasting.utils.display.format_num(float("inf")) == "∞"
    assert tea_tasting.utils.display.format_num(float("-inf")) == "-∞"
    assert tea_tasting.utils.display.format_num(0.00012345) == "1.23e-04"
    assert tea_tasting.utils.display.format_num(0.00099999) == "1.00e-03"
    assert tea_tasting.utils.display.format_num(9_999_999) == "9999999"
    assert tea_tasting.utils.display.format_num(10_000_000) == "1.00e+07"
    assert tea_tasting.utils.display.format_num(12345, thousands_sep=" ") == "12 345"
    assert tea_tasting.utils.display.format_num(1.2345, decimal_point=",") == "1,23"
    assert tea_tasting.utils.display.format_num(0) == "0.00"


def test_get_and_format_num() -> None:
    data: dict[str, object] = {
        "name": "metric",
        "metric": 0.12345,
        "rel_metric": 0.12345,
        "metric_ci_lower": 0.12345,
        "metric_ci_upper": 0.98765,
        "rel_metric_ci_lower": 0.12345,
        "rel_metric_ci_upper": 0.98765,
        "power": 0.87654,
    }
    assert tea_tasting.utils.display.get_and_format_num(data, "name") == "metric"
    assert tea_tasting.utils.display.get_and_format_num(data, "metric") == "0.123"
    assert tea_tasting.utils.display.get_and_format_num(data, "rel_metric") == "12%"
    assert (
        tea_tasting.utils.display.get_and_format_num(data, "metric_ci")
        == "[0.123, 0.988]"
    )
    assert (
        tea_tasting.utils.display.get_and_format_num(data, "rel_metric_ci")
        == "[12%, 99%]"
    )
    assert tea_tasting.utils.display.get_and_format_num(data, "power") == "88%"


@pytest.fixture
def dicts_repr() -> tea_tasting.utils.display.DictsReprMixin:
    class DictsRepr(tea_tasting.utils.display.DictsReprMixin):
        default_keys = ("name", "a", "b")
        default_text_keys = ("name",)
        def to_dicts(self) -> tuple[dict[str, object], ...]:
            return (
                {"name": "x", "a": 0.12345, "b": 0.23456},
                {"name": "yy", "a": 0.34567, "b": 0.45678},
                {"name": "zzz", "a": 0.56789, "b": 0.67890},
            )
    return DictsRepr()

def test_dicts_repr_mixin_to_arrow(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_arrow().equals(pa.table({
        "name": ("x", "yy", "zzz"),
        "a": (0.12345, 0.34567, 0.56789),
        "b": (0.23456, 0.45678, 0.67890),
    }))

def test_dicts_repr_mixin_to_pandas(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    pandas.testing.assert_frame_equal(
        dicts_repr.to_pandas(),
        pd.DataFrame({
            "name": ("x", "yy", "zzz"),
            "a": (0.12345, 0.34567, 0.56789),
            "b": (0.23456, 0.45678, 0.67890),
        }),
    )

def test_dicts_repr_mixin_to_polars(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    polars.testing.assert_frame_equal(
        dicts_repr.to_polars(),
        pl.DataFrame({
            "name": ("x", "yy", "zzz"),
            "a": (0.12345, 0.34567, 0.56789),
            "b": (0.23456, 0.45678, 0.67890),
        }),
    )

def test_dicts_repr_mixin_to_pretty_dicts_default(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_pretty_dicts() == [
        {"name": "x", "a": "0.123", "b": "0.235"},
        {"name": "yy", "a": "0.346", "b": "0.457"},
        {"name": "zzz", "a": "0.568", "b": "0.679"},
    ]

def test_dicts_repr_mixin_to_pretty_dicts_max_rows(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_pretty_dicts(max_rows=2) == [
        {"name": "x", "a": "0.123", "b": "0.235"},
        {"name": "…", "a": "…", "b": "…"},
        {"name": "zzz", "a": "0.568", "b": "0.679"},
    ]

def test_dicts_repr_mixin_with_defaults(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    new_dicts_repr = dicts_repr.with_defaults(
        keys=("c", "d"),
        max_rows=42,
        align="left",
    )
    assert new_dicts_repr.default_keys == ("c", "d")
    assert new_dicts_repr.default_text_keys == ("name",)
    assert new_dicts_repr.default_max_rows == 42
    assert new_dicts_repr.default_align == "left"

def test_dicts_repr_mixin_with_keys(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.with_keys(("c", "d")).default_keys == ("c", "d")

def test_dicts_repr_mixin_with_max_rows(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.with_max_rows(42).default_max_rows == 42

def test_dicts_repr_mixin_to_string(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string() == textwrap.dedent("""\
        name     a     b
        x    0.123 0.235
        yy   0.346 0.457
        zzz  0.568 0.679""")

def test_dicts_repr_mixin_to_string_align_left(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(align="left") == textwrap.dedent("""\
        name a     b
        x    0.123 0.235
        yy   0.346 0.457
        zzz  0.568 0.679""")

def test_dicts_repr_mixin_to_string_align_right(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(align="right") == textwrap.dedent("""\
        name     a     b
           x 0.123 0.235
          yy 0.346 0.457
         zzz 0.568 0.679""")

def test_dicts_repr_mixin_to_string_markdown(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(table_format="markdown") == textwrap.dedent("""\
        | name |     a |     b |
        | :--- | ----: | ----: |
        | x    | 0.123 | 0.235 |
        | yy   | 0.346 | 0.457 |
        | zzz  | 0.568 | 0.679 |""")


def test_dicts_repr_mixin_to_markdown(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_markdown() == textwrap.dedent("""\
        | name |     a |     b |
        | :--- | ----: | ----: |
        | x    | 0.123 | 0.235 |
        | yy   | 0.346 | 0.457 |
        | zzz  | 0.568 | 0.679 |""")


def test_dicts_repr_mixin_to_markdown_align_left_and_max_rows(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_markdown(
        align="left",
        max_rows=2,
    ) == textwrap.dedent("""\
        | name | a     | b     |
        | :--- | :---- | :---- |
        | x    | 0.123 | 0.235 |
        | …    | …     | …     |
        | zzz  | 0.568 | 0.679 |""")


def test_dicts_repr_mixin_to_string_markdown_align_left(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(
        align="left",
        table_format="markdown",
    ) == textwrap.dedent("""\
        | name | a     | b     |
        | :--- | :---- | :---- |
        | x    | 0.123 | 0.235 |
        | yy   | 0.346 | 0.457 |
        | zzz  | 0.568 | 0.679 |""")

def test_dicts_repr_mixin_to_string_markdown_align_right(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(
        align="right",
        table_format="markdown",
    ) == textwrap.dedent("""\
        | name |     a |     b |
        | ---: | ----: | ----: |
        |    x | 0.123 | 0.235 |
        |   yy | 0.346 | 0.457 |
        |  zzz | 0.568 | 0.679 |""")

def test_dicts_repr_mixin_to_string_markdown_max_rows(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_string(
        max_rows=2,
        table_format="markdown",
    ) == textwrap.dedent("""\
        | name |     a |     b |
        | :--- | ----: | ----: |
        | x    | 0.123 | 0.235 |
        | …    |     … |     … |
        | zzz  | 0.568 | 0.679 |""")

def test_dicts_repr_mixin_to_string_markdown_escape() -> None:
    class DictsRepr(tea_tasting.utils.display.DictsReprMixin):
        default_keys = ("na|me", "val")
        default_text_keys = ("na|me",)
        def to_dicts(self) -> tuple[dict[str, object], ...]:
            return ({"na|me": "a|b\\c\nd", "val": "x|y"},)

    assert DictsRepr().to_string(table_format="markdown") == textwrap.dedent("""\
        | na\\|me       |  val |
        | :----------- | ---: |
        | a\\|b\\\\c<br>d | x\\|y |""")

def test_dicts_repr_mixin_to_html(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_html() == (
        '<table class="dataframe" style="text-align: right;">'
        '<thead><tr><th style="text-align: left;">name</th>'
        '<th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td style="text-align: left;">x</td><td>0.123</td><td>0.235</td></tr>'
        '<tr><td style="text-align: left;">yy</td><td>0.346</td><td>0.457</td></tr>'
        '<tr><td style="text-align: left;">zzz</td><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )

def test_dicts_repr_mixin_to_html_align_left(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_html(align="left") == (
        '<table class="dataframe" style="text-align: left;">'
        '<thead><tr><th>name</th><th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td>x</td><td>0.123</td><td>0.235</td></tr>'
        '<tr><td>yy</td><td>0.346</td><td>0.457</td></tr>'
        '<tr><td>zzz</td><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )

def test_dicts_repr_mixin_to_html_align_right(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_html(align="right") == (
        '<table class="dataframe" style="text-align: right;">'
        '<thead><tr><th>name</th><th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td>x</td><td>0.123</td><td>0.235</td></tr>'
        '<tr><td>yy</td><td>0.346</td><td>0.457</td></tr>'
        '<tr><td>zzz</td><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )

def test_dicts_repr_mixin_to_html_indent(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr.to_html(indent="    ") == textwrap.dedent("""\
        <table class="dataframe" style="text-align: right;">
            <thead>
                <tr>
                    <th style="text-align: left;">name</th>
                    <th>a</th>
                    <th>b</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align: left;">x</td>
                    <td>0.123</td>
                    <td>0.235</td>
                </tr>
                <tr>
                    <td style="text-align: left;">yy</td>
                    <td>0.346</td>
                    <td>0.457</td>
                </tr>
                <tr>
                    <td style="text-align: left;">zzz</td>
                    <td>0.568</td>
                    <td>0.679</td>
                </tr>
            </tbody>
        </table>""")

def test_dicts_repr_mixin_mime_marimo(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    mime_type, data = dicts_repr._mime_()
    assert mime_type == "text/html"
    assert data.startswith("<marimo-ui-element")

@pytest.mark.parametrize(
    ("align", "expected"),
    [
        ("auto", {"name": "left", "a": "right", "b": "right"}),
        ("left", {"name": "left", "a": "left", "b": "left"}),
        ("right", {"name": "right", "a": "right", "b": "right"}),
    ],
)
def test_dicts_repr_mixin_mime_marimo_text_justify_columns(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
    align: Literal["auto", "left", "right"],
    expected: dict[str, str],
) -> None:
    class MockTable:
        def _mime_(self) -> tuple[str, str]:
            return "text/html", "<marimo-ui-element></marimo-ui-element>"

    with unittest.mock.patch("marimo.ui.table", return_value=MockTable()) as table:
        dicts_repr.with_defaults(align=align)._mime_()

    assert table.call_count == 1
    assert table.call_args.kwargs["text_justify_columns"] == expected

def test_dicts_repr_mixin_mime_exception(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    def import_side_effect(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> object:
        if name == "marimo":
            raise ImportError("No module named 'marimo'")
        return builtins.__import__(name, globals_, locals_, fromlist, level)
    with unittest.mock.patch("builtins.__import__", side_effect=import_side_effect):
        mime_type, data = dicts_repr._mime_()
    assert mime_type == "text/html"
    assert data.startswith('<table class="dataframe"')

def test_dicts_repr_mixin_repr_html(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert dicts_repr._repr_html_() == (
        '<table class="dataframe" style="text-align: right;">'
        '<thead><tr><th style="text-align: left;">name</th>'
        '<th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td style="text-align: left;">x</td><td>0.123</td><td>0.235</td></tr>'
        '<tr><td style="text-align: left;">yy</td><td>0.346</td><td>0.457</td></tr>'
        '<tr><td style="text-align: left;">zzz</td><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )

def test_dicts_repr_mixin_repr(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert repr(dicts_repr) == textwrap.dedent("""\
        name     a     b
        x    0.123 0.235
        yy   0.346 0.457
        zzz  0.568 0.679""")

def test_dicts_repr_mixin_str(
    dicts_repr: tea_tasting.utils.display.DictsReprMixin,
) -> None:
    assert str(dicts_repr) == textwrap.dedent("""\
        name     a     b
        x    0.123 0.235
        yy   0.346 0.457
        zzz  0.568 0.679""")


def test_repr_mixin_repr() -> None:
    class Repr(tea_tasting.utils.display.ReprMixin):
        def __init__(self, a: int, *, b: bool, c: str) -> None:
            self._a = -1
            self.a_ = -1
            self.a = a
            self.b_ = -1
            self.b = b
            self.c = c
    r = Repr(a=1, b=False, c="c")
    assert repr(r) == f"Repr(a=1, b=False, c={'c'!r})"

def test_repr_mixin_repr_obj() -> None:
    class Obj(tea_tasting.utils.display.ReprMixin):
        ...
    obj = Obj()
    assert repr(obj) == "Obj()"

def test_repr_mixin_repr_pos() -> None:
    class Pos(tea_tasting.utils.display.ReprMixin):
        def __init__(self, *args: int) -> None:
            self.args = args
    pos = Pos(1, 2, 3)
    with pytest.raises(RuntimeError):
        repr(pos)
