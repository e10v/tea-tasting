from __future__ import annotations

import math
import textwrap

import pandas as pd
import pandas.testing
import polars as pl
import polars.testing
import pyarrow as pa
import pytest

import tea_tasting.utils


def test_check_scalar_typ():
    assert tea_tasting.utils.check_scalar(1, typ=int) == 1
    with pytest.raises(TypeError):
        tea_tasting.utils.check_scalar(1, typ=str)

def test_check_scalar_ge():
    assert tea_tasting.utils.check_scalar(1, ge=1) == 1
    with pytest.raises(ValueError, match="must be >="):
        tea_tasting.utils.check_scalar(1, ge=2)

def test_check_scalar_gt():
    assert tea_tasting.utils.check_scalar(1, gt=0) == 1
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.check_scalar(1, gt=1)

def test_check_scalar_le():
    assert tea_tasting.utils.check_scalar(1, le=1) == 1
    with pytest.raises(ValueError, match="must be <="):
        tea_tasting.utils.check_scalar(1, le=0)

def test_check_scalar_lt():
    assert tea_tasting.utils.check_scalar(1, lt=2) == 1
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.check_scalar(1, lt=1)

def test_check_scalar_ne():
    assert tea_tasting.utils.check_scalar(1, ne=2) == 1
    with pytest.raises(ValueError, match="must be !="):
        tea_tasting.utils.check_scalar(1, ne=1)

def test_check_scalar_is_in():
    assert tea_tasting.utils.check_scalar(1, in_={0, 1}) == 1
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.check_scalar(1, in_={0, 2})


def test_auto_check_alpha():
    assert tea_tasting.utils.auto_check(0.05, "alpha") == 0.05
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "alpha")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0.0, "alpha")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.auto_check(1.0, "alpha")

def test_auto_check_alternative():
    assert tea_tasting.utils.auto_check("two-sided", "alternative") == "two-sided"
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(2, "alternative")
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.auto_check("2s", "alternative")

def test_auto_check_confidence_level():
    assert tea_tasting.utils.auto_check(0.95, "confidence_level") == 0.95
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "confidence_level")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0.0, "confidence_level")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.auto_check(1.0, "confidence_level")

def test_auto_check_correction():
    assert tea_tasting.utils.auto_check(True, "correction") is True
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "correction")

def test_auto_check_equal_var():
    assert tea_tasting.utils.auto_check(True, "equal_var") is True
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "equal_var")

def test_auto_check_n_obs():
    assert tea_tasting.utils.auto_check(2, "n_obs") == 2
    assert tea_tasting.utils.auto_check((2, 3), "n_obs") == (2, 3)
    assert tea_tasting.utils.auto_check(None, "n_obs") is None
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0.5, "n_obs")
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check((0.5, 2), "n_obs")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(1, "n_obs")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check((1, 2), "n_obs")

def test_auto_check_n_resamples():
    assert tea_tasting.utils.auto_check(1, "n_resamples") == 1
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0, "n_resamples")
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0.5, "n_resamples")

def test_auto_check_power():
    assert tea_tasting.utils.auto_check(0.8, "power") == 0.8
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "power")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0.0, "power")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.auto_check(1.0, "power")

def test_auto_check_ratio():
    assert tea_tasting.utils.auto_check(1.5, "ratio") == 1.5
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check("str", "ratio")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0.0, "ratio")

def test_auto_check_use_t():
    assert tea_tasting.utils.auto_check(False, "use_t") is False
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "use_t")


def test_format_num():
    assert tea_tasting.utils.format_num(1.2345) == "1.23"
    assert tea_tasting.utils.format_num(0.9999) == "1.00"
    assert tea_tasting.utils.format_num(1.2345, sig=2) == "1.2"
    assert tea_tasting.utils.format_num(0.12345, pct=True) == "12.3%"
    assert tea_tasting.utils.format_num(None) == "-"
    assert tea_tasting.utils.format_num(float("nan")) == "-"
    assert tea_tasting.utils.format_num(float("inf")) == "∞"
    assert tea_tasting.utils.format_num(float("-inf")) == "-∞"
    assert tea_tasting.utils.format_num(0.00012345) == "1.23e-04"
    assert tea_tasting.utils.format_num(0.00099999) == "1.00e-03"
    assert tea_tasting.utils.format_num(9_999_999) == "9999999"
    assert tea_tasting.utils.format_num(10_000_000) == "1.00e+07"
    assert tea_tasting.utils.format_num(12345, thousands_sep=" ") == "12 345"
    assert tea_tasting.utils.format_num(1.2345, decimal_point=",") == "1,23"
    assert tea_tasting.utils.format_num(0) == "0.00"


def test_get_and_format_num():
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
    assert tea_tasting.utils.get_and_format_num(data, "name") == "metric"
    assert tea_tasting.utils.get_and_format_num(data, "metric") == "0.123"
    assert tea_tasting.utils.get_and_format_num(data, "rel_metric") == "12%"
    assert tea_tasting.utils.get_and_format_num(data, "metric_ci") == "[0.123, 0.988]"
    assert tea_tasting.utils.get_and_format_num(data, "rel_metric_ci") == "[12%, 99%]"
    assert tea_tasting.utils.get_and_format_num(data, "power") == "88%"


@pytest.fixture
def dicts_repr() -> tea_tasting.utils.DictsReprMixin:
    class DictsRepr(tea_tasting.utils.DictsReprMixin):
        default_keys = ("a", "b")
        def to_dicts(self) -> tuple[dict[str, object], ...]:
            return (
                {"a": 0.12345, "b": 0.23456},
                {"a": 0.34567, "b": 0.45678},
                {"a": 0.56789, "b": 0.67890},
            )
    return DictsRepr()

def test_dicts_repr_mixin_to_arrow(dicts_repr: tea_tasting.utils.DictsReprMixin):
    assert dicts_repr.to_arrow().equals(pa.table({
        "a": (0.12345, 0.34567, 0.56789),
        "b": (0.23456, 0.45678, 0.67890),
    }))

def test_dicts_repr_mixin_to_pandas(dicts_repr: tea_tasting.utils.DictsReprMixin):
    pandas.testing.assert_frame_equal(
        dicts_repr.to_pandas(),
        pd.DataFrame({
            "a": (0.12345, 0.34567, 0.56789),
            "b": (0.23456, 0.45678, 0.67890),
        }),
    )

def test_dicts_repr_mixin_to_polars(dicts_repr: tea_tasting.utils.DictsReprMixin):
    polars.testing.assert_frame_equal(
        dicts_repr.to_polars(),
        pl.DataFrame({
            "a": (0.12345, 0.34567, 0.56789),
            "b": (0.23456, 0.45678, 0.67890),
        }),
    )

def test_dicts_repr_mixin_to_pretty_dicts(
    dicts_repr: tea_tasting.utils.DictsReprMixin,
):
    assert dicts_repr.to_pretty_dicts() == [
        {"a": "0.123", "b": "0.235"},
        {"a": "0.346", "b": "0.457"},
        {"a": "0.568", "b": "0.679"},
    ]

def test_dicts_repr_mixin_to_string(dicts_repr: tea_tasting.utils.DictsReprMixin):
    assert dicts_repr.to_string() == textwrap.dedent("""\
            a     b
        0.123 0.235
        0.346 0.457
        0.568 0.679""")

def test_dicts_repr_mixin_to_html(dicts_repr: tea_tasting.utils.DictsReprMixin):
    assert dicts_repr.to_html() == (
        '<table class="dataframe" style="text-align: right;">'
        '<thead><tr><th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td>0.123</td><td>0.235</td></tr>'
        '<tr><td>0.346</td><td>0.457</td></tr>'
        '<tr><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )

def test_dicts_repr_mixin_to_html_indent(
    dicts_repr: tea_tasting.utils.DictsReprMixin,
):
    assert dicts_repr.to_html(indent="    ") == textwrap.dedent("""\
        <table class="dataframe" style="text-align: right;">
            <thead>
                <tr>
                    <th>a</th>
                    <th>b</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>0.123</td>
                    <td>0.235</td>
                </tr>
                <tr>
                    <td>0.346</td>
                    <td>0.457</td>
                </tr>
                <tr>
                    <td>0.568</td>
                    <td>0.679</td>
                </tr>
            </tbody>
        </table>""")

def test_dicts_repr_mixin_str(dicts_repr: tea_tasting.utils.DictsReprMixin):
    assert str(dicts_repr) == textwrap.dedent("""\
            a     b
        0.123 0.235
        0.346 0.457
        0.568 0.679""")

def test_dicts_repr_mixin_repr_html(dicts_repr: tea_tasting.utils.DictsReprMixin):
    assert dicts_repr._repr_html_() == (
        '<table class="dataframe" style="text-align: right;">'
        '<thead><tr><th>a</th><th>b</th></tr></thead>'
        '<tbody>'
        '<tr><td>0.123</td><td>0.235</td></tr>'
        '<tr><td>0.346</td><td>0.457</td></tr>'
        '<tr><td>0.568</td><td>0.679</td></tr>'
        '</tbody></table>'
    )


def test_repr_mixin_repr():
    class Repr(tea_tasting.utils.ReprMixin):
        def __init__(self, a: int, *, b: bool, c: str) -> None:
            self._a = -1
            self.a_ = -1
            self.a = a
            self.b_ = -1
            self.b = b
            self.c = c
    r = Repr(a=1, b=False, c="c")
    assert repr(r) == f"Repr(a=1, b=False, c={'c'!r})"

def test_repr_mixin_repr_obj():
    class Obj(tea_tasting.utils.ReprMixin):
        ...
    obj = Obj()
    assert repr(obj) == "Obj()"

def test_repr_mixin_repr_pos():
    class Pos(tea_tasting.utils.ReprMixin):
        def __init__(self, *args: int) -> None:
            self.args = args
    pos = Pos(1, 2, 3)
    with pytest.raises(RuntimeError):
        repr(pos)


def test_div():
    assert tea_tasting.utils.div(1, 2) == 0.5
    assert tea_tasting.utils.div(1, 0, 3) == 3
    assert tea_tasting.utils.div(1, 0) == float("inf")
    assert math.isnan(tea_tasting.utils.div(0, 0))
    assert math.isnan(tea_tasting.utils.div(-1, 0))


def test_float():
    typ = tea_tasting.utils.Float
    assert typ(1) + 2 == typ(3)
    assert 1 + typ(2) == typ(3)
    assert typ(1) - 2 == typ(-1)
    assert 1 - typ(2) == typ(-1)
    assert typ(1) / 2 == 0.5
    assert 1 / typ(2) == 0.5
    assert math.isnan(typ(0) / 0)
    assert math.isnan(0 / typ(0))
    assert typ(1) / 0 == float("inf")
    assert 1 / typ(0) == float("inf")
    assert math.isnan(typ(-1) / 0)
    assert math.isnan(-1 / typ(0))
    assert typ(5) // 2 == typ(2)
    assert 5 // typ(2) == typ(2)
    assert typ(5) % 2 == typ(1)
    assert 5 % typ(2) == typ(1)
    assert divmod(typ(5), 2) == (typ(2), typ(1))
    assert divmod(5, typ(2)) == (typ(2), typ(1))
    assert typ(2) ** 3 == typ(8)
    assert 2 ** typ(3) == typ(8)
    assert -typ(1) == typ(-1)
    assert +typ(1) == typ(1)
    assert abs(typ(-1)) == typ(1)
    assert int(typ(1.0)) == 1
    assert float(typ(1.0)) == 1.0
    assert round(typ(11), -1) == typ(10)
    assert math.trunc(typ(1.2)) == typ(1)
    assert math.floor(typ(1.2)) == typ(1)
    assert math.ceil(typ(1.2)) == typ(2)


def test_int():
    typ = tea_tasting.utils.Int
    assert typ(1) + 2 == typ(3)
    assert 1 + typ(2) == typ(3)
    assert typ(1) - 2 == typ(-1)
    assert 1 - typ(2) == typ(-1)
    assert typ(1) / 2 == 0.5
    assert 1 / typ(2) == 0.5
    assert math.isnan(typ(0) / 0)
    assert math.isnan(0 / typ(0))
    assert typ(1) / 0 == float("inf")
    assert 1 / typ(0) == float("inf")
    assert math.isnan(typ(-1) / 0)
    assert math.isnan(-1 / typ(0))
    assert typ(5) // 2 == typ(2)
    assert 5 // typ(2) == typ(2)
    assert typ(5) % 2 == typ(1)
    assert 5 % typ(2) == typ(1)
    assert divmod(typ(5), 2) == (typ(2), typ(1))
    assert divmod(5, typ(2)) == (typ(2), typ(1))
    assert typ(2) ** 3 == typ(8)
    assert 2 ** typ(3) == typ(8)
    assert -typ(1) == typ(-1)
    assert +typ(1) == typ(1)
    assert abs(typ(-1)) == typ(1)
    assert int(typ(1.0)) == 1
    assert float(typ(1.0)) == 1.0
    assert round(typ(11), -1) == typ(10)
    assert math.trunc(typ(1.2)) == typ(1)
    assert math.floor(typ(1.2)) == typ(1)
    assert math.ceil(typ(1.2)) == typ(1)


def test_numeric():
    assert isinstance(tea_tasting.utils.numeric(1), tea_tasting.utils.Int)
    assert isinstance(tea_tasting.utils.numeric("1"), tea_tasting.utils.Int)
    assert isinstance(tea_tasting.utils.numeric(1.0), tea_tasting.utils.Float)
    assert isinstance(tea_tasting.utils.numeric("inf"), tea_tasting.utils.Float)
