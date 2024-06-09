from __future__ import annotations

import math

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

def test_check_scalar_is_in():
    assert tea_tasting.utils.check_scalar(1, in_={0, 1}) == 1
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.check_scalar(1, in_={0, 2})


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

def test_auto_check_n_resamples():
    assert tea_tasting.utils.auto_check(1, "n_resamples") == 1
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.auto_check(0, "n_resamples")
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0.5, "n_resamples")

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
    assert tea_tasting.utils.format_num(12345, thousands_sep=" ") == "12 345"
    assert tea_tasting.utils.format_num(1.2345, decimal_point=",") == "1,23"
    assert tea_tasting.utils.format_num(0) == "0.00"


def test_div():
    assert tea_tasting.utils.div(1, 2) == 0.5
    assert tea_tasting.utils.div(1, 0, 3) == 3
    assert math.isnan(tea_tasting.utils.div(0, 0))
    assert tea_tasting.utils.div(1, 0) == float("inf")
    assert tea_tasting.utils.div(-1, 0) == float("-inf")


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
    assert typ(-1) / 0 == float("-inf")
    assert -1 / typ(0) == float("-inf")
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
    assert typ(-1) / 0 == float("-inf")
    assert -1 / typ(0) == float("-inf")
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


def test_repr_mixin_repr():
    class Repr(tea_tasting.utils.ReprMixin):
        def __init__(self, a: int, b: bool, c: str) -> None:
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
