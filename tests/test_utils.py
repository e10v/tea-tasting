from __future__ import annotations

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
    assert tea_tasting.utils.check_scalar(1, is_in={0, 1}) == 1
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.check_scalar(1, is_in={0, 2})


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

def test_auto_check_equal_var():
    assert tea_tasting.utils.auto_check(True, "equal_var") is True
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "equal_var")

def test_auto_check_use_t():
    assert tea_tasting.utils.auto_check(False, "use_t") is False
    with pytest.raises(TypeError):
        tea_tasting.utils.auto_check(0, "use_t")


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
