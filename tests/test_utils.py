from __future__ import annotations

import pytest

import tea_tasting.utils


def test_check_scalar_typ():
    tea_tasting.utils.check_scalar(1, typ=int)
    with pytest.raises(TypeError):
        tea_tasting.utils.check_scalar(1, typ=str)

def test_check_scalar_ge():
    tea_tasting.utils.check_scalar(1, ge=1)
    with pytest.raises(ValueError, match="must be >="):
        tea_tasting.utils.check_scalar(1, ge=2)

def test_check_scalar_gt():
    tea_tasting.utils.check_scalar(1, gt=0)
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.check_scalar(1, gt=1)

def test_check_scalar_le():
    tea_tasting.utils.check_scalar(1, le=1)
    with pytest.raises(ValueError, match="must be <="):
        tea_tasting.utils.check_scalar(1, le=0)

def test_check_scalar_lt():
    tea_tasting.utils.check_scalar(1, lt=2)
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.check_scalar(1, lt=1)


def test_repr_mixin_repr():
    class Repr(tea_tasting.utils.ReprMixin):
        def __init__(self, a: int, b: int, c: int) -> None:
            self._a = a
            self.a_ = -1
            self.a = -1
            self.b_ = b
            self.b = -1
            self.c = c
    r = Repr(a=1, b=2, c=3)
    assert repr(r) == "Repr(a=1, b=2, c=3)"

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
