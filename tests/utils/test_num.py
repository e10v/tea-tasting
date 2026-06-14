from __future__ import annotations

import importlib
import math


numeric_utils = importlib.import_module("tea_tasting.utils.num")


def test_div() -> None:
    assert numeric_utils.div(1, 2) == 0.5
    assert numeric_utils.div(1, 0, 3) == 3
    assert numeric_utils.div(1, 0) == float("inf")
    assert math.isnan(numeric_utils.div(0, 0))
    assert math.isnan(numeric_utils.div(-1, 0))


def test_float() -> None:
    typ = numeric_utils.Float
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


def test_int() -> None:
    typ = numeric_utils.Int
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


def test_numeric() -> None:
    assert isinstance(
        numeric_utils.numeric(1),
        numeric_utils.Int,
    )
    assert isinstance(
        numeric_utils.numeric("1"),
        numeric_utils.Int,
    )
    assert isinstance(
        numeric_utils.numeric(1.0),
        numeric_utils.Float,
    )
    assert isinstance(
        numeric_utils.numeric("inf"),
        numeric_utils.Float,
    )
