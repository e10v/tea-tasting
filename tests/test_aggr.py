# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tea_tasting.aggr
import tea_tasting.datasets


if TYPE_CHECKING:
    from ibis.expr.types import Table


COUNT = 100
MEAN = {"x": 5.0, "y": 4}
VAR = {"x": 3.0, "y": 2}
COV = {("x", "y"): 1.0}

@pytest.fixture
def aggr() -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count=COUNT,
        mean=MEAN,
        var=VAR,
        cov=COV,  # type: ignore
    )


@pytest.fixture
def data() -> Table:
    return tea_tasting.datasets.sample_users_data(size=100, seed=42)


def test_aggregates_init(aggr: tea_tasting.aggr.Aggregates):
    assert aggr._count == COUNT
    assert aggr._mean == MEAN
    assert aggr._var == VAR
    assert aggr._cov == COV

def test_aggregates_repr(aggr: tea_tasting.aggr.Aggregates):
    from tea_tasting.aggr import Aggregates  # type: ignore # noqa: F401
    aggr_repr = eval(repr(aggr))  # noqa: S307, PGH001
    assert aggr_repr._count == aggr._count
    assert aggr_repr._mean == aggr._mean
    assert aggr_repr._var == aggr._var
    assert aggr_repr._cov == aggr._cov

def test_aggregates_calls(aggr: tea_tasting.aggr.Aggregates):
    assert aggr.count() == COUNT
    assert aggr.mean("x") == MEAN["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.var("x") == VAR["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.cov("x", "y") == COV[("x", "y")]

def test_aggregates_count_raises():
    aggr = tea_tasting.aggr.Aggregates(count=None, mean={}, var={}, cov={})
    with pytest.raises(RuntimeError):
        aggr.count()

def test_aggregates_none(aggr: tea_tasting.aggr.Aggregates):
    assert aggr.mean(None) == 1
    assert aggr.var(None) == 0
    assert aggr.cov(None, "y") == 0
    assert aggr.cov("x", None) == 0

def test_aggregates_ratio_var(aggr: tea_tasting.aggr.Aggregates):
    assert aggr.ratio_var("x", "y") == pytest.approx(0.2265625)

def test_aggregates_ratio_cov():
    aggr = tea_tasting.aggr.Aggregates(
        count=None,
        mean={"a": 8, "b": 7, "c": 6, "d": 5},
        var={},
        cov={("a", "c"): 4, ("a", "d"): 3, ("b", "c"): 2, ("b", "d"): 1},
    )
    assert aggr.ratio_cov("a", "b", "c", "d") == pytest.approx(-0.0146938775510204)

def test_aggregates_filter(aggr: tea_tasting.aggr.Aggregates):
    filtered_aggr = aggr.filter(
        has_count=False,
        mean_cols=("x",),
        var_cols=("x",),
        cov_cols=(),
    )
    assert filtered_aggr._count == COUNT
    assert filtered_aggr._mean == {"x": MEAN["x"]}
    assert filtered_aggr._var == {"x": VAR["x"]}
    assert filtered_aggr._cov == {}

def test_aggregates_add(data: Table):
    d = data.to_pandas()
    aggr = tea_tasting.aggr.Aggregates(
        count=len(d),
        mean={"visits": d["visits"].mean(), "orders": d["orders"].mean()},  # type: ignore
        var={"visits": d["visits"].var(), "orders": d["orders"].var()},  # type: ignore
        cov={("visits", "orders"): d["visits"].cov(d["orders"])},  # type: ignore
    )
    aggrs = tuple(
        tea_tasting.aggr.Aggregates(
            count=len(d),
            mean={"visits": d["visits"].mean(), "orders": d["orders"].mean()},  # type: ignore
            var={"visits": d["visits"].var(), "orders": d["orders"].var()},  # type: ignore
            cov={("visits", "orders"): d["visits"].cov(d["orders"])},  # type: ignore
        )
        for _, d in d.groupby("variant")
    )
    aggrs_add = aggrs[0] + aggrs[1]
    assert aggrs_add._count == pytest.approx(aggr._count)
    assert aggrs_add._mean == pytest.approx(aggr._mean)
    assert aggrs_add._var == pytest.approx(aggr._var)
    assert aggrs_add._cov == pytest.approx(aggr._cov)


def test_read_aggregates_groups(data: Table):
    correct_aggrs = {
        v: tea_tasting.aggr.Aggregates(
            count=len(d),
            mean={"visits": d["visits"].mean(), "orders": d["orders"].mean()},  # type: ignore
            var={"visits": d["visits"].var(), "orders": d["orders"].var()},  # type: ignore
            cov={("orders", "visits"): d["visits"].cov(d["orders"])},  # type: ignore
        )
        for v, d in data.to_pandas().groupby("variant")
    }
    aggrs = tea_tasting.aggr.read_aggregates(
        data,
        group_col="variant",
        has_count=True,
        mean_cols=("visits", "orders"),
        var_cols=("visits", "orders"),
        cov_cols=(("visits", "orders"),),
    )
    for i in (0, 1):
        assert aggrs[i]._count == pytest.approx(correct_aggrs[i]._count)
        assert aggrs[i]._mean == pytest.approx(correct_aggrs[i]._mean)
        assert aggrs[i]._var == pytest.approx(correct_aggrs[i]._var)
        assert aggrs[i]._cov == pytest.approx(correct_aggrs[i]._cov)

def test_read_aggregates_no_groups(data: Table):
    d = data.to_pandas()
    correct_aggr = tea_tasting.aggr.Aggregates(
        count=len(d),
        mean={"visits": d["visits"].mean(), "orders": d["orders"].mean()},  # type: ignore
        var={"visits": d["visits"].var(), "orders": d["orders"].var()},  # type: ignore
        cov={("orders", "visits"): d["visits"].cov(d["orders"])},  # type: ignore
    )
    aggr = tea_tasting.aggr.read_aggregates(
        data,
        group_col=None,
        has_count=True,
        mean_cols=("visits", "orders"),
        var_cols=("visits", "orders"),
        cov_cols=(("visits", "orders"),),
    )
    assert aggr._count == pytest.approx(correct_aggr._count)
    assert aggr._mean == pytest.approx(correct_aggr._mean)
    assert aggr._var == pytest.approx(correct_aggr._var)
    assert aggr._cov == pytest.approx(correct_aggr._cov)

def test_read_aggregates_no_count(data: Table):
    aggr = tea_tasting.aggr.read_aggregates(
        data,
        group_col=None,
        has_count=False,
        mean_cols=("visits", "orders"),
        var_cols=(),
        cov_cols=(),
    )
    assert aggr._count is None
    assert aggr._var == {}
    assert aggr._cov == {}
