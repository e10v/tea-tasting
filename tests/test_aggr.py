from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import polars as pl
import pytest

import tea_tasting.aggr
import tea_tasting.datasets


if TYPE_CHECKING:
    import ibis.expr.types  # noqa: TC004
    import pandas as pd


    Frame = ibis.expr.types.Table | pd.DataFrame | pl.LazyFrame


COUNT = 100
MEAN = {"x": 5.0, "y": 4}
VAR = {"x": 3.0, "y": 2}
COV = {("x", "y"): 1.0}

@pytest.fixture
def aggr() -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=COUNT,
        mean_=MEAN,
        var_=VAR,
        cov_=COV,  # type: ignore
    )


@pytest.fixture
def data_pandas() -> pd.DataFrame:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)

@pytest.fixture
def data_polars(data_pandas: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(data_pandas)

@pytest.fixture
def data_polars_lazy(data_polars: pl.DataFrame) -> pl.LazyFrame:
    return data_polars.lazy()

@pytest.fixture
def data_duckdb(data_pandas: pd.DataFrame) -> ibis.expr.types.Table:
    return ibis.connect("duckdb://").create_table("data", data_pandas)

@pytest.fixture
def data_sqlite(data_pandas: pd.DataFrame) -> ibis.expr.types.Table:
    return ibis.connect("sqlite://").create_table("data", data_pandas)

@pytest.fixture(params=[
    "data_pandas", "data_polars", "data_polars_lazy", "data_duckdb", "data_sqlite"])
def data(request: pytest.FixtureRequest) -> Frame:
    return request.getfixturevalue(request.param)


@pytest.fixture
def correct_aggr(data_pandas: pd.DataFrame) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=len(data_pandas),
        mean_={
            "sessions": data_pandas["sessions"].mean(),
            "orders": data_pandas["orders"].mean(),
        },  # type: ignore
        var_={
            "sessions": data_pandas["sessions"].var(),
            "orders": data_pandas["orders"].var(),
        },  # type: ignore
        cov_={
            ("orders", "sessions"): data_pandas["sessions"].cov(data_pandas["orders"])},  # type: ignore
    )

@pytest.fixture
def correct_aggrs(data_pandas: pd.DataFrame) -> dict[int, tea_tasting.aggr.Aggregates]:
    return {
        v: tea_tasting.aggr.Aggregates(
            count_=len(d),
            mean_={"sessions": d["sessions"].mean(), "orders": d["orders"].mean()},  # type: ignore
            var_={"sessions": d["sessions"].var(), "orders": d["orders"].var()},  # type: ignore
            cov_={("orders", "sessions"): d["sessions"].cov(d["orders"])},  # type: ignore
        )
        for v, d in data_pandas.groupby("variant")
    }


def test_aggregates_init(aggr: tea_tasting.aggr.Aggregates):
    assert aggr.count_ == COUNT
    assert aggr.mean_ == MEAN
    assert aggr.var_ == VAR
    assert aggr.cov_ == COV

def test_aggregates_calls(aggr: tea_tasting.aggr.Aggregates):
    assert aggr.count() == COUNT
    assert aggr.mean("x") == MEAN["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.var("x") == VAR["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.cov("x", "y") == COV["x", "y"]

def test_aggregates_count_raises():
    aggr = tea_tasting.aggr.Aggregates(count_=None, mean_={}, var_={}, cov_={})
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
        count_=None,
        mean_={"a": 8, "b": 7, "c": 6, "d": 5},
        var_={},
        cov_={("a", "c"): 4, ("a", "d"): 3, ("b", "c"): 2, ("b", "d"): 1},
    )
    assert aggr.ratio_cov("a", "b", "c", "d") == pytest.approx(-0.0146938775510204)

def test_aggregates_add(data_pandas: pd.DataFrame):
    aggr = tea_tasting.aggr.Aggregates(
        count_=len(data_pandas),
        mean_={
            "sessions": data_pandas["sessions"].mean(),
            "orders": data_pandas["orders"].mean(),
        },  # type: ignore
        var_={
            "sessions": data_pandas["sessions"].var(),
            "orders": data_pandas["orders"].var(),
        },  # type: ignore
        cov_={
            ("sessions", "orders"): data_pandas["sessions"].cov(data_pandas["orders"])},  # type: ignore
    )
    aggrs = tuple(
        tea_tasting.aggr.Aggregates(
            count_=len(d),
            mean_={"sessions": d["sessions"].mean(), "orders": d["orders"].mean()},  # type: ignore
            var_={"sessions": d["sessions"].var(), "orders": d["orders"].var()},  # type: ignore
            cov_={("sessions", "orders"): d["sessions"].cov(d["orders"])},  # type: ignore
        )
        for _, d in data_pandas.groupby("variant")
    )
    aggrs_add = aggrs[0] + aggrs[1]
    assert aggrs_add.count_ == pytest.approx(aggr.count_)
    assert aggrs_add.mean_ == pytest.approx(aggr.mean_)
    assert aggrs_add.var_ == pytest.approx(aggr.var_)
    assert aggrs_add.cov_ == pytest.approx(aggr.cov_)


def test_read_aggregates_groups(
    data: Frame,
    correct_aggrs: dict[int, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.aggr.read_aggregates(
        data,
        group_col="variant",
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("sessions", "orders"),
        cov_cols=(("sessions", "orders"),),
    )
    for i in (0, 1):
        assert aggrs[i].count_ == pytest.approx(correct_aggrs[i].count_)
        assert aggrs[i].mean_ == pytest.approx(correct_aggrs[i].mean_)
        assert aggrs[i].var_ == pytest.approx(correct_aggrs[i].var_)
        assert aggrs[i].cov_ == pytest.approx(correct_aggrs[i].cov_)

def test_read_aggregates_no_groups(
    data: Frame,
    correct_aggr: tea_tasting.aggr.Aggregates,
):
    aggr = tea_tasting.aggr.read_aggregates(
        data,
        group_col=None,
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("sessions", "orders"),
        cov_cols=(("sessions", "orders"),),
    )
    assert aggr.count_ == pytest.approx(correct_aggr.count_)
    assert aggr.mean_ == pytest.approx(correct_aggr.mean_)
    assert aggr.var_ == pytest.approx(correct_aggr.var_)
    assert aggr.cov_ == pytest.approx(correct_aggr.cov_)

def test_read_aggregates_no_count(data_pandas: pd.DataFrame):
    aggr = tea_tasting.aggr.read_aggregates(
        data_pandas,
        group_col=None,
        has_count=False,
        mean_cols=("sessions", "orders"),
        var_cols=(),
        cov_cols=(),
    )
    assert aggr.count_ is None
    assert aggr.var_ == {}
    assert aggr.cov_ == {}
