from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.datasets


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
        cov_=COV,
    )

@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, rng=42)

@pytest.fixture
def correct_aggr(data_arrow: pa.Table) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=data_arrow.num_rows,
        mean_={
            "sessions": pc.mean(data_arrow["sessions"]).as_py(),
            "orders": pc.mean(data_arrow["orders"]).as_py(),
        },
        var_={
            "sessions": pc.variance(data_arrow["sessions"], ddof=1).as_py(),
            "orders": pc.variance(data_arrow["orders"], ddof=1).as_py(),
        },
        cov_={
            ("orders", "sessions"): np.cov(
                data_arrow["sessions"].combine_chunks().to_numpy(zero_copy_only=False),
                data_arrow["orders"].combine_chunks().to_numpy(zero_copy_only=False),
                ddof=1,
            )[0, 1],
        },
    )

@pytest.fixture
def correct_aggrs(data_arrow: pa.Table) -> dict[int, tea_tasting.aggr.Aggregates]:
    variant_col = data_arrow["variant"]
    aggrs = {}
    for var in variant_col.unique().to_pylist():
        var_data = data_arrow.filter(pc.equal(variant_col, pa.scalar(var)))
        aggrs |= {var: tea_tasting.aggr.Aggregates(
            count_=var_data.num_rows,
            mean_={
                "sessions": pc.mean(var_data["sessions"]).as_py(),
                "orders": pc.mean(var_data["orders"]).as_py(),
            },
            var_={
                "sessions": pc.variance(var_data["sessions"], ddof=1).as_py(),
                "orders": pc.variance(var_data["orders"], ddof=1).as_py(),
            },
            cov_={
                ("orders", "sessions"): np.cov(
                    var_data["sessions"].combine_chunks().to_numpy(zero_copy_only=False),
                    var_data["orders"].combine_chunks().to_numpy(zero_copy_only=False),
                    ddof=1,
                )[0, 1],
            },
        )}
    return aggrs


def test_aggregates_init(aggr: tea_tasting.aggr.Aggregates) -> None:
    assert aggr.count_ == COUNT
    assert aggr.mean_ == MEAN
    assert aggr.var_ == VAR
    assert aggr.cov_ == COV

def test_aggregates_calls(aggr: tea_tasting.aggr.Aggregates) -> None:
    assert aggr.count() == COUNT
    assert aggr.mean("x") == MEAN["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.var("x") == VAR["x"]
    assert aggr.mean("y") == MEAN["y"]
    assert aggr.cov("x", "y") == COV["x", "y"]

def test_aggregates_count_raises() -> None:
    aggr = tea_tasting.aggr.Aggregates(count_=None, mean_={}, var_={}, cov_={})
    with pytest.raises(RuntimeError):
        aggr.count()

def test_aggregates_none(aggr: tea_tasting.aggr.Aggregates) -> None:
    assert aggr.mean(None) == 1
    assert aggr.var(None) == 0
    assert aggr.cov(None, "y") == 0
    assert aggr.cov("x", None) == 0

def test_aggregates_ratio_var(aggr: tea_tasting.aggr.Aggregates) -> None:
    assert aggr.ratio_var("x", "y") == pytest.approx(0.2265625)

def test_aggregates_ratio_cov() -> None:
    aggr = tea_tasting.aggr.Aggregates(
        count_=None,
        mean_={"a": 8, "b": 7, "c": 6, "d": 5},
        var_={},
        cov_={("a", "c"): 4, ("a", "d"): 3, ("b", "c"): 2, ("b", "d"): 1},
    )
    assert aggr.ratio_cov("a", "b", "c", "d") == pytest.approx(-0.0146938775510204)

def test_aggregates_add(
    correct_aggr: tea_tasting.aggr.Aggregates,
    correct_aggrs: dict[int, tea_tasting.aggr.Aggregates],
) -> None:
    aggrs_add = correct_aggrs[0] + correct_aggrs[1]
    assert aggrs_add.count_ == pytest.approx(correct_aggr.count_)
    assert aggrs_add.mean_ == pytest.approx(correct_aggr.mean_)
    assert aggrs_add.var_ == pytest.approx(correct_aggr.var_)
    assert aggrs_add.cov_ == pytest.approx(correct_aggr.cov_)
