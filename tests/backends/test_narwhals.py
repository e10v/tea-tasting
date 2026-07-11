from __future__ import annotations

import math
from typing import TYPE_CHECKING

import narwhals as nw
import narwhals.typing
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.backends.narwhals


if TYPE_CHECKING:
    from collections.abc import Hashable

pytest_plugins = ("tests.fixtures",)


@pytest.fixture
def adapter(
    data_narwhals: narwhals.typing.IntoFrame,
) -> tea_tasting.backends.narwhals.NarwhalsFrame:
    return tea_tasting.backends.narwhals.NarwhalsFrame(data_narwhals)


@pytest.fixture
def group_adapter(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
) -> tea_tasting.backends.narwhals.NarwhalsFrameGroupBy:
    return tea_tasting.backends.narwhals.NarwhalsFrameGroupBy(
        adapter,
        "variant",
    )


def _expected_aggr(data: pa.Table) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=data.num_rows,
        mean_={
            "sessions": pc.mean(data["sessions"]).as_py(),
            "orders": pc.mean(data["orders"]).as_py(),
        },
        var_={
            "sessions": pc.variance(data["sessions"], ddof=1).as_py(),
            "orders": pc.variance(data["orders"], ddof=1).as_py(),
        },
        cov_={
            ("orders", "sessions"): np.cov(
                data["sessions"].combine_chunks().to_numpy(zero_copy_only=False),
                data["orders"].combine_chunks().to_numpy(zero_copy_only=False),
                ddof=1,
            )[0, 1],
        },
    )


def _expected_aggrs(data: pa.Table) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    variant_col = data["variant"]
    return {
        variant: _expected_aggr(
            data.filter(pc.equal(variant_col, pa.scalar(variant))),
        )
        for variant in variant_col.unique().to_pylist()
    }


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == pytest.approx(right.count_)
    assert left.mean_ == pytest.approx(right.mean_)
    assert left.var_ == pytest.approx(right.var_)
    assert left.cov_ == pytest.approx(right.cov_)


def test_narwhals_frame_init(data_narwhals: narwhals.typing.IntoFrame) -> None:
    adapter = tea_tasting.backends.narwhals.NarwhalsFrame(data_narwhals)
    assert isinstance(adapter._frame, nw.LazyFrame)
    assert adapter.cov == tea_tasting.backends.narwhals.IMPLEMENTATION_COV[
        adapter._frame.implementation
    ]


@pytest.mark.parametrize("cov", ["full", "ungrouped_only", "none"])
def test_narwhals_frame_init_cov(cov: str, data_arrow: pa.Table) -> None:
    adapter = tea_tasting.backends.narwhals.NarwhalsFrame(
        data_arrow,
        cov=cov,  # ty: ignore[invalid-argument-type]
    )
    assert adapter.cov == cov


def test_narwhals_frame_init_cov_invalid(data_arrow: pa.Table) -> None:
    with pytest.raises(ValueError, match="cov"):
        tea_tasting.backends.narwhals.NarwhalsFrame(
            data_arrow,
            cov="invalid",  # ty: ignore[invalid-argument-type]
        )


def test_narwhals_frame_select(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
) -> None:
    selected = adapter.select("sessions", "orders")
    assert selected.column_names == ["sessions", "orders"]


def test_narwhals_frame_select_all(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
    data_arrow: pa.Table,
) -> None:
    assert adapter.select().equals(data_arrow)


def test_narwhals_frame_select_col_unique(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
) -> None:
    assert set(adapter.select_col_unique("variant")) == {0, 1}


def test_narwhals_frame_group_by(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
) -> None:
    grouped = adapter.group_by("variant")
    assert isinstance(grouped, tea_tasting.backends.narwhals.NarwhalsFrameGroupBy)
    assert grouped.narwhals_frame is adapter
    assert grouped.by == "variant"


def test_narwhals_frame_aggregate(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
    data_arrow: pa.Table,
) -> None:
    aggr = adapter.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions", "orders"),
            cov_cols=(("orders", "sessions"),),
        ),
    )
    _compare_aggrs(aggr, _expected_aggr(data_arrow))


def test_narwhals_frame_aggregate_no_count(
    adapter: tea_tasting.backends.narwhals.NarwhalsFrame,
) -> None:
    aggr = adapter.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=("sessions",),
            var_cols=(),
            cov_cols=(),
        ),
    )
    assert aggr.count_ is None
    assert set(aggr.mean_) == {"sessions"}
    assert aggr.var_ == {}
    assert aggr.cov_ == {}


@pytest.mark.parametrize("cov", ["none", "ungrouped_only"])
def test_narwhals_frame_aggregate_nulls(cov: str) -> None:
    data = pa.table({
        "left": [1.0, 2.0, None, 4.0, 10.0, 20.0, 30.0, None],
        "right": [2.0, 4.0, 100.0, None, 1.0, None, 3.0, 4.0],
    })
    adapter = tea_tasting.backends.narwhals.NarwhalsFrame(
        data,
        cov=cov,  # ty: ignore[invalid-argument-type]
    )
    aggr = adapter.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=(),
            var_cols=("left", "right"),
            cov_cols=(("left", "right"),),
        ),
    )

    valid_left = np.array([1.0, 2.0, 10.0, 30.0])
    valid_right = np.array([2.0, 4.0, 1.0, 3.0])
    assert aggr.count_ == 8
    assert aggr.var_ == pytest.approx({
        "left": np.var([1.0, 2.0, 4.0, 10.0, 20.0, 30.0], ddof=1),
        "right": np.var([2.0, 4.0, 100.0, 1.0, 3.0, 4.0], ddof=1),
    })
    assert aggr.cov_ == pytest.approx({
        ("left", "right"): np.cov(valid_left, valid_right, ddof=1)[0, 1],
    })


def test_narwhals_frame_aggregate_insufficient_covariance_data() -> None:
    data = pa.table({
        "left": [1.0, None],
        "right": [2.0, 3.0],
    })
    aggr = tea_tasting.backends.narwhals.NarwhalsFrame(
        data,
        cov="none",
    ).aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=(),
            var_cols=(),
            cov_cols=(("left", "right"),),
        ),
    )

    assert math.isnan(aggr.cov_["left", "right"])


def test_narwhals_frame_group_by_init(data_narwhals: narwhals.typing.IntoFrame) -> None:
    adapter = tea_tasting.backends.narwhals.NarwhalsFrame(data_narwhals)
    grouped = tea_tasting.backends.narwhals.NarwhalsFrameGroupBy(
        adapter,
        "variant",
    )
    assert grouped.narwhals_frame is adapter
    assert grouped.by == "variant"


def test_narwhals_frame_group_by_aggregate(
    group_adapter: tea_tasting.backends.narwhals.NarwhalsFrameGroupBy,
    data_arrow: pa.Table,
) -> None:
    aggrs = group_adapter.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions", "orders"),
            cov_cols=(("orders", "sessions"),),
        ),
    )
    expected = _expected_aggrs(data_arrow)
    assert set(aggrs) == {0, 1}
    for variant, expected_aggr in expected.items():
        _compare_aggrs(aggrs[variant], expected_aggr)


@pytest.mark.parametrize("cov", ["none", "ungrouped_only"])
def test_narwhals_frame_group_by_aggregate_nulls(cov: str) -> None:
    data = pa.table({
        "group": [0, 0, 0, 0, 1, 1, 1, 1],
        "left": [1.0, 2.0, None, 4.0, 10.0, 20.0, 30.0, None],
        "right": [2.0, 4.0, 100.0, None, 1.0, None, 3.0, 4.0],
    })
    adapter = tea_tasting.backends.narwhals.NarwhalsFrame(
        data,
        cov=cov,  # ty: ignore[invalid-argument-type]
    )
    aggrs = adapter.group_by("group").aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=(),
            var_cols=(),
            cov_cols=(("left", "right"),),
        ),
    )

    assert aggrs[0].count_ == 4
    assert aggrs[0].cov_ == pytest.approx({("left", "right"): 1.0})
    assert aggrs[1].count_ == 4
    assert aggrs[1].cov_ == pytest.approx({("left", "right"): 20.0})
