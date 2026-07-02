from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.backends.narwhals


if TYPE_CHECKING:
    from collections.abc import Hashable

    import narwhals.typing


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
    assert adapter.data is data_narwhals


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
