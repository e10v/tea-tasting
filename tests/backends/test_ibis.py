from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.backends.ibis


if TYPE_CHECKING:
    from collections.abc import Hashable

    import ibis


pytest_plugins = ("tests.fixtures",)


@pytest.fixture
def adapter(data_ibis: ibis.Frame) -> tea_tasting.backends.ibis.IbisTable:
    return tea_tasting.backends.ibis.IbisTable(data_ibis)


@pytest.fixture
def group_adapter(
    adapter: tea_tasting.backends.ibis.IbisTable,
) -> tea_tasting.backends.ibis.IbisTableGroupBy:
    return adapter.group_by("variant")


@pytest.fixture
def data_ibis_null() -> ibis.Frame:
    import ibis  # noqa: PLC0415

    return ibis.connect("duckdb://").create_table(
        "data",
        pa.table({
            "variant": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
            "x": pa.array([1, None, 3, None, 5, None], type=pa.float64()),
            "y": pa.array([2, 4, None, 8, None, None], type=pa.float64()),
            "z": pa.array([10, 20, 30, None, 50, 60], type=pa.float64()),
            "empty": pa.array(
                [None, None, None, None, None, None],
                type=pa.float64(),
            ),
        }),
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
    assert left.mean_ == pytest.approx(right.mean_, nan_ok=True)
    assert left.var_ == pytest.approx(right.var_, nan_ok=True)
    assert left.cov_ == pytest.approx(right.cov_, nan_ok=True)


def _expected_null_aggr() -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=6,
        mean_={
            "x": 3.0,
            "y": 14 / 3,
            "z": 34.0,
            "empty": np.nan,
        },
        var_={
            "x": 4.0,
            "y": 28 / 3,
            "z": 430.0,
            "empty": np.nan,
        },
        cov_={
            ("x", "y"): np.nan,
            ("x", "z"): 40.0,
            ("empty", "x"): np.nan,
        },
    )


def _expected_null_aggrs() -> dict[int, tea_tasting.aggr.Aggregates]:
    return {
        0: tea_tasting.aggr.Aggregates(
            count_=3,
            mean_={
                "x": 2.0,
                "y": 3.0,
                "z": 20.0,
                "empty": np.nan,
            },
            var_={
                "x": 2.0,
                "y": 2.0,
                "z": 100.0,
                "empty": np.nan,
            },
            cov_={
                ("x", "y"): np.nan,
                ("x", "z"): 20.0,
                ("empty", "x"): np.nan,
            },
        ),
        1: tea_tasting.aggr.Aggregates(
            count_=3,
            mean_={
                "x": 5.0,
                "y": 8.0,
                "z": 55.0,
                "empty": np.nan,
            },
            var_={
                "x": np.nan,
                "y": np.nan,
                "z": 50.0,
                "empty": np.nan,
            },
            cov_={
                ("x", "y"): np.nan,
                ("x", "z"): np.nan,
                ("empty", "x"): np.nan,
            },
        ),
    }


def test_ibis_table_init(data_ibis: ibis.Frame) -> None:
    import ibis  # noqa: PLC0415
    import ibis.expr.operations  # noqa: PLC0415

    adapter = tea_tasting.backends.ibis.IbisTable(data_ibis)
    assert adapter.data is data_ibis
    backend = ibis.get_backend(data_ibis)
    assert adapter.has_var is backend.has_operation(ibis.expr.operations.Variance)
    assert adapter.has_cov is backend.has_operation(ibis.expr.operations.Covariance)


def test_ibis_table_init_overrides(data_ibis: ibis.Frame) -> None:
    adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis,
        has_var=False,
        has_cov=True,
    )
    assert adapter.has_var is False
    assert adapter.has_cov is True


def test_ibis_table_select(adapter: tea_tasting.backends.ibis.IbisTable) -> None:
    selected = adapter.select("sessions", "orders")
    assert selected.column_names == ["sessions", "orders"]


def test_ibis_table_select_all(
    adapter: tea_tasting.backends.ibis.IbisTable,
    data_arrow: pa.Table,
) -> None:
    assert adapter.select().equals(data_arrow)


def test_ibis_table_select_col_unique(
    adapter: tea_tasting.backends.ibis.IbisTable,
) -> None:
    assert set(adapter.select_col_unique("variant")) == {0, 1}


def test_ibis_table_group_by(adapter: tea_tasting.backends.ibis.IbisTable) -> None:
    grouped = adapter.group_by("variant")
    assert isinstance(grouped, tea_tasting.backends.ibis.IbisTableGroupBy)
    assert grouped.ibis_table is adapter
    assert grouped.by == "variant"


def test_ibis_table_aggregate(
    adapter: tea_tasting.backends.ibis.IbisTable,
    data_arrow: pa.Table,
) -> None:
    aggr = adapter.aggregate(
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("sessions", "orders"),
        cov_cols=(("orders", "sessions"),),
    )
    _compare_aggrs(aggr, _expected_aggr(data_arrow))


def test_ibis_table_aggregate_no_count(
    adapter: tea_tasting.backends.ibis.IbisTable,
) -> None:
    aggr = adapter.aggregate(
        has_count=False,
        mean_cols=("sessions",),
        var_cols=(),
        cov_cols=(),
    )
    assert aggr.count_ is None
    assert set(aggr.mean_) == {"sessions"}
    assert aggr.var_ == {}
    assert aggr.cov_ == {}


def test_ibis_table_group_by_init(
    data_ibis: ibis.Frame,
) -> None:
    adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis,
        has_var=True,
        has_cov=False,
    )
    grouped = tea_tasting.backends.ibis.IbisTableGroupBy(
        adapter,
        "variant",
    )
    assert grouped.ibis_table is adapter
    assert grouped.by == "variant"


def test_ibis_table_group_by_aggregate(
    group_adapter: tea_tasting.backends.ibis.IbisTableGroupBy,
    data_arrow: pa.Table,
) -> None:
    aggrs = group_adapter.aggregate(
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("sessions", "orders"),
        cov_cols=(("orders", "sessions"),),
    )
    expected = _expected_aggrs(data_arrow)
    assert set(aggrs) == {0, 1}
    for variant, expected_aggr in expected.items():
        _compare_aggrs(aggrs[variant], expected_aggr)


@pytest.mark.parametrize(
    ("has_var", "has_cov"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_ibis_table_aggregate_nulls(
    data_ibis_null: ibis.Frame,
    has_var: bool,  # noqa: FBT001
    has_cov: bool,  # noqa: FBT001
) -> None:
    adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis_null,
        has_var=has_var,
        has_cov=has_cov,
    )

    aggr = adapter.aggregate(
        has_count=True,
        mean_cols=("x", "y", "z", "empty"),
        var_cols=("x", "y", "z", "empty"),
        cov_cols=(("x", "y"), ("x", "z"), ("empty", "x")),
    )

    _compare_aggrs(aggr, _expected_null_aggr())


def test_ibis_table_aggregate_fallback_is_numerically_stable() -> None:
    import ibis  # noqa: PLC0415

    data = pa.table({
        "x": pa.array(
            [1e12 + 1, 1e12 + 2, 1e12 + 3, 1e12 + 4],
            type=pa.float64(),
        ),
        "y": pa.array(
            [2e12 + 3, 2e12 + 5, 2e12 + 7, 2e12 + 9],
            type=pa.float64(),
        ),
    })
    data_ibis = ibis.connect("duckdb://").create_table("data", data)
    adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis,
        has_var=False,
        has_cov=False,
    )

    aggr = adapter.aggregate(
        has_count=False,
        mean_cols=(),
        var_cols=("x",),
        cov_cols=(("x", "y"),),
    )

    assert aggr.var_ == pytest.approx({"x": 5 / 3})
    assert aggr.cov_ == pytest.approx({("x", "y"): 10 / 3})


@pytest.mark.parametrize(
    ("has_var", "has_cov"),
    [
        (True, False),
        (False, True),
    ],
)
def test_ibis_table_aggregate_mixed_fallback_with_overlapping_cols(
    data_ibis_duckdb: ibis.Frame,
    data_arrow: pa.Table,
    has_var: bool,  # noqa: FBT001
    has_cov: bool,  # noqa: FBT001
) -> None:
    adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis_duckdb,
        has_var=has_var,
        has_cov=has_cov,
    )

    aggr = adapter.aggregate(
        has_count=False,
        mean_cols=("sessions",),
        var_cols=("sessions",),
        cov_cols=(("orders", "sessions"),),
    )

    expected = tea_tasting.aggr.Aggregates(
        count_=None,
        mean_={"sessions": pc.mean(data_arrow["sessions"]).as_py()},
        var_={"sessions": pc.variance(data_arrow["sessions"], ddof=1).as_py()},
        cov_={
            ("orders", "sessions"): np.cov(
                data_arrow["sessions"].combine_chunks().to_numpy(
                    zero_copy_only=False,
                ),
                data_arrow["orders"].combine_chunks().to_numpy(
                    zero_copy_only=False,
                ),
                ddof=1,
            )[0, 1],
        },
    )
    _compare_aggrs(aggr, expected)


@pytest.mark.parametrize(
    ("has_var", "has_cov"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_ibis_table_group_by_aggregate_nulls(
    data_ibis_null: ibis.Frame,
    has_var: bool,  # noqa: FBT001
    has_cov: bool,  # noqa: FBT001
) -> None:
    group_adapter = tea_tasting.backends.ibis.IbisTable(
        data_ibis_null,
        has_var=has_var,
        has_cov=has_cov,
    ).group_by("variant")

    aggrs = group_adapter.aggregate(
        has_count=True,
        mean_cols=("x", "y", "z", "empty"),
        var_cols=("x", "y", "z", "empty"),
        cov_cols=(("x", "y"), ("x", "z"), ("empty", "x")),
    )

    expected = _expected_null_aggrs()
    assert set(aggrs) == {0, 1}
    for variant, expected_aggr in expected.items():
        _compare_aggrs(aggrs[variant], expected_aggr)
