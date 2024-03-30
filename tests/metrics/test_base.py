from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.metrics.base


if TYPE_CHECKING:
    import ibis.expr.types


def test_aggr_cols_or():
    aggr_cols0 = tea_tasting.metrics.base.AggrCols(
        has_count=False,
        mean_cols=("a", "b"),
        var_cols=("b", "c"),
        cov_cols=(("a", "b"), ("c", "b")),
    )

    aggr_cols1 = tea_tasting.metrics.base.AggrCols(
        has_count=True,
        mean_cols=("b", "c"),
        var_cols=("c", "d"),
        cov_cols=(("b", "c"), ("d", "c")),
    )

    aggr_cols = aggr_cols0 | aggr_cols1

    assert isinstance(aggr_cols, tea_tasting.metrics.base.AggrCols)
    assert aggr_cols.has_count is True
    assert set(aggr_cols.mean_cols) == {"a", "b", "c"}
    assert len(aggr_cols.mean_cols) == 3
    assert set(aggr_cols.var_cols) == {"b", "c", "d"}
    assert len(aggr_cols.var_cols) == 3
    assert set(aggr_cols.cov_cols) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert len(aggr_cols.cov_cols) == 3


def test_aggr_cols_len():
    assert len(tea_tasting.metrics.base.AggrCols(
        has_count=False,
        mean_cols=("a", "b"),
        var_cols=("b", "c"),
        cov_cols=(("a", "b"), ("c", "b")),
    )) == 6
    assert len(tea_tasting.metrics.base.AggrCols(
        has_count=True,
        mean_cols=("b", "c"),
        var_cols=("c", "d"),
        cov_cols=(("b", "c"), ("d", "c")),
    )) == 7


@pytest.fixture
def data() -> ibis.expr.types.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)

@pytest.fixture
def correct_aggrs(
    data: ibis.expr.types.Table,
) -> dict[Any, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data,
        group_col="variant",
        has_count=True,
        mean_cols=("visits", "orders"),
        var_cols=("orders", "revenue"),
        cov_cols=(("visits", "revenue"),),
    )

@pytest.fixture
def aggr_metric() -> tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]:
    class AggrMetric(tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]):
        def __init__(self) -> None:
            return None

        @property
        def aggr_cols(
            self,
        ) -> tea_tasting.metrics.base.AggrCols:
            return tea_tasting.metrics.base.AggrCols(
                has_count=True,
                mean_cols=("visits", "orders"),
                var_cols=("orders", "revenue"),
                cov_cols=(("visits", "revenue"),),
            )

        def analyze_aggregates(
            self,
            data: dict[Any, tea_tasting.aggr.Aggregates],  # noqa: ARG002
            control: Any,  # noqa: ARG002
            treatment: Any,  # noqa: ARG002
        ) -> dict[str, Any]:
            return {}

    return AggrMetric()


def _compare_aggrs(
    left: dict[Any, tea_tasting.aggr.Aggregates],
    right: dict[Any, tea_tasting.aggr.Aggregates],
) -> None:
    assert left.keys() == right.keys()
    for variant in left:
        l = left[variant]  # noqa: E741
        r = right[variant]
        assert l.count_ == r.count_
        assert l.mean_ == r.mean_
        assert l.var_ == r.var_
        assert l.cov_ == r.cov_


def test_metric_base_aggregated_aggregate_by_variants_table(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = aggr_metric.aggregate_by_variants(data, variant_col="variant")
    _compare_aggrs(aggrs, correct_aggrs)

def test_metric_base_aggregated_aggregate_by_variants_df(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = aggr_metric.aggregate_by_variants(data.to_pandas(), variant_col="variant")
    _compare_aggrs(aggrs, correct_aggrs)

def test_metric_base_aggregated_aggregate_by_variants_aggrs(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = aggr_metric.aggregate_by_variants(correct_aggrs)
    _compare_aggrs(aggrs, correct_aggrs)

def test_metric_base_aggregated_aggregate_by_variants_raises(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
):
    with pytest.raises(ValueError, match="variant_col"):
        aggr_metric.aggregate_by_variants(data)  # type: ignore

    with pytest.raises(TypeError):
        aggr_metric.aggregate_by_variants(1)  # type: ignore
