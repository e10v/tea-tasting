from __future__ import annotations

from typing import TYPE_CHECKING, Any
import unittest.mock

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
def aggr_cols() -> tea_tasting.metrics.base.AggrCols:
    return tea_tasting.metrics.base.AggrCols(
        has_count=True,
        mean_cols=("visits", "orders"),
        var_cols=("orders", "revenue"),
        cov_cols=(("visits", "revenue"),),
    )

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
def aggr_metric(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]:
    class AggrMetric(tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]):
        def __init__(self) -> None:
            return None

        @property
        def aggr_cols(self) -> tea_tasting.metrics.base.AggrCols:
            return aggr_cols

        def analyze_aggregates(
            self,
            control: tea_tasting.aggr.Aggregates,  # noqa: ARG002
            treatment: tea_tasting.aggr.Aggregates,  # noqa: ARG002
        ) -> dict[str, Any]:
            return {}

    return AggrMetric()


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == right.count_
    assert left.mean_ == right.mean_
    assert left.var_ == right.var_
    assert left.cov_ == right.cov_


def test_metric_base_aggregated_analyze_table(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(data, control=0, treatment=1, variant_col="variant")
    aggr_metric.analyze_aggregates.assert_called_once()
    kwargs = aggr_metric.analyze_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["control"], correct_aggrs[0])
    _compare_aggrs(kwargs["treatment"], correct_aggrs[1])

def test_metric_base_aggregated_analyze_df(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(data.to_pandas(), control=0, treatment=1, variant_col="variant")
    aggr_metric.analyze_aggregates.assert_called_once()
    kwargs = aggr_metric.analyze_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["control"], correct_aggrs[0])
    _compare_aggrs(kwargs["treatment"], correct_aggrs[1])

def test_metric_base_aggregated_analyze_aggrs(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(correct_aggrs, control=0, treatment=1)
    aggr_metric.analyze_aggregates.assert_called_once()
    kwargs = aggr_metric.analyze_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["control"], correct_aggrs[0])
    _compare_aggrs(kwargs["treatment"], correct_aggrs[1])


def test_aggregate_by_variants_table(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        data,
        aggr_cols=aggr_cols,
        variant_col="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_df(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        data.to_pandas(),
        aggr_cols=aggr_cols,
        variant_col="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_aggrs(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        correct_aggrs,
        aggr_cols=aggr_cols,
        variant_col="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_raises(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
):
    with pytest.raises(ValueError, match="variant_col"):
        tea_tasting.metrics.base.aggregate_by_variants(data, aggr_cols=aggr_cols)

    with pytest.raises(TypeError):
        tea_tasting.metrics.base.aggregate_by_variants(1, aggr_cols=aggr_cols)  # type: ignore
