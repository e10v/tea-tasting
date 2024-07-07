from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple
import unittest.mock

import pandas as pd
import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.metrics.base


if TYPE_CHECKING:
    from typing import Literal

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
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42, to_ibis=True)

@pytest.fixture
def aggr_cols() -> tea_tasting.metrics.base.AggrCols:
    return tea_tasting.metrics.base.AggrCols(
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("orders", "revenue"),
        cov_cols=(("sessions", "revenue"),),
    )

@pytest.fixture
def correct_aggrs(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> dict[Any, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data,
        group_col="variant",
        **aggr_cols._asdict(),
    )

@pytest.fixture
def correct_aggr(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.read_aggregates(
        data,
        group_col=None,
        **aggr_cols._asdict(),
    )

@pytest.fixture
def cols() -> tuple[str, ...]:
    return ("sessions", "orders", "revenue")

@pytest.fixture
def correct_dfs(
    data: ibis.expr.types.Table,
    cols: tuple[str, ...],
) -> dict[Any, pd.DataFrame]:
    return dict(tuple(data.select(*cols, "variant").to_pandas().groupby("variant")))

@pytest.fixture
def aggr_metric(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]:
    class AggrMetric(tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]]):
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

@pytest.fixture
def aggr_power(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.metrics.base.PowerBaseAggregated[
    tea_tasting.metrics.base.MetricPowerResults[dict[str, Any]]
]:
    class AggrPower(
        tea_tasting.metrics.base.PowerBaseAggregated[
            tea_tasting.metrics.base.MetricPowerResults[dict[str, Any]]
        ],
    ):
        @property
        def aggr_cols(self) -> tea_tasting.metrics.base.AggrCols:
            return aggr_cols

        def solve_power_from_aggregates(
            self,
            data: tea_tasting.aggr.Aggregates,  # noqa: ARG002
            parameter: Literal[  # noqa: ARG002
                "power",
                "effect_size",
                "rel_effect_size",
                "n_obs",
            ] = "power",
        ) -> tea_tasting.metrics.base.MetricPowerResults[dict[str, Any]]:
            return tea_tasting.metrics.base.MetricPowerResults()
    return AggrPower()

@pytest.fixture
def gran_metric(
    cols: tuple[str, ...],
) -> tea_tasting.metrics.base.MetricBaseGranular[dict[str, Any]]:
    class GranMetric(tea_tasting.metrics.base.MetricBaseGranular[dict[str, Any]]):
        @property
        def cols(self) -> tuple[str, ...]:
            return cols

        def analyze_dataframes(
            self,
            control: pd.DataFrame,  # noqa: ARG002
            treatment: pd.DataFrame,  # noqa: ARG002
        ) -> dict[str, Any]:
            return {}

    return GranMetric()


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == right.count_
    assert left.mean_ == right.mean_
    assert left.var_ == right.var_
    assert left.cov_ == right.cov_


def test_metric_power_results_to_dicts():
    result0 = {
        "power": 0.8,
        "effect_size": 1,
        "rel_effect_size": 0.05,
        "n_obs": 10_000,
    }
    result1 = {
        "power": 0.9,
        "effect_size": 2,
        "rel_effect_size": 0.1,
        "n_obs": 20_000,
    }

    results = tea_tasting.metrics.base.MetricPowerResults[dict[str, Any]](
        [result0, result1])
    assert results.to_dicts() == (result0, result1)

    class PowerResult(NamedTuple):
        power: float
        effect_size: float
        rel_effect_size: float
        n_obs: float
    results = tea_tasting.metrics.base.MetricPowerResults[PowerResult]([
        PowerResult(**result0),
        PowerResult(**result1),
    ])
    assert results.to_dicts() == (result0, result1)


def test_metric_base_aggregated_analyze_table(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(data, control=0, treatment=1, variant="variant")
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
    aggr_metric.analyze(data.to_pandas(), control=0, treatment=1, variant="variant")
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


def test_power_base_aggregated_analyze_table(
    aggr_power: tea_tasting.metrics.base.PowerBaseAggregated[Any],
    data: ibis.expr.types.Table,
    correct_aggr: tea_tasting.aggr.Aggregates,
):
    aggr_power.solve_power_from_aggregates = unittest.mock.MagicMock()
    aggr_power.solve_power(data, "effect_size")
    aggr_power.solve_power_from_aggregates.assert_called_once()
    kwargs = aggr_power.solve_power_from_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["data"], correct_aggr)
    assert kwargs["parameter"] == "effect_size"

def test_power_base_aggregated_analyze_df(
    aggr_power: tea_tasting.metrics.base.PowerBaseAggregated[Any],
    data: ibis.expr.types.Table,
    correct_aggr: tea_tasting.aggr.Aggregates,
):
    aggr_power.solve_power_from_aggregates = unittest.mock.MagicMock()
    aggr_power.solve_power(data.to_pandas(), "n_obs")
    aggr_power.solve_power_from_aggregates.assert_called_once()
    kwargs = aggr_power.solve_power_from_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["data"], correct_aggr)
    assert kwargs["parameter"] == "n_obs"

def test_power_base_aggregated_analyze_aggr(
    aggr_power: tea_tasting.metrics.base.PowerBaseAggregated[Any],
    correct_aggr: tea_tasting.aggr.Aggregates,
):
    aggr_power.solve_power_from_aggregates = unittest.mock.MagicMock()
    aggr_power.solve_power(correct_aggr, "rel_effect_size")
    aggr_power.solve_power_from_aggregates.assert_called_once()
    kwargs = aggr_power.solve_power_from_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["data"], correct_aggr)
    assert kwargs["parameter"] == "rel_effect_size"


def test_aggregate_by_variants_table(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[Any, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        data,
        aggr_cols=aggr_cols,
        variant="variant",
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
        variant="variant",
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
        variant="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_raises(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
):
    with pytest.raises(ValueError, match="variant"):
        tea_tasting.metrics.base.aggregate_by_variants(data, aggr_cols=aggr_cols)

    with pytest.raises(TypeError):
        tea_tasting.metrics.base.aggregate_by_variants(
            1, aggr_cols=aggr_cols, variant="variant")  # type: ignore


def test_metric_base_granular_table(
    gran_metric: tea_tasting.metrics.base.MetricBaseGranular[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_dfs: dict[Any, pd.DataFrame],
):
    gran_metric.analyze_dataframes = unittest.mock.MagicMock()
    gran_metric.analyze(data, control=0, treatment=1, variant="variant")
    gran_metric.analyze_dataframes.assert_called_once()
    kwargs = gran_metric.analyze_dataframes.call_args.kwargs
    pd.testing.assert_frame_equal(kwargs["control"], correct_dfs[0])
    pd.testing.assert_frame_equal(kwargs["treatment"], correct_dfs[1])

def test_metric_base_granular_df(
    gran_metric: tea_tasting.metrics.base.MetricBaseGranular[dict[str, Any]],
    data: ibis.expr.types.Table,
    correct_dfs: dict[Any, pd.DataFrame],
):
    gran_metric.analyze_dataframes = unittest.mock.MagicMock()
    gran_metric.analyze(data.to_pandas(), control=0, treatment=1, variant="variant")
    gran_metric.analyze_dataframes.assert_called_once()
    kwargs = gran_metric.analyze_dataframes.call_args.kwargs
    pd.testing.assert_frame_equal(kwargs["control"], correct_dfs[0])
    pd.testing.assert_frame_equal(kwargs["treatment"], correct_dfs[1])

def test_metric_base_granular_dfs(
    gran_metric: tea_tasting.metrics.base.MetricBaseGranular[dict[str, Any]],
    correct_dfs: dict[Any, pd.DataFrame],
):
    gran_metric.analyze_dataframes = unittest.mock.MagicMock()
    gran_metric.analyze(correct_dfs, control=0, treatment=1)
    gran_metric.analyze_dataframes.assert_called_once()
    kwargs = gran_metric.analyze_dataframes.call_args.kwargs
    pd.testing.assert_frame_equal(kwargs["control"], correct_dfs[0])
    pd.testing.assert_frame_equal(kwargs["treatment"], correct_dfs[1])


def test_read_dataframes_table(
    data: ibis.expr.types.Table,
    cols: tuple[str, ...],
    correct_dfs: dict[Any, pd.DataFrame],
):
    dfs = tea_tasting.metrics.base.read_dataframes(
        data,
        cols=cols,
        variant="variant",
    )
    pd.testing.assert_frame_equal(dfs[0], correct_dfs[0])
    pd.testing.assert_frame_equal(dfs[1], correct_dfs[1])

def test_read_dataframes_df(
    data: ibis.expr.types.Table,
    cols: tuple[str, ...],
    correct_dfs: dict[Any, pd.DataFrame],
):
    dfs = tea_tasting.metrics.base.read_dataframes(
        data.to_pandas(),
        cols=cols,
        variant="variant",
    )
    pd.testing.assert_frame_equal(dfs[0], correct_dfs[0])
    pd.testing.assert_frame_equal(dfs[1], correct_dfs[1])

def test_read_dataframes_dfs(
    cols: tuple[str, ...],
    correct_dfs: dict[Any, pd.DataFrame],
):
    dfs = tea_tasting.metrics.base.read_dataframes(
        correct_dfs,
        cols=cols,
        variant="variant",
    )
    pd.testing.assert_frame_equal(dfs[0], correct_dfs[0])
    pd.testing.assert_frame_equal(dfs[1], correct_dfs[1])

def test_read_dataframes_raises(
    data: ibis.expr.types.Table,
    cols: tuple[str, ...],
):
    with pytest.raises(ValueError, match="variant"):
        tea_tasting.metrics.base.read_dataframes(data, cols=cols)

    with pytest.raises(TypeError):
        tea_tasting.metrics.base.read_dataframes(1, cols=cols, variant="variant")  # type: ignore
