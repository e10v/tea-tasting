from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import unittest.mock

import ibis
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.metrics.base


if TYPE_CHECKING:
    from typing import Any, Literal

    import ibis.expr.types  # noqa: TC004
    import pandas as pd


    Frame = ibis.expr.types.Table | pa.Table | pd.DataFrame | pl.LazyFrame


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
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)

@pytest.fixture
def data_pandas(data_arrow: pa.Table) -> pd.DataFrame:
    return data_arrow.to_pandas()

@pytest.fixture
def data_polars(data_arrow: pa.Table) -> pl.DataFrame:
    return pl.from_arrow(data_arrow)  # type: ignore

@pytest.fixture
def data_polars_lazy(data_polars: pl.DataFrame) -> pl.LazyFrame:
    return data_polars.lazy()

@pytest.fixture
def data_duckdb(data_arrow: pa.Table) -> ibis.expr.types.Table:
    return ibis.connect("duckdb://").create_table("data", data_arrow)

@pytest.fixture
def data_sqlite(data_arrow: pa.Table) -> ibis.expr.types.Table:
    return ibis.connect("sqlite://").create_table("data", data_arrow)

@pytest.fixture(params=[
    "data_arrow", "data_pandas",
    "data_polars", "data_polars_lazy",
    "data_duckdb", "data_sqlite",
])
def data(request: pytest.FixtureRequest) -> Frame:
    return request.getfixturevalue(request.param)


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
    data_arrow: pa.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> dict[object, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
        group_col="variant",
        **aggr_cols._asdict(),
    )

@pytest.fixture
def correct_aggr(
    data_arrow: pa.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
        group_col=None,
        **aggr_cols._asdict(),
    )

@pytest.fixture
def cols() -> tuple[str, ...]:
    return ("sessions", "orders", "revenue")

@pytest.fixture
def correct_gran(
    data_arrow: pa.Table,
    cols: tuple[str, ...],
) -> dict[object, pa.Table]:
    variant_col = data_arrow["variant"]
    table = data_arrow.select(cols)
    return {
        var: table.filter(pc.equal(variant_col, pa.scalar(var)))  # type: ignore
        for var in variant_col.unique().to_pylist()
    }

@pytest.fixture
def aggr_metric(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.metrics.base.MetricBaseAggregated[dict[str, object]]:
    class AggrMetric(tea_tasting.metrics.base.MetricBaseAggregated[dict[str, object]]):
        @property
        def aggr_cols(self) -> tea_tasting.metrics.base.AggrCols:
            return aggr_cols

        def analyze_aggregates(
            self,
            control: tea_tasting.aggr.Aggregates,  # noqa: ARG002
            treatment: tea_tasting.aggr.Aggregates,  # noqa: ARG002
        ) -> dict[str, object]:
            return {}

    return AggrMetric()

@pytest.fixture
def aggr_power(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
) -> tea_tasting.metrics.base.PowerBaseAggregated[
    tea_tasting.metrics.base.MetricPowerResults[dict[str, object]]
]:
    class AggrPower(
        tea_tasting.metrics.base.PowerBaseAggregated[
            tea_tasting.metrics.base.MetricPowerResults[dict[str, object]]
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
        ) -> tea_tasting.metrics.base.MetricPowerResults[dict[str, object]]:
            return tea_tasting.metrics.base.MetricPowerResults()
    return AggrPower()

@pytest.fixture
def gran_metric(
    cols: tuple[str, ...],
) -> tea_tasting.metrics.base.MetricBaseGranular[dict[str, object]]:
    class GranMetric(tea_tasting.metrics.base.MetricBaseGranular[dict[str, object]]):
        @property
        def cols(self) -> tuple[str, ...]:
            return cols

        def analyze_granular(
            self,
            control: pa.Table,  # noqa: ARG002
            treatment: pa.Table,  # noqa: ARG002
        ) -> dict[str, object]:
            return {}

    return GranMetric()


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == right.count_
    assert left.mean_ == pytest.approx(right.mean_)
    assert left.var_ == pytest.approx(right.var_)
    assert left.cov_ == pytest.approx(right.cov_)


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

    results = tea_tasting.metrics.base.MetricPowerResults[dict[str, float | int]](  # type: ignore
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


def test_metric_base_aggregated_analyze_frame(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, object]],
    data_arrow: pa.Table,
    correct_aggrs: dict[object, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(data_arrow, control=0, treatment=1, variant="variant")
    aggr_metric.analyze_aggregates.assert_called_once()
    kwargs = aggr_metric.analyze_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["control"], correct_aggrs[0])
    _compare_aggrs(kwargs["treatment"], correct_aggrs[1])

def test_metric_base_aggregated_analyze_aggrs(
    aggr_metric: tea_tasting.metrics.base.MetricBaseAggregated[dict[str, object]],
    correct_aggrs: dict[object, tea_tasting.aggr.Aggregates],
):
    aggr_metric.analyze_aggregates = unittest.mock.MagicMock()
    aggr_metric.analyze(correct_aggrs, control=0, treatment=1)
    aggr_metric.analyze_aggregates.assert_called_once()
    kwargs = aggr_metric.analyze_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["control"], correct_aggrs[0])
    _compare_aggrs(kwargs["treatment"], correct_aggrs[1])


def test_power_base_aggregated_analyze_frame(
    aggr_power: tea_tasting.metrics.base.PowerBaseAggregated[Any],
    data_arrow: pa.Table,
    correct_aggr: tea_tasting.aggr.Aggregates,
):
    aggr_power.solve_power_from_aggregates = unittest.mock.MagicMock()
    aggr_power.solve_power(data_arrow, "effect_size")
    aggr_power.solve_power_from_aggregates.assert_called_once()
    kwargs = aggr_power.solve_power_from_aggregates.call_args.kwargs
    _compare_aggrs(kwargs["data"], correct_aggr)
    assert kwargs["parameter"] == "effect_size"

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


def test_aggregate_by_variants_frame(
    data_arrow: pa.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[object, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        data_arrow,
        aggr_cols=aggr_cols,
        variant="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_aggrs(
    aggr_cols: tea_tasting.metrics.base.AggrCols,
    correct_aggrs: dict[object, tea_tasting.aggr.Aggregates],
):
    aggrs = tea_tasting.metrics.base.aggregate_by_variants(
        correct_aggrs,
        aggr_cols=aggr_cols,
        variant="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_aggregate_by_variants_raises(
    data_arrow: pa.Table,
    aggr_cols: tea_tasting.metrics.base.AggrCols,
):
    with pytest.raises(ValueError, match="variant"):
        tea_tasting.metrics.base.aggregate_by_variants(data_arrow, aggr_cols=aggr_cols)


def test_metric_base_granular_frame(
    gran_metric: tea_tasting.metrics.base.MetricBaseGranular[dict[str, object]],
    data_arrow: pa.Table,
    correct_gran: dict[object, pa.Table],
):
    gran_metric.analyze_granular = unittest.mock.MagicMock()
    gran_metric.analyze(data_arrow, control=0, treatment=1, variant="variant")
    gran_metric.analyze_granular.assert_called_once()
    kwargs = gran_metric.analyze_granular.call_args.kwargs
    assert kwargs["control"].equals(correct_gran[0])
    assert kwargs["treatment"].equals(correct_gran[1])

def test_metric_base_granular_gran(
    gran_metric: tea_tasting.metrics.base.MetricBaseGranular[dict[str, object]],
    correct_gran: dict[object, pa.Table],
):
    gran_metric.analyze_granular = unittest.mock.MagicMock()
    gran_metric.analyze(correct_gran, control=0, treatment=1)
    gran_metric.analyze_granular.assert_called_once()
    kwargs = gran_metric.analyze_granular.call_args.kwargs
    assert kwargs["control"].equals(correct_gran[0])
    assert kwargs["treatment"].equals(correct_gran[1])


def test_read_granular_frame(
    data: Frame,
    cols: tuple[str, ...],
    correct_gran: dict[object, pa.Table],
):
    gran = tea_tasting.metrics.base.read_granular(
        data,
        cols=cols,
        variant="variant",
    )
    assert gran[0].equals(correct_gran[0])
    assert gran[1].equals(correct_gran[1])

def test_read_granular_dict(
    cols: tuple[str, ...],
    correct_gran: dict[object, pa.Table],
):
    gran = tea_tasting.metrics.base.read_granular(
        correct_gran,
        cols=cols,
        variant="variant",
    )
    assert gran[0].equals(correct_gran[0])
    assert gran[1].equals(correct_gran[1])

def test_read_granular_raises(
    data_arrow: pa.Table,
    cols: tuple[str, ...],
):
    with pytest.raises(ValueError, match="variant"):
        tea_tasting.metrics.base.read_granular(data_arrow, cols=cols)
