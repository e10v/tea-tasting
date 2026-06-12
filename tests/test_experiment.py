from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import ibis.expr.types
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.experiment
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from typing import Any, Literal

    import ibis
    import narwhals.typing
    import numpy as np

    from tests.fixtures import Frame


pytest_plugins = ("tests.fixtures",)


class _MetricResultTuple(NamedTuple):
    control: float
    treatment: float
    effect_size: float

class _MetricResultDict(TypedDict):
    control: float
    treatment: float
    effect_size: float

class _PowerResult(NamedTuple):
    power: float
    effect_size: float
    rel_effect_size: float
    n_obs: float


class _Metric(
    tea_tasting.metrics.MetricBase[_MetricResultTuple],
    tea_tasting.metrics.PowerBase[tea_tasting.metrics.MetricPowerResults[_PowerResult]],
):
    def __init__(self, column: str) -> None:
        self.column = column

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table | dict[
            Hashable, tea_tasting.aggr.Aggregates],
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> _MetricResultTuple:
        if not isinstance(data, dict):
            data = tea_tasting.aggr.read_aggregates(
                data,
                variant,
                has_count=False,
                mean_cols=(self.column,),
                var_cols=(),
                cov_cols=(),
            )
        return _MetricResultTuple(
            control=data[control].mean(self.column),  # ty:ignore[invalid-argument-type]
            treatment=data[treatment].mean(self.column),  # ty:ignore[invalid-argument-type]
            effect_size=data[treatment].mean(self.column) -  # ty:ignore[invalid-argument-type]
                data[control].mean(self.column),  # ty:ignore[invalid-argument-type]
        )

    def solve_power(
        self,
        data: narwhals.typing.IntoFrame | ibis.Table,  # noqa: ARG002
        parameter: Literal[  # noqa: ARG002
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> tea_tasting.metrics.MetricPowerResults[_PowerResult]:
        return tea_tasting.metrics.MetricPowerResults((
            _PowerResult(power=0.8, effect_size=1, rel_effect_size=0.05, n_obs=10_000),
            _PowerResult(power=0.9, effect_size=2, rel_effect_size=0.1, n_obs=20_000),
        ))


class _MetricAggregated(
    tea_tasting.metrics.MetricBaseAggregated[_MetricResultTuple],
    tea_tasting.metrics.PowerBaseAggregated[
        tea_tasting.metrics.MetricPowerResults[dict[str, object]]],
):
    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def aggr_cols(self) -> tea_tasting.metrics.AggrCols:
        return tea_tasting.metrics.AggrCols(mean_cols=(self.column,))

    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> _MetricResultTuple:
        contr_mean = control.mean(self.column)
        treat_mean = treatment.mean(self.column)
        return _MetricResultTuple(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=treat_mean - contr_mean,
        )

    def solve_power_from_aggregates(
        self,
        data: tea_tasting.aggr.Aggregates,  # noqa: ARG002
        parameter: Literal[  # noqa: ARG002
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> tea_tasting.metrics.MetricPowerResults[dict[str, Any]]:
        return tea_tasting.metrics.MetricPowerResults((
            {"power": 0.8, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 10_000},
            {"power": 0.9, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 20_000},
        ))


class _MetricGranular(tea_tasting.metrics.MetricBaseGranular[_MetricResultDict]):
    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def cols(self) -> tuple[str, ...]:
        return (self.column,)

    def analyze_granular(
        self,
        control: pa.Table,
        treatment: pa.Table,
    ) -> _MetricResultDict:
        contr_mean = pc.mean(control[self.column]).as_py()
        treat_mean = pc.mean(treatment[self.column]).as_py()
        return _MetricResultDict(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=treat_mean - contr_mean,
        )


@pytest.fixture
def get_result() -> Callable[
    [tuple[float, float, float], tuple[float, float, float]],
    tea_tasting.experiment.ExperimentResult,
]:
    def _get_result(
        x: tuple[float, float, float],
        y: tuple[float, float, float],
    ) -> tea_tasting.experiment.ExperimentResult:
        return tea_tasting.experiment.ExperimentResult(
            metric_tuple=_MetricResultTuple(*x),
            metric_dict=_MetricResultDict(
                control=y[0],
                treatment=y[1],
                effect_size=y[2],
            ),
        )

    return _get_result


@pytest.fixture
def result(
    get_result: Callable[
        [tuple[float, float, float], tuple[float, float, float]],
        tea_tasting.experiment.ExperimentResult,
    ],
) -> tea_tasting.experiment.ExperimentResult:
    return get_result((10, 11, 1), (20, 22, 2))


@pytest.fixture
def result2() -> tea_tasting.experiment.ExperimentResult:
    class MetricResultTuple(NamedTuple):
        control: float
        treatment: float
        rel_effect_size: float
        rel_effect_size_ci_lower: float
        rel_effect_size_ci_upper: float
        pvalue: float

    return tea_tasting.experiment.ExperimentResult(
        metric_tuple=MetricResultTuple(
            control=4.4444,
            treatment=5.5555,
            rel_effect_size=0.2,
            rel_effect_size_ci_lower=0.12345,
            rel_effect_size_ci_upper=float("inf"),
            pvalue=0.23456,
        ),
        metric_dict={
            "control": 9.9999,
            "treatment": 11.111,
            "rel_effect_size": 0.11111,
            "rel_effect_size_ci_lower": 0,
        },
    )


@pytest.fixture
def results(
    result: tea_tasting.experiment.ExperimentResult,
) -> tea_tasting.experiment.ExperimentResults:
    return tea_tasting.experiment.ExperimentResults({(0, 1): result})


@pytest.fixture
def results2(
    get_result: Callable[
        [tuple[float, float, float], tuple[float, float, float]],
        tea_tasting.experiment.ExperimentResult,
    ],
) -> tea_tasting.experiment.ExperimentResults:
    return tea_tasting.experiment.ExperimentResults({
        (0, 1): get_result((10, 11, 1), (20, 22, 2)),
        (0, 2): get_result((10, 11, 1), (30, 33, 3)),
    })


@pytest.fixture
def data_arrow_multi(data_arrow: pa.Table) -> pa.Table:
    data2 = data_arrow.filter(pc.equal(data_arrow["variant"], pa.scalar(1)))
    return pa.concat_tables((
        data_arrow,
        data2.set_column(
            data_arrow.schema.get_field_index("variant"),
            "variant",
            pa.array([2] * data2.num_rows),
        ),
    ))


@pytest.fixture
def data_aggr(data_arrow: pa.Table) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
        group_col="variant",
        has_count=False,
        mean_cols=("sessions", "orders"),
        var_cols=(),
        cov_cols=(),
    )


@pytest.fixture
def data_aggr_multi(
    data_arrow_multi: pa.Table,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data_arrow_multi,
        group_col="variant",
        has_count=False,
        mean_cols=("sessions", "orders"),
        var_cols=(),
        cov_cols=(),
    )


@pytest.fixture
def ref_result(
    data_arrow: pa.Table,
) -> tea_tasting.experiment.ExperimentResult:
    sessions = _Metric("sessions")
    orders = _MetricAggregated("orders")
    revenue = _MetricGranular("revenue")
    return tea_tasting.experiment.ExperimentResult(
        avg_sessions=sessions.analyze(data_arrow, 0, 1, "variant"),
        avg_orders=orders.analyze(data_arrow, 0, 1, "variant"),
        avg_revenue=revenue.analyze(data_arrow, 0, 1, "variant"),
    )


def test_experiment_result_to_dicts(
    result: tea_tasting.experiment.ExperimentResult,
) -> None:
    assert result.to_dicts() == (
        {"metric": "metric_tuple", "control": 10, "treatment": 11, "effect_size": 1},
        {"metric": "metric_dict", "control": 20, "treatment": 22, "effect_size": 2},
    )


def test_experiment_results_to_dicts(
    results2: tea_tasting.experiment.ExperimentResults,
) -> None:
    assert results2.to_dicts() == (
        {
            "variants": "(0, 1)",
            "metric": "metric_tuple",
            "control": 10,
            "treatment": 11,
            "effect_size": 1,
        },
        {
            "variants": "(0, 1)",
            "metric": "metric_dict",
            "control": 20,
            "treatment": 22,
            "effect_size": 2,
        },
        {
            "variants": "(0, 2)",
            "metric": "metric_tuple",
            "control": 10,
            "treatment": 11,
            "effect_size": 1,
        },
        {
            "variants": "(0, 2)",
            "metric": "metric_dict",
            "control": 30,
            "treatment": 33,
            "effect_size": 3,
        },
    )


def test_simulation_results_to_dicts(
    results2: tea_tasting.experiment.ExperimentResults,
) -> None:
    assert tea_tasting.experiment.SimulationResults(results2.values()).to_dicts() == (
        {
            "metric": "metric_tuple",
            "control": 10,
            "treatment": 11,
            "effect_size": 1,
        },
        {
            "metric": "metric_dict",
            "control": 20,
            "treatment": 22,
            "effect_size": 2,
        },
        {
            "metric": "metric_tuple",
            "control": 10,
            "treatment": 11,
            "effect_size": 1,
        },
        {
            "metric": "metric_dict",
            "control": 30,
            "treatment": 33,
            "effect_size": 3,
        },
    )


def test_experiment_power_result_to_dicts() -> None:
    raw_results = (
        {"power": 0.8, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 20_000},
        {"power": 0.9, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 10_000},
        {"power": 0.8, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 10_000},
        {"power": 0.9, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 20_000},
    )
    result = tea_tasting.experiment.ExperimentPowerResult({
        "metric_dict": tea_tasting.metrics.MetricPowerResults[dict[str, float | int]](
            raw_results[0:2]),
        "metric_tuple": tea_tasting.metrics.MetricPowerResults[_PowerResult]([
            _PowerResult(**raw_results[2]),
            _PowerResult(**raw_results[3]),
        ]),
    })
    assert isinstance(result, tea_tasting.utils.DictsReprMixin)
    assert result.default_keys == (
        "metric", "power", "effect_size", "rel_effect_size", "n_obs")
    assert result.to_dicts() == (
        {"metric": "metric_dict", "power": 0.8, "effect_size": 1,
            "rel_effect_size": 0.05, "n_obs": 20_000},
        {"metric": "metric_dict", "power": 0.9, "effect_size": 1,
            "rel_effect_size": 0.05, "n_obs": 10_000},
        {"metric": "metric_tuple", "power": 0.8, "effect_size": 2,
            "rel_effect_size": 0.1, "n_obs": 10_000},
        {"metric": "metric_tuple", "power": 0.9, "effect_size": 2,
            "rel_effect_size": 0.1, "n_obs": 20_000},
    )


def test_experiment_init_default() -> None:
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics)  # ty:ignore[invalid-argument-type]
    assert experiment.metrics == metrics
    assert experiment.variant == "variant"

def test_experiment_init_kwargs() -> None:
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(**metrics)  # ty:ignore[invalid-argument-type]
    assert experiment.metrics == metrics
    assert experiment.variant == "variant"

def test_experiment_init_custom() -> None:
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics, "group")  # ty:ignore[invalid-argument-type]
    assert experiment.metrics == metrics
    assert experiment.variant == "group"


def test_experiment_analyze_default(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    assert experiment.analyze(data) == ref_result

def test_experiment_analyze_base(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_sessions=ref_result["avg_sessions"])

def test_experiment_analyze_aggr(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_orders": _MetricAggregated("orders"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_orders=ref_result["avg_orders"])

def test_experiment_analyze_gran(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_revenue": _MetricGranular("revenue"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_revenue=ref_result["avg_revenue"])


def test_experiment_analyze_aggregated_data(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricAggregated("orders"),
    })
    assert experiment.analyze(data_aggr) == tea_tasting.experiment.ExperimentResult(
        avg_sessions=ref_result["avg_sessions"],
        avg_orders=ref_result["avg_orders"],
    )


def test_experiment_analyze_aggregated_data_all_pairs(
    data_aggr_multi: dict[Hashable, tea_tasting.aggr.Aggregates],
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricAggregated("orders"),
    })
    results = experiment.analyze(data_aggr_multi, control=0, all_variants=True)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): tea_tasting.experiment.ExperimentResult(
            avg_sessions=ref_result["avg_sessions"],
            avg_orders=ref_result["avg_orders"],
        ),
        (0, 2): tea_tasting.experiment.ExperimentResult(
            avg_sessions=ref_result["avg_sessions"],
            avg_orders=ref_result["avg_orders"],
        ),
    })


def test_experiment_analyze_aggregated_data_raises_for_non_aggregated_metric(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
    })
    with pytest.raises(TypeError, match="not based on aggregated statistics"):
        experiment.analyze(data_aggr)


def test_experiment_analyze_aggregated_data_raises_for_granular_metric(
    data_aggr: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    with pytest.raises(TypeError, match="not based on aggregated statistics"):
        experiment.analyze(data_aggr)


def test_experiment_analyze_all_pairs(
    data_arrow_multi: pa.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    results = experiment.analyze(data_arrow_multi, all_variants=True)
    assert set(results.keys()) == {(0, 1), (0, 2), (1, 2)}
    assert results[0, 1] == ref_result
    assert results[0, 2] == ref_result

def test_experiment_analyze_all_pairs_raises(data_arrow_multi: pa.Table) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    with pytest.raises(ValueError, match="all_variants"):
        experiment.analyze(data_arrow_multi)

def test_experiment_analyze_two_treatments(
    data_arrow_multi: pa.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
) -> None:
    experiment = tea_tasting.experiment.Experiment(
        {
            "avg_sessions": _Metric("sessions"),
            "avg_orders": _MetricAggregated("orders"),
            "avg_revenue": _MetricGranular("revenue"),
        },
    )
    results = experiment.analyze(data_arrow_multi, control=0, all_variants=True)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): ref_result,
        (0, 2): ref_result,
    })


def test_experiment_solve_power(data_arrow: pa.Table) -> None:
    experiment = tea_tasting.experiment.Experiment(
        metric=_Metric("sessions"),
        metric_aggr=_MetricAggregated("orders"),
    )
    result = experiment.solve_power(data_arrow)
    assert result == tea_tasting.experiment.ExperimentPowerResult({
        "metric": tea_tasting.metrics.MetricPowerResults((
            _PowerResult(power=0.8, effect_size=1, rel_effect_size=0.05, n_obs=10_000),
            _PowerResult(power=0.9, effect_size=2, rel_effect_size=0.1, n_obs=20_000),
        )),
        "metric_aggr": tea_tasting.metrics.MetricPowerResults((
            {"power": 0.8, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 10_000},
            {"power": 0.9, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 20_000},
        )),
    })


class ExperimentWithSimulationResults(tea_tasting.experiment.Experiment):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.simulation_results = tea_tasting.experiment.SimulationResults()
        self.data = []
        super().__init__(*args, **kwargs)

    def analyze(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tea_tasting.experiment.ExperimentResult:  # ty:ignore[invalid-method-override]
        self.data.append(args[0])
        result = super().analyze(*args, **kwargs)
        self.simulation_results.append(result)
        return result

def test_experiment_simulate_default(data: Frame) -> None:
    experiment = ExperimentWithSimulationResults({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    assert experiment.simulate(data, 10, rng=42) == experiment.simulation_results

def test_experiment_simulate_cols(data: Frame, data_arrow: pa.Table) -> None:
    experiment = ExperimentWithSimulationResults({
        "avg_sessions": _MetricAggregated("sessions"),
    })
    assert experiment.simulate(data, 10, rng=42) == experiment.simulation_results
    assert experiment.data[0].column_names == ["sessions", "variant"]
    experiment = ExperimentWithSimulationResults({
        "avg_orders": _MetricGranular("orders"),
    })
    assert experiment.simulate(data, 10, rng=42) == experiment.simulation_results
    assert experiment.data[0].column_names == ["orders", "variant"]
    experiment = ExperimentWithSimulationResults({
        "avg_revenue": _Metric("revenue"),
    })
    assert experiment.simulate(data, 10, rng=42) == experiment.simulation_results
    assert set(experiment.data[0].column_names) == set(data_arrow.column_names)

def test_experiment_simulate_callable() -> None:
    experiment = ExperimentWithSimulationResults({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    tables = []
    def make_data(rng: np.random.Generator) -> pa.Table:
        table = tea_tasting.datasets.make_users_data(rng=rng, n_users=100)
        tables.append(table)
        return table
    results = experiment.simulate(make_data, 10, rng=42)
    assert results == experiment.simulation_results
    assert results[0] == experiment.analyze(tables[0])


def test_experiment_simulate_callable_aggregated() -> None:
    experiment = ExperimentWithSimulationResults({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricAggregated("orders"),
    })
    aggrs = []
    def make_data(
        rng: np.random.Generator,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        table = tea_tasting.datasets.make_users_data(rng=rng, n_users=100)
        aggr = tea_tasting.aggr.read_aggregates(
            table,
            group_col="variant",
            has_count=False,
            mean_cols=("sessions", "orders"),
            var_cols=(),
            cov_cols=(),
        )
        aggrs.append(aggr)
        return aggr
    results = experiment.simulate(make_data, 10, rng=42)
    assert results == experiment.simulation_results
    assert results[0] == experiment.analyze(aggrs[0])


def test_experiment_simulate_callable_aggregated_raises_for_ratio() -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
    })
    def make_data(
        rng: np.random.Generator,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        return tea_tasting.aggr.read_aggregates(
            tea_tasting.datasets.make_users_data(rng=rng, n_users=100),
            group_col="variant",
            has_count=False,
            mean_cols=("sessions",),
            var_cols=(),
            cov_cols=(),
        )
    with pytest.raises(ValueError, match="ratio parameter"):
        experiment.simulate(make_data, 1, rng=42, ratio=2)


def test_experiment_simulate_callable_aggregated_raises_for_treat() -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
    })
    def make_data(
        rng: np.random.Generator,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        return tea_tasting.aggr.read_aggregates(
            tea_tasting.datasets.make_users_data(rng=rng, n_users=100),
            group_col="variant",
            has_count=False,
            mean_cols=("sessions",),
            var_cols=(),
            cov_cols=(),
        )

    def treat(data: pa.Table) -> pa.Table:
        return data

    with pytest.raises(ValueError, match="treat parameter"):
        experiment.simulate(make_data, 1, rng=42, treat=treat)


def test_experiment_simulate_map(data_arrow: pa.Table) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = experiment.simulate(data_arrow, 10, rng=42, map_=executor.map)
    assert results == experiment.simulate(data_arrow, 10, rng=42)


def test_experiment_simulate_batch_size(data_arrow: pa.Table) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    results = experiment.simulate(data_arrow, 10, rng=42, batch_size=4)
    assert results == experiment.simulate(data_arrow, 10, rng=42, batch_size=1)


def test_experiment_simulate_progress(data_arrow: pa.Table) -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    events: list[object] = []

    class ProgressBar:
        def __enter__(self) -> ProgressBar:
            events.append("__enter__")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("__exit__")

        def update(self, n: int) -> None:
            events.append(n)

    def progress(
        *args: object,
        total: int,
        **kwargs: object,  # noqa: ARG001
    ) -> ProgressBar:
        events.append((args, total))
        return ProgressBar()

    results = experiment.simulate(
        data_arrow,
        10,
        rng=42,
        batch_size=4,
        progress=progress,
    )
    assert results == experiment.simulate(data_arrow, 10, rng=42)
    assert events == [((), 10), "__enter__", 4, 4, 2, "__exit__"]


def test_experiment_simulate_progress_tqdm(data_arrow: pa.Table) -> None:
    tqdm = pytest.importorskip("tqdm")
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
    })
    results = experiment.simulate(
        data_arrow,
        2,
        rng=42,
        batch_size=2,
        progress=tqdm.tqdm,
    )
    assert results == experiment.simulate(data_arrow, 2, rng=42)


def test_experiment_simulate_progress_marimo(data_arrow: pa.Table) -> None:
    marimo = pytest.importorskip("marimo")
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
    })
    results = experiment.simulate(
        data_arrow,
        2,
        rng=42,
        batch_size=2,
        progress=marimo.status.progress_bar,
    )
    assert results == experiment.simulate(data_arrow, 2, rng=42)


def test_experiment_simulate_treat() -> None:
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _MetricAggregated("sessions"),
        "avg_orders": _MetricGranular("orders"),
        "avg_revenue": _Metric("revenue"),
    })
    data = tea_tasting.datasets.make_users_data(
        rng=42,
        n_users=1000,
        orders_uplift=0,
        revenue_uplift=0,
    )
    def treat(data: pa.Table) -> pa.Table:
        return (
            data.drop_columns(["orders", "revenue"])
            .append_column("orders", pc.multiply(data["orders"], pa.scalar(1.1)))
            .append_column("revenue", pc.multiply(data["revenue"], pa.scalar(1.1)))
        )
    results = experiment.simulate(data, 100, rng=42, treat=treat)
    means = (
        results.to_polars()
        .select("metric", "effect_size")
        .group_by("metric").mean()
        .sort("metric")
    )
    assert means.item(2, "effect_size") == pytest.approx(0, abs=0.001)
    assert means.item(0, "effect_size") == pytest.approx(0.05, rel=0.1)
    assert means.item(1, "effect_size") == pytest.approx(0.5, rel=0.1)
