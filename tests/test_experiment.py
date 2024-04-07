from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypedDict

import ibis
import ibis.expr.types
import pandas as pd
import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.experiment
import tea_tasting.metrics


if TYPE_CHECKING:
    from collections.abc import Callable


class _MetricResultTuple(NamedTuple):
    control: float
    treatment: float
    effect_size: float

class _MetricResultDict(TypedDict):
    control: float
    treatment: float
    effect_size: float


class _Metric(tea_tasting.metrics.MetricBase[_MetricResultTuple]):
    def __init__(self, value: str) -> None:
        self.value = value

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: int,
        treatment: int,
        variant_col: str,
    ) -> _MetricResultTuple:
        if isinstance(data, pd.DataFrame):
            con = ibis.pandas.connect()
            data = con.create_table("data", data)
        agg_data = (
            data.group_by(variant_col)
            .agg(mean=data[self.value].mean())  # type: ignore
            .to_pandas()
        )
        contr_mean = agg_data.loc[control, "mean"]
        treat_mean = agg_data.loc[treatment, "mean"]
        return _MetricResultTuple(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=treat_mean - contr_mean,  # type: ignore
        )


class _MetricAggregated(tea_tasting.metrics.MetricBaseAggregated[_MetricResultTuple]):
    def __init__(self, value: str) -> None:
        self.value = value

    @property
    def aggr_cols(self) -> tea_tasting.metrics.AggrCols:
        return tea_tasting.metrics.AggrCols(mean_cols=(self.value,))

    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> _MetricResultTuple:
        contr_mean = control.mean(self.value)
        treat_mean = treatment.mean(self.value)
        return _MetricResultTuple(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=treat_mean - contr_mean,
        )


class _MetricGranular(tea_tasting.metrics.MetricBaseGranular[_MetricResultDict]):  # type: ignore
    def __init__(self, value: str) -> None:
        self.value = value

    @property
    def cols(self) -> tuple[str, ...]:
        return (self.value,)

    def analyze_dataframes(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
    ) -> _MetricResultDict:
        contr_mean = control.loc[:, self.value].mean()
        treat_mean = treatment.loc[:, self.value].mean()
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
        return tea_tasting.experiment.ExperimentResult({
            "metric_tuple": _MetricResultTuple(*x),
            "metric_dict": _MetricResultDict(
                control=y[0],
                treatment=y[1],
                effect_size=y[2],
            ),  # type: ignore
        })

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
def data() -> ibis.expr.types.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)


@pytest.fixture
def ref_result(
    data: ibis.expr.types.Table,
) -> tea_tasting.experiment.ExperimentResults:
    sessions = _Metric("sessions")
    orders = _MetricAggregated("orders")
    revenue = _MetricGranular("revenue")
    return tea_tasting.experiment.ExperimentResult({
        "avg_sessions": sessions.analyze(data, 0, 1, "variant"),
        "avg_orders": orders.analyze(data, 0, 1, "variant"),
        "avg_revenue": revenue.analyze(data, 0, 1, "variant"),  # type: ignore
    })


def test_experiment_result_keys(result: tea_tasting.experiment.ExperimentResult):
    assert result.keys() == ("metric_tuple", "metric_dict")


def test_experiment_result_get(result: tea_tasting.experiment.ExperimentResult):
    assert result.get("metric_tuple") == _MetricResultTuple(10, 11, 1)
    assert result.get("metric_dict") == _MetricResultDict(
        control=20, treatment=22, effect_size=2)


def test_experiment_result_to_dicts(result: tea_tasting.experiment.ExperimentResult):
    assert result.to_dicts() == (
        {"metric": "metric_tuple", "control": 10, "treatment": 11, "effect_size": 1},
        {"metric": "metric_dict", "control": 20, "treatment": 22, "effect_size": 2},
    )


def test_experiment_result_to_pandas(result: tea_tasting.experiment.ExperimentResult):
    pd.testing.assert_frame_equal(
        result.to_pandas(),
        pd.DataFrame({
            "metric": ("metric_tuple", "metric_dict"),
            "control": (10, 20),
            "treatment": (11, 22),
            "effect_size": (1, 2),
        }),
    )


def test_experiment_results_keys(
    results: tea_tasting.experiment.ExperimentResults,
    results2: tea_tasting.experiment.ExperimentResults,
):
    assert results.keys() == ((0, 1),)
    assert results2.keys() == ((0, 1), (0, 2))


def test_experiment_results_get_default(
    results: tea_tasting.experiment.ExperimentResults,
):
    result = results.get()
    assert isinstance(result, tea_tasting.experiment.ExperimentResult)
    assert result.result == {
        "metric_tuple": _MetricResultTuple(10, 11, 1),
        "metric_dict": _MetricResultDict(
            control=20,
            treatment=22,
            effect_size=2,
        ),  # type: ignore
    }

def test_experiment_results_get_param(
    results2: tea_tasting.experiment.ExperimentResults,
):
    result = results2.get(0, 2)
    assert isinstance(result, tea_tasting.experiment.ExperimentResult)
    assert result.result == {
        "metric_tuple": _MetricResultTuple(10, 11, 1),
        "metric_dict": _MetricResultDict(
            control=30,
            treatment=33,
            effect_size=3,
        ),  # type: ignore
    }

def test_experiment_results_get_raises(
    results2: tea_tasting.experiment.ExperimentResults,
):
    with pytest.raises(ValueError, match="not None"):
        results2.get()


def test_experiment_results_to_dicts_default(
    results: tea_tasting.experiment.ExperimentResults,
):
    assert results.to_dicts() == (
        {"metric": "metric_tuple", "control": 10, "treatment": 11, "effect_size": 1},
        {"metric": "metric_dict", "control": 20, "treatment": 22, "effect_size": 2},
    )

def test_experiment_results_to_dicts_param(
    results2: tea_tasting.experiment.ExperimentResults,
):
    assert results2.to_dicts(0, 2) == (
        {"metric": "metric_tuple", "control": 10, "treatment": 11, "effect_size": 1},
        {"metric": "metric_dict", "control": 30, "treatment": 33, "effect_size": 3},
    )


def test_experiment_results_to_pandas_default(
    results: tea_tasting.experiment.ExperimentResults,
):
    pd.testing.assert_frame_equal(
        results.to_pandas(),
        pd.DataFrame({
            "metric": ("metric_tuple", "metric_dict"),
            "control": (10, 20),
            "treatment": (11, 22),
            "effect_size": (1, 2),
        }),
    )

def test_experiment_results_to_pandas_param(
    results2: tea_tasting.experiment.ExperimentResults,
):
    pd.testing.assert_frame_equal(
        results2.to_pandas(0, 2),
        pd.DataFrame({
            "metric": ("metric_tuple", "metric_dict"),
            "control": (10, 30),
            "treatment": (11, 33),
            "effect_size": (1, 3),
        }),
    )


def test_experiment_init_default():
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics) # type: ignore
    assert experiment.metrics == metrics
    assert experiment.variant_col == "variant"
    assert experiment.control is None

def test_experiment_init_custom():
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics, "group", 0) # type: ignore
    assert experiment.metrics == metrics
    assert experiment.variant_col == "group"
    assert experiment.control == 0


def test_experiment_analyze_default(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    results = experiment.analyze(data)
    assert results == tea_tasting.experiment.ExperimentResults({(0, 1): ref_result})

def test_experiment_analyze_base(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
    })
    results = experiment.analyze(data)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): tea_tasting.experiment.ExperimentResult(
            {"avg_sessions": ref_result.get("avg_sessions")}),
    })

def test_experiment_analyze_base_pandas(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
    })
    results = experiment.analyze(data.to_pandas())
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): tea_tasting.experiment.ExperimentResult(
            {"avg_sessions": ref_result.get("avg_sessions")}),
    })

def test_experiment_analyze_aggr(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_orders": _MetricAggregated("orders"),
    })
    results = experiment.analyze(data)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): tea_tasting.experiment.ExperimentResult(
            {"avg_orders": ref_result.get("avg_orders")}),
    })

def test_experiment_analyze_gran(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_revenue": _MetricGranular("revenue"),
    })
    results = experiment.analyze(data)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): tea_tasting.experiment.ExperimentResult(
            {"avg_revenue": ref_result.get("avg_revenue")}),
    })

def test_experiment_analyze_all_pairs(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    data = ibis.union(
        data,
        data.filter(data["variant"] == 1)  # type: ignore
            .mutate(variant=ibis.literal(2, data.schema().fields["variant"])),
    )
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    results = experiment.analyze(data)
    assert set(results.keys()) == {(0, 1), (0, 2), (1, 2)}
    assert results.get(0, 1) == ref_result
    assert results.get(0, 2) == ref_result

def test_experiment_analyze_two_treatments(
    data: ibis.expr.types.Table,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    data = ibis.union(
        data,
        data.filter(data["variant"] == 1)  # type: ignore
            .mutate(variant=ibis.literal(2, data.schema().fields["variant"])),
    )
    experiment = tea_tasting.experiment.Experiment(
        {
            "avg_sessions": _Metric("sessions"),
            "avg_orders": _MetricAggregated("orders"),
            "avg_revenue": _MetricGranular("revenue"),
        },
        control=0,
    )
    results = experiment.analyze(data)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): ref_result,
        (0, 2): ref_result,
    })
