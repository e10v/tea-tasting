from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import ibis
import ibis.expr.types
import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.experiment
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import narwhals.typing  # noqa: TC004


    Frame = ibis.expr.types.Table | pd.DataFrame | pl.LazyFrame


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
    def __init__(self, value: str) -> None:
        self.value = value

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: int,
        treatment: int,
        variant: str,
    ) -> _MetricResultTuple:
        if not isinstance(data, pd.DataFrame):
            if not isinstance(data, ibis.expr.types.Table):
                data = nw.from_native(data)
                if isinstance(data, nw.LazyFrame):
                    data = data.collect()
            data = data.to_pandas()

        agg_data = data.loc[:, [variant, self.value]].groupby(variant).agg("mean")
        contr_mean = agg_data.loc[control, self.value]
        treat_mean = agg_data.loc[treatment, self.value]
        return _MetricResultTuple(
            control=contr_mean,  # type: ignore
            treatment=treat_mean,  # type: ignore
            effect_size=treat_mean - contr_mean,  # type: ignore
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
        tea_tasting.metrics.MetricPowerResults[dict[str, Any]]],
):
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
        return tea_tasting.experiment.ExperimentResult(
            metric_tuple=_MetricResultTuple(*x),
            metric_dict=_MetricResultDict(
                control=y[0],
                treatment=y[1],
                effect_size=y[2],
            ),  # type: ignore
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
def data_pandas() -> pd.DataFrame:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)

@pytest.fixture
def data_polars(data_pandas: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(data_pandas)

@pytest.fixture
def data_polars_lazy(data_polars: pl.DataFrame) -> pl.LazyFrame:
    return data_polars.lazy()

@pytest.fixture
def data_duckdb(data_pandas: pd.DataFrame) -> ibis.expr.types.Table:
    return ibis.connect("duckdb://").create_table("data", data_pandas)

@pytest.fixture
def data_sqlite(data_pandas: pd.DataFrame) -> ibis.expr.types.Table:
    return ibis.connect("sqlite://").create_table("data", data_pandas)

@pytest.fixture(params=[
    "data_pandas", "data_polars", "data_polars_lazy", "data_duckdb", "data_sqlite"])
def data(request: pytest.FixtureRequest) -> Frame:
    return request.getfixturevalue(request.param)

@pytest.fixture
def data_pandas_multi(data_pandas: pd.DataFrame) -> pd.DataFrame:
    return pd.concat((
        data_pandas,
        data_pandas.query("variant==1").assign(variant=2),
    ))


@pytest.fixture
def ref_result(
    data_pandas: pd.DataFrame,
) -> tea_tasting.experiment.ExperimentResults:
    sessions = _Metric("sessions")
    orders = _MetricAggregated("orders")
    revenue = _MetricGranular("revenue")
    return tea_tasting.experiment.ExperimentResult(
        avg_sessions=sessions.analyze(data_pandas, 0, 1, "variant"),
        avg_orders=orders.analyze(data_pandas, 0, 1, "variant"),
        avg_revenue=revenue.analyze(data_pandas, 0, 1, "variant"),  # type: ignore
    )


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

def test_experiment_result_to_pretty(result2: tea_tasting.experiment.ExperimentResult):
    pd.testing.assert_frame_equal(
        result2.to_pretty(),
        pd.DataFrame((
            {
                "metric": "metric_tuple",
                "control": "4.44",
                "treatment": "5.56",
                "rel_effect_size": "20%",
                "rel_effect_size_ci": "[12%, ∞]",
                "pvalue": "0.235",
            },
            {
                "metric": "metric_dict",
                "control": "10.0",
                "treatment": "11.1",
                "rel_effect_size": "11%",
                "rel_effect_size_ci": "[0.0%, -]",
                "pvalue": "-",
            },
        )),
    )

def test_experiment_result_to_string(result2: tea_tasting.experiment.ExperimentResult):
    assert result2.to_string() == pd.DataFrame((
        {
            "metric": "metric_tuple",
            "control": "4.44",
            "treatment": "5.56",
            "rel_effect_size": "20%",
            "rel_effect_size_ci": "[12%, ∞]",
            "pvalue": "0.235",
        },
        {
            "metric": "metric_dict",
            "control": "10.0",
            "treatment": "11.1",
            "rel_effect_size": "11%",
            "rel_effect_size_ci": "[0.0%, -]",
            "pvalue": "-",
        },
    )).to_string(index=False)

def test_experiment_result_to_html(result2: tea_tasting.experiment.ExperimentResult):
    assert result2.to_html() == pd.DataFrame((
        {
            "metric": "metric_tuple",
            "control": "4.44",
            "treatment": "5.56",
            "rel_effect_size": "20%",
            "rel_effect_size_ci": "[12%, ∞]",
            "pvalue": "0.235",
        },
        {
            "metric": "metric_dict",
            "control": "10.0",
            "treatment": "11.1",
            "rel_effect_size": "11%",
            "rel_effect_size_ci": "[0.0%, -]",
            "pvalue": "-",
        },
    )).to_html(index=False)


def test_experiment_results_to_dicts(
    results2: tea_tasting.experiment.ExperimentResults,
):
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


def test_experiment_power_result_to_dicts():
    raw_results = (
        {"power": 0.8, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 20_000},
        {"power": 0.9, "effect_size": 1, "rel_effect_size": 0.05, "n_obs": 10_000},
        {"power": 0.8, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 10_000},
        {"power": 0.9, "effect_size": 2, "rel_effect_size": 0.1, "n_obs": 20_000},
    )
    result = tea_tasting.experiment.ExperimentPowerResult({
        "metric_dict": tea_tasting.metrics.MetricPowerResults[dict[str, Any]](
            raw_results[0:2]),
        "metric_tuple": tea_tasting.metrics.MetricPowerResults[_PowerResult]([
            _PowerResult(**raw_results[2]),
            _PowerResult(**raw_results[3]),
        ]),
    })
    assert isinstance(result, tea_tasting.utils.PrettyDictsMixin)
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


def test_experiment_init_default():
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics)  # type: ignore
    assert experiment.metrics == metrics
    assert experiment.variant == "variant"

def test_experiment_init_kwargs():
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(**metrics)  # type: ignore
    assert experiment.metrics == metrics
    assert experiment.variant == "variant"

def test_experiment_init_custom():
    metrics = {
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    }
    experiment = tea_tasting.experiment.Experiment(metrics, "group")  # type: ignore
    assert experiment.metrics == metrics
    assert experiment.variant == "group"


def test_experiment_analyze_default(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    assert experiment.analyze(data) == ref_result

def test_experiment_analyze_base(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_sessions=ref_result["avg_sessions"])

def test_experiment_analyze_aggr(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_orders": _MetricAggregated("orders"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_orders=ref_result["avg_orders"])

def test_experiment_analyze_gran(
    data: Frame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_revenue": _MetricGranular("revenue"),
    })
    assert experiment.analyze(data) == tea_tasting.experiment.ExperimentResult(
        avg_revenue=ref_result["avg_revenue"])

def test_experiment_analyze_all_pairs(
    data_pandas_multi: pd.DataFrame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    results = experiment.analyze(data_pandas_multi, all_variants=True)
    assert set(results.keys()) == {(0, 1), (0, 2), (1, 2)}
    assert results[0, 1] == ref_result
    assert results[0, 2] == ref_result

def test_experiment_analyze_all_pairs_raises(data_pandas_multi: pd.DataFrame):
    experiment = tea_tasting.experiment.Experiment({
        "avg_sessions": _Metric("sessions"),
        "avg_orders": _MetricAggregated("orders"),
        "avg_revenue": _MetricGranular("revenue"),
    })
    with pytest.raises(ValueError, match="all_variants"):
        experiment.analyze(data_pandas_multi)

def test_experiment_analyze_two_treatments(
    data_pandas_multi: pd.DataFrame,
    ref_result: tea_tasting.experiment.ExperimentResult,
):
    experiment = tea_tasting.experiment.Experiment(
        {
            "avg_sessions": _Metric("sessions"),
            "avg_orders": _MetricAggregated("orders"),
            "avg_revenue": _MetricGranular("revenue"),
        },
    )
    results = experiment.analyze(data_pandas_multi, control=0, all_variants=True)
    assert results == tea_tasting.experiment.ExperimentResults({
        (0, 1): ref_result,
        (0, 2): ref_result,
    })


def test_experiment_solve_power(data_pandas: pd.DataFrame):
    experiment = tea_tasting.experiment.Experiment(
        metric=_Metric("sessions"),
        metric_aggr=_MetricAggregated("orders"),
    )
    result = experiment.solve_power(data_pandas)
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
