from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypedDict

import pandas as pd
import pytest

import tea_tasting.experiment


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
