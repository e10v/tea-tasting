from __future__ import annotations

from typing import NamedTuple, TypedDict

import pandas as pd
import pytest

import tea_tasting.experiment


@pytest.fixture
def result() -> tea_tasting.experiment.ExperimentResult:
    class MetricResultTuple(NamedTuple):
        control: float
        treatment: float
        effect_size: float

    class MetricResultDict(TypedDict):
        control: float
        treatment: float
        effect_size: float

    return tea_tasting.experiment.ExperimentResult({
        "metric_tuple": MetricResultTuple(10, 11, 1),
        "metric_dict": MetricResultDict(control=20, treatment=22, effect_size=2), # type: ignore
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
