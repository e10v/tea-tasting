# ty:ignore[invalid-argument-type]
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest

import tea_tasting.experiment
import tea_tasting.multiplicity


if TYPE_CHECKING:
    from collections.abc import Hashable


class _MetricResult(NamedTuple):
    pvalue: float


@pytest.fixture
def experiment_results() -> dict[Hashable, tea_tasting.experiment.ExperimentResult]:
    return {
        (0, 1): tea_tasting.experiment.ExperimentResult({
            "metric1": _MetricResult(pvalue=0.005),
            "metric2": {"pvalue": 0.06},
        }),
        (0, 2): tea_tasting.experiment.ExperimentResult({
            "metric2": _MetricResult(pvalue=0.007),
            "metric3": {"pvalue": 0.04},
        }),
        (0, 3): tea_tasting.experiment.ExperimentResult({
            "metric1": _MetricResult(pvalue=0.01),
            "metric3": {"pvalue": 0.03},
        }),
    }


def test_multiple_comparisons_results_to_dicts() -> None:
    results = tea_tasting.multiplicity.MultipleComparisonsResults({
        (0, 1): tea_tasting.experiment.ExperimentResult({
            "x": {"control": 10, "treatment": 11},
            "y": {"control": 20, "treatment": 22},
        }),
        "a/b": tea_tasting.experiment.ExperimentResult({
            "y": {"control": 30, "treatment": 33},
            "z": {"control": 40, "treatment": 44},
        }),
    })
    assert results.to_dicts() == (
        {"comparison": "(0, 1)", "metric": "x", "control": 10, "treatment": 11},
        {"comparison": "(0, 1)", "metric": "y", "control": 20, "treatment": 22},
        {"comparison": "a/b", "metric": "y", "control": 30, "treatment": 33},
        {"comparison": "a/b", "metric": "z", "control": 40, "treatment": 44},
    )


def test_adjust_fdr_default(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fdr(experiment_results)
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.02)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.06)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.02)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.048)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.02)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.045)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.0416666666666667)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.05)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.0416666666666667)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.0416666666666667)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0416666666666667)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0416666666666667)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 1
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 1

def test_adjust_fdr_arbitrary_dependence(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fdr(
        experiment_results,
        ("metric1", "metric2", "metric3"),
        arbitrary_dependence=True,
    )
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.049)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.147)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.049)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.1176)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.049)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.11025)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.0102040816326531)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.0204081632653061)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.0102040816326531)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.0170068027210884)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0102040816326531)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0136054421768707)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 0
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 0

def test_adjust_fdr_single_experiment() -> None:
    results = tea_tasting.multiplicity.adjust_fdr(
        tea_tasting.experiment.ExperimentResult({
            "metric1": _MetricResult(pvalue=0.005),
            "metric2": {"pvalue": 0.06},
        }),
        arbitrary_dependence=True,
    )
    assert results["-"]["metric1"]["pvalue_adj"] == pytest.approx(0.015)
    assert results["-"]["metric2"]["pvalue_adj"] == pytest.approx(0.09)
    assert results["-"]["metric1"]["alpha_adj"] == pytest.approx(0.016666666666666666)
    assert results["-"]["metric2"]["alpha_adj"] == pytest.approx(0.03333333333333333)
    assert results["-"]["metric1"]["null_rejected"] == 1
    assert results["-"]["metric2"]["null_rejected"] == 0

def test_adjust_fdr_single_metric(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fdr(
        experiment_results,
        "metric1",
        arbitrary_dependence=True,
    )
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.015)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.015)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.03333333333333333)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.03333333333333333)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1


def test_adjust_fwer_default(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fwer(experiment_results)
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.0296274906437343)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.0600000000000001)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.0345134180118071)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.0600000000000001)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.0394039900000001)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.0600000000000001)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.0127414550985662)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.05)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.0127414550985662)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.0253205655191037)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0127414550985662)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0169524275084415)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 0
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 0

def test_adjust_fwer_arbitrary_dependence(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fwer(
        experiment_results,
        arbitrary_dependence=True,
    )
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.0296274906437343)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.087327)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.0345134180118071)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.087327)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.0394039900000001)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.087327)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.0085124446108471)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.0169524275084415)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.0102062183130115)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.0169524275084415)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0127414550985662)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0169524275084415)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 0
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 0

def test_adjust_fwer_bonferroni(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fwer(
        experiment_results,
        method="bonferroni",
    )
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.03)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.06)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.035)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.06)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.04)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.06)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.0125)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.05)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.0125)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.025)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0125)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0166666666666667)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 0
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 0

def test_adjust_fwer_arbitrary_dependence_bonferroni(
    experiment_results: dict[Hashable, tea_tasting.experiment.ExperimentResult],
) -> None:
    results = tea_tasting.multiplicity.adjust_fwer(
        experiment_results,
        arbitrary_dependence=True,
        method="bonferroni",
    )
    assert results[(0, 1)]["metric1"]["pvalue_adj"] == pytest.approx(0.03)
    assert results[(0, 1)]["metric2"]["pvalue_adj"] == pytest.approx(0.09)
    assert results[(0, 2)]["metric2"]["pvalue_adj"] == pytest.approx(0.035)
    assert results[(0, 2)]["metric3"]["pvalue_adj"] == pytest.approx(0.09)
    assert results[(0, 3)]["metric1"]["pvalue_adj"] == pytest.approx(0.04)
    assert results[(0, 3)]["metric3"]["pvalue_adj"] == pytest.approx(0.09)
    assert results[(0, 1)]["metric1"]["alpha_adj"] == pytest.approx(0.00833333333333333)
    assert results[(0, 1)]["metric2"]["alpha_adj"] == pytest.approx(0.0166666666666667)
    assert results[(0, 2)]["metric2"]["alpha_adj"] == pytest.approx(0.01)
    assert results[(0, 2)]["metric3"]["alpha_adj"] == pytest.approx(0.0166666666666667)
    assert results[(0, 3)]["metric1"]["alpha_adj"] == pytest.approx(0.0125)
    assert results[(0, 3)]["metric3"]["alpha_adj"] == pytest.approx(0.0166666666666667)
    assert results[(0, 1)]["metric1"]["null_rejected"] == 1
    assert results[(0, 1)]["metric2"]["null_rejected"] == 0
    assert results[(0, 2)]["metric2"]["null_rejected"] == 1
    assert results[(0, 2)]["metric3"]["null_rejected"] == 0
    assert results[(0, 3)]["metric1"]["null_rejected"] == 1
    assert results[(0, 3)]["metric3"]["null_rejected"] == 0
