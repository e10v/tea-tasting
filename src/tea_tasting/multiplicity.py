"""Methods for multiple comparisons."""

from __future__ import annotations

import abc
from collections import UserDict
from typing import TYPE_CHECKING, Any

import tea_tasting.config
import tea_tasting.experiment
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence


class MultipleComparisonsResult(
    UserDict[Any, tea_tasting.experiment.ExperimentResult],
    tea_tasting.utils.PrettyDictsMixin,
):
    """Multiple comparisons result."""
    default_keys = (
        "comparison",
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "pvalue",
    )

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Convert the result to a sequence of dictionaries."""
        return tuple(
            {"comparison": str(comparison)} | metric_result
            for comparison, experiment_result in self.items()
            for metric_result in experiment_result.to_dicts()
        )


def adjust_fdr(
    experiment_results: tea_tasting.experiment.ExperimentResult | dict[
        Any, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
    *,
    alpha: float | None = None,
    arbitrary_dependence: bool = True,
) -> MultipleComparisonsResult:
    alpha = (
        tea_tasting.utils.auto_check(alpha, "alpha")
        if alpha is not None
        else tea_tasting.config.get_config("alpha")
    )
    arbitrary_dependence = tea_tasting.utils.check_scalar(
        arbitrary_dependence, "arbitrary_dependence", typ=bool)

    results, metric_results = _copy_results(experiment_results, metrics)
    _adjust_stepup(metric_results, _Benjamini(
        alpha=alpha,  # type: ignore
        m=len(metric_results),
        arbitrary_dependence=arbitrary_dependence,
    ))

    return MultipleComparisonsResult(results)


def _adjust_stepup(
    metric_results: Sequence[dict[str, Any]],
    adjustment: _Adjustment,
) -> None:
    ...


def _copy_results(
    experiment_results: tea_tasting.experiment.ExperimentResult | dict[
        Any, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
) -> tuple[
    dict[Any, tea_tasting.experiment.ExperimentResult],
    list[dict[str, Any]],
]:
    if not isinstance(experiment_results, dict):
        experiment_results = {"-": experiment_results}

    if metrics is not None:
        if isinstance(metrics, str):
            metrics = {metrics}
        elif not isinstance(metrics, set):
            metrics = set(metrics)

    copy_of_experiment_results = {}
    copy_of_metric_results = []
    for comparison, experiment_result in experiment_results.items():
        result = {}
        for metric, metric_result in experiment_result.items():
            if metrics is None or metric in metrics:
                copy_of_metric_result = (
                    metric_result.copy()
                    if isinstance(metric_result, dict)
                    else metric_result._asdict()
                )
                result |= {metric: copy_of_metric_result}
                copy_of_metric_results.append(copy_of_metric_result)
        copy_of_experiment_results |= {
            comparison: tea_tasting.experiment.ExperimentResult(result)}

    return copy_of_experiment_results, copy_of_metric_results


class _Adjustment(abc.ABC):
    @abc.abstractmethod
    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        ...


class _Benjamini(_Adjustment):
    def __init__(self,
        alpha: float,
        m: int,
        *,
        arbitrary_dependence: bool,
    ) -> None:
        self.alpha = alpha
        self.denom_ = (
            m * sum(1/(i+1) for i in range(m))
            if arbitrary_dependence
            else m
        )

    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        coef = k / self.denom_
        return pvalue / coef, self.alpha * coef
