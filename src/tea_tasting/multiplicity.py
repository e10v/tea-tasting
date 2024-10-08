"""Methods for multiple comparisons."""

from __future__ import annotations

import abc
from collections import UserDict
from typing import TYPE_CHECKING, Any

import tea_tasting.config
import tea_tasting.experiment
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal


NO_NAME_COMPARISON = "-"


class MultipleComparisonsResults(
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
        "pvalue_adj",
        "null_rejected",
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
) -> MultipleComparisonsResults:
    """Adjust p-value and alpha to control the false discovery rate (FDR).

    The number of hypotheses tested is the total number of metrics included in
    the comparison in all experiment results. For example, if there are
    3 experiments with 2 metrics in each, the number of hypotheses is 6.

    The function performs one of the following corrections, depending on parameters:

    - Benjamini-Yekutieli procedure, assuming arbitrary dependence between
        hypotheses (`arbitrary_dependence=True`).
    - Benjamini-Hochberg procedure, assuming non-negative correlation between
        hypotheses (`arbitrary_dependence=False`).

    The function adds the following attributes to the results:
        `pvalue_adj`: The adjusted p-value, which should be compared with the unadjusted
            FDR (`alpha`).
        `alpha_adj`: "The adjusted FDR, which should be compared with the unadjusted
            p-value (`pvalue`).
        `null_rejected`: A binary indicator (`0` or `1`) that shows whether
            the null hypothesis is rejected.

    Args:
        experiment_results: Experiment results.
        metrics: Metrics included in the comparison.
            If `None`, all metrics are included.
        alpha: Significance level. If `None`, the value from global settings is used.
        arbitrary_dependence: If `True`, arbitrary dependence between hypotheses
            is assumed and Benjamini-Yekutieli procedure is performed.
            If `False`, non-negative correlation between hypotheses is assumed
            and Benjamini-Hochberg procedure is performed.

    Returns:
        The experiments results with adjusted p-values and alphas.
    """
    alpha = (
        tea_tasting.utils.auto_check(alpha, "alpha")
        if alpha is not None
        else tea_tasting.config.get_config("alpha")
    )
    arbitrary_dependence = tea_tasting.utils.check_scalar(
        arbitrary_dependence, "arbitrary_dependence", typ=bool)

    # results and metric_results refer to the same dicts.
    results, metric_results = _copy_results(experiment_results, metrics)
    method = _Benjamini(
        alpha=alpha,  # type: ignore
        m=len(metric_results),
        arbitrary_dependence=arbitrary_dependence,
    )
    # In-place update.
    _hochberg_stepup(metric_results, method.adjust)

    return MultipleComparisonsResults(results)


def adjust_fwer(
    experiment_results: tea_tasting.experiment.ExperimentResult | dict[
        Any, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
    *,
    alpha: float | None = None,
    method: Literal["sidak", "bonferroni"] = "sidak",
    arbitrary_dependence: bool = True,
) -> MultipleComparisonsResults:
    """Adjust p-value and alpha to control the family-wise error rate (FWER).

    The number of hypotheses tested is the total number of metrics included in
    the comparison in all experiment results. For example, if there are
    3 experiments with 2 metrics in each, the number of hypotheses is 6.

    The function performs one of the following procedures, depending on parameters:

    - Holm's step-down procedure, assuming arbitrary dependence between
        hypotheses (`arbitrary_dependence=True`).
    - Hochberg's step-up procedure, assuming non-negative correlation between
        hypotheses (`arbitrary_dependence=False`).

    Args:
        experiment_results: Experiment results.
        metrics: Metrics included in the comparison.
            If `None`, all metrics are included.
        alpha: Significance level. If `None`, the value from global settings is used.
        method: Correction method, Šidák (`"sidak"`) or Bonferroni (`"bonferroni"`).
        arbitrary_dependence: If `True`, arbitrary dependence between hypotheses
            is assumed and Holm's step-down procedure is performed.
            If `False`, non-negative correlation between hypotheses is assumed
            and Hochberg's step-up procedure is performed.

    Returns:
        The experiments results with adjusted p-values and alphas.
    """
    alpha = (
        tea_tasting.utils.auto_check(alpha, "alpha")
        if alpha is not None
        else tea_tasting.config.get_config("alpha")
    )
    method = tea_tasting.utils.check_scalar(
        method, "method", typ=str, in_={"sidak", "bonferroni"})
    arbitrary_dependence = tea_tasting.utils.check_scalar(
        arbitrary_dependence, "arbitrary_dependence", typ=bool)

    # results and metric_results refer to the same dicts.
    results, metric_results = _copy_results(experiment_results, metrics)
    method_cls = _Sidak if method == "sidak" else _Bonferroni
    method_ = method_cls(alpha=alpha, m=len(metric_results))  # type: ignore
    procedure = _holm_stepdown if arbitrary_dependence else _hochberg_stepup
    # In-place update.
    procedure(metric_results, method_.adjust)

    return MultipleComparisonsResults(results)


def _copy_results(
    experiment_results: tea_tasting.experiment.ExperimentResult | dict[
        Any, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
) -> tuple[
    dict[Any, tea_tasting.experiment.ExperimentResult],
    list[dict[str, Any]],
]:
    if not isinstance(experiment_results, dict):
        experiment_results = {NO_NAME_COMPARISON: experiment_results}

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


def _hochberg_stepup(
    metric_results: Sequence[dict[str, Any]],
    adjust: Callable[[float, int], tuple[float, float]],
) -> None:
    pvalue_adj_max = 1
    alpha_adj_min = 0
    m = len(metric_results)
    for i, metric_result in enumerate(
        sorted(metric_results, key=lambda d: -d["pvalue"]),
    ):
        k = m - i
        pvalue = metric_result["pvalue"]
        pvalue_adj, alpha_adj = adjust(pvalue, k)

        if alpha_adj_min == 0 and pvalue <= alpha_adj:
            alpha_adj_min = alpha_adj
        alpha_adj = max(alpha_adj, alpha_adj_min)
        pvalue_adj = pvalue_adj_max = min(pvalue_adj, pvalue_adj_max)

        metric_result.update(
            pvalue_adj=pvalue_adj,
            alpha_adj=alpha_adj,
            null_rejected=int(pvalue <= alpha_adj),
        )


def _holm_stepdown(
    metric_results: Sequence[dict[str, Any]],
    adjust: Callable[[float, int], tuple[float, float]],
) -> None:
    pvalue_adj_min = 0
    alpha_adj_max = 1
    for k, metric_result in enumerate(
        sorted(metric_results, key=lambda d: d["pvalue"]),
        start=1,
    ):
        pvalue = metric_result["pvalue"]
        pvalue_adj, alpha_adj = adjust(pvalue, k)

        if alpha_adj_max == 1 and pvalue > alpha_adj:
            alpha_adj_max = alpha_adj
        alpha_adj = min(alpha_adj, alpha_adj_max)
        pvalue_adj = pvalue_adj_min = max(pvalue_adj, pvalue_adj_min)

        metric_result.update(
            pvalue_adj=pvalue_adj,
            alpha_adj=alpha_adj,
            null_rejected=int(pvalue <= alpha_adj),
        )


class _Correction(abc.ABC):
    @abc.abstractmethod
    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        ...


class _Benjamini(_Correction):
    def __init__(self,
        alpha: float,
        m: int,
        *,
        arbitrary_dependence: bool,
    ) -> None:
        self.alpha = alpha
        self.m_adj_ = (
            m * sum(1 / i for i in range(1, m + 1))
            if arbitrary_dependence
            else m
        )

    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        coef = self.m_adj_ / k
        return min(pvalue * coef, 1), self.alpha / coef


class _Bonferroni(_Correction):
    def __init__(self, alpha: float, m: int) -> None:
        self.alpha = alpha
        self.m = m

    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        coef = self.m - k + 1
        return min(pvalue * coef, 1), self.alpha / coef


class _Sidak(_Correction):
    def __init__(self, alpha: float, m: int) -> None:
        self.alpha = alpha
        self.m = m

    def adjust(self, pvalue: float, k: int) -> tuple[float, float]:
        coef = self.m - k + 1
        return 1 - (1 - pvalue)**coef, 1 - (1 - self.alpha)**(1 / coef)
