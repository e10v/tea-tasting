"""Multiple hypothesis testing."""

from __future__ import annotations

import abc
from collections import UserDict
from typing import TYPE_CHECKING

import tea_tasting.config
import tea_tasting.experiment
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Literal


NO_NAME_COMPARISON = "-"


class MultipleComparisonsResults(
    UserDict[object, tea_tasting.experiment.ExperimentResult],
    tea_tasting.utils.DictsReprMixin,
):
    """Multiple comparisons result."""
    default_keys = (
        "comparison",
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "pvalue",
        "pvalue_adj",
    )

    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the result to a sequence of dictionaries."""
        return tuple(
            {"comparison": str(comparison)} | metric_result
            for comparison, experiment_result in self.items()
            for metric_result in experiment_result.to_dicts()
        )


def adjust_fdr(
    experiment_results: tea_tasting.experiment.ExperimentResult | Mapping[
        object, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
    *,
    alpha: float | None = None,
    arbitrary_dependence: bool = False,
) -> MultipleComparisonsResults:
    """Adjust p-value and alpha to control the false discovery rate (FDR).

    The number of hypotheses tested is the total number of metrics included in
    the comparison in all experiment results. For example, if there are
    3 experiments with 2 metrics in each, the number of hypotheses is 6.

    The function performs one of the following corrections, depending on parameters:

    - Benjamini-Hochberg procedure, assuming non-negative correlation between
        hypotheses (`arbitrary_dependence=False`).
    - Benjamini-Yekutieli procedure, assuming arbitrary dependence between
        hypotheses (`arbitrary_dependence=True`).

    The function adds the following attributes to the results:

    - `pvalue_adj`: The adjusted p-value, which should be compared with
        the unadjusted FDR (`alpha`).
    - `alpha_adj`: The adjusted FDR, which should be compared with the unadjusted
        p-value (`pvalue`).
    - `null_rejected`: A binary indicator (`0` or `1`) that shows whether
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

    Parameter defaults:
        Default for parameter `alpha` can be changed using the `config_context`
        and `set_context` functions.
        See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
        reference for details.

    References:
        - [Multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).
        - [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate).

    Examples:
        ```pycon
        >>> import polars as pl
        >>> import tea_tasting as tt

        >>> data = pl.concat((
        ...     tt.make_users_data(
        ...         seed=42,
        ...         orders_uplift=0.10,
        ...         revenue_uplift=0.15,
        ...         return_type="polars",
        ...     ),
        ...     tt.make_users_data(
        ...         seed=21,
        ...         orders_uplift=0.15,
        ...         revenue_uplift=0.20,
        ...         return_type="polars",
        ...     )
        ...         .filter(pl.col("variant").eq(1))
        ...         .with_columns(variant=pl.lit(2, pl.Int64)),
        ... ))
        >>> print(data)
        shape: (6_046, 5)
        ┌──────┬─────────┬──────────┬────────┬─────────┐
        │ user ┆ variant ┆ sessions ┆ orders ┆ revenue │
        │ ---  ┆ ---     ┆ ---      ┆ ---    ┆ ---     │
        │ i64  ┆ i64     ┆ i64      ┆ i64    ┆ f64     │
        ╞══════╪═════════╪══════════╪════════╪═════════╡
        │ 0    ┆ 1       ┆ 2        ┆ 1      ┆ 9.58    │
        │ 1    ┆ 0       ┆ 2        ┆ 1      ┆ 6.43    │
        │ 2    ┆ 1       ┆ 2        ┆ 1      ┆ 8.3     │
        │ 3    ┆ 1       ┆ 2        ┆ 1      ┆ 16.65   │
        │ 4    ┆ 0       ┆ 1        ┆ 1      ┆ 7.14    │
        │ …    ┆ …       ┆ …        ┆ …      ┆ …       │
        │ 3989 ┆ 2       ┆ 4        ┆ 4      ┆ 34.93   │
        │ 3991 ┆ 2       ┆ 1        ┆ 0      ┆ 0.0     │
        │ 3992 ┆ 2       ┆ 3        ┆ 3      ┆ 27.96   │
        │ 3994 ┆ 2       ┆ 2        ┆ 1      ┆ 17.22   │
        │ 3998 ┆ 2       ┆ 3        ┆ 0      ┆ 0.0     │
        └──────┴─────────┴──────────┴────────┴─────────┘

        >>> experiment = tt.Experiment(
        ...     sessions_per_user=tt.Mean("sessions"),
        ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        ...     orders_per_user=tt.Mean("orders"),
        ...     revenue_per_user=tt.Mean("revenue"),
        ... )

        >>> # Results without correction.
        >>> results = experiment.analyze(data, control=0, all_variants=True)
        >>> print(results)
        variants             metric control treatment rel_effect_size rel_effect_size_ci  pvalue
          (0, 1)  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]   0.674
          (0, 1) orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%]  0.0762
          (0, 1)    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]   0.118
          (0, 1)   revenue_per_user    5.24      5.99             14%        [2.1%, 28%]  0.0211
          (0, 2)  sessions_per_user    2.00      2.02           0.98%      [-2.1%, 4.1%]   0.532
          (0, 2) orders_per_session   0.266     0.295             11%        [1.2%, 22%]  0.0273
          (0, 2)    orders_per_user   0.530     0.594             12%        [1.7%, 23%]  0.0213
          (0, 2)   revenue_per_user    5.24      6.25             19%        [6.6%, 33%] 0.00218

        >>> # Success metrics.
        >>> metrics = {"orders_per_user", "revenue_per_user"}

        >>> # Benjamini-Hochberg procedure,
        >>> # assuming non-negative correlation between hypotheses.
        >>> adjusted_results_fdr = tt.adjust_fdr(results, metrics)
        >>> print(adjusted_results_fdr)
        comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0284
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0284
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

        >>> # The adjusted confidence level alpha.
        >>> print(adjusted_results_fdr.to_string(keys=(
        ...     "comparison",
        ...     "metric",
        ...     "control",
        ...     "treatment",
        ...     "rel_effect_size",
        ...     "pvalue",
        ...     "alpha_adj",
        ... )))
        comparison           metric control treatment rel_effect_size  pvalue alpha_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118    0.0500
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211    0.0375
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213    0.0375
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.0375

        >>> # Benjamini-Yekutieli procedure,
        >>> # assuming arbitrary dependence between hypotheses.
        >>> print(tt.adjust_fdr(results, metrics, arbitrary_dependence=True))
        comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.245
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0592
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0592
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218     0.0182

        ```
    """  # noqa: E501
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
    experiment_results: tea_tasting.experiment.ExperimentResult | Mapping[
        object, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
    *,
    alpha: float | None = None,
    arbitrary_dependence: bool = False,
    method: Literal["bonferroni", "sidak"] = "sidak",
) -> MultipleComparisonsResults:
    """Adjust p-value and alpha to control the family-wise error rate (FWER).

    The number of hypotheses tested is the total number of metrics included in
    the comparison in all experiment results. For example, if there are
    3 experiments with 2 metrics in each, the number of hypotheses is 6.

    The function performs one of the following procedures, depending on parameters:

    - Hochberg's step-up procedure, assuming non-negative correlation between
        hypotheses (`arbitrary_dependence=False`).
    - Holm's step-down procedure, assuming arbitrary dependence between
        hypotheses (`arbitrary_dependence=True`).

    The function adds the following attributes to the results:

    - `pvalue_adj`: The adjusted p-value, which should be compared with
        the unadjusted FDR (`alpha`).
    - `alpha_adj`: The adjusted FWER, which should be compared with the unadjusted
        p-value (`pvalue`).
    - `null_rejected`: A binary indicator (`0` or `1`) that shows whether
        the null hypothesis is rejected.

    Args:
        experiment_results: Experiment results.
        metrics: Metrics included in the comparison.
            If `None`, all metrics are included.
        alpha: Significance level. If `None`, the value from global settings is used.
        arbitrary_dependence: If `True`, arbitrary dependence between hypotheses
            is assumed and Holm's step-down procedure is performed.
            If `False`, non-negative correlation between hypotheses is assumed
            and Hochberg's step-up procedure is performed.
        method: Correction method, Bonferroni (`"bonferroni"`) or Šidák (`"sidak"`).

    Returns:
        The experiments results with adjusted p-values and alphas.

    Parameter defaults:
        Default for parameter `alpha` can be changed using the `config_context`
        and `set_context` functions.
        See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
        reference for details.

    References:
        - [Multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).
        - [Family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate).
        - [Holm–Bonferroni method](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method).

    Examples:
        ```pycon
        >>> import polars as pl
        >>> import tea_tasting as tt

        >>> data = pl.concat((
        ...     tt.make_users_data(
        ...         seed=42,
        ...         orders_uplift=0.10,
        ...         revenue_uplift=0.15,
        ...         return_type="polars",
        ...     ),
        ...     tt.make_users_data(
        ...         seed=21,
        ...         orders_uplift=0.15,
        ...         revenue_uplift=0.20,
        ...         return_type="polars",
        ...     )
        ...         .filter(pl.col("variant").eq(1))
        ...         .with_columns(variant=pl.lit(2, pl.Int64)),
        ... ))
        >>> print(data)
        shape: (6_046, 5)
        ┌──────┬─────────┬──────────┬────────┬─────────┐
        │ user ┆ variant ┆ sessions ┆ orders ┆ revenue │
        │ ---  ┆ ---     ┆ ---      ┆ ---    ┆ ---     │
        │ i64  ┆ i64     ┆ i64      ┆ i64    ┆ f64     │
        ╞══════╪═════════╪══════════╪════════╪═════════╡
        │ 0    ┆ 1       ┆ 2        ┆ 1      ┆ 9.58    │
        │ 1    ┆ 0       ┆ 2        ┆ 1      ┆ 6.43    │
        │ 2    ┆ 1       ┆ 2        ┆ 1      ┆ 8.3     │
        │ 3    ┆ 1       ┆ 2        ┆ 1      ┆ 16.65   │
        │ 4    ┆ 0       ┆ 1        ┆ 1      ┆ 7.14    │
        │ …    ┆ …       ┆ …        ┆ …      ┆ …       │
        │ 3989 ┆ 2       ┆ 4        ┆ 4      ┆ 34.93   │
        │ 3991 ┆ 2       ┆ 1        ┆ 0      ┆ 0.0     │
        │ 3992 ┆ 2       ┆ 3        ┆ 3      ┆ 27.96   │
        │ 3994 ┆ 2       ┆ 2        ┆ 1      ┆ 17.22   │
        │ 3998 ┆ 2       ┆ 3        ┆ 0      ┆ 0.0     │
        └──────┴─────────┴──────────┴────────┴─────────┘

        >>> experiment = tt.Experiment(
        ...     sessions_per_user=tt.Mean("sessions"),
        ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        ...     orders_per_user=tt.Mean("orders"),
        ...     revenue_per_user=tt.Mean("revenue"),
        ... )

        >>> # Results without correction.
        >>> results = experiment.analyze(data, control=0, all_variants=True)
        >>> print(results)
        variants             metric control treatment rel_effect_size rel_effect_size_ci  pvalue
          (0, 1)  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]   0.674
          (0, 1) orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%]  0.0762
          (0, 1)    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]   0.118
          (0, 1)   revenue_per_user    5.24      5.99             14%        [2.1%, 28%]  0.0211
          (0, 2)  sessions_per_user    2.00      2.02           0.98%      [-2.1%, 4.1%]   0.532
          (0, 2) orders_per_session   0.266     0.295             11%        [1.2%, 22%]  0.0273
          (0, 2)    orders_per_user   0.530     0.594             12%        [1.7%, 23%]  0.0213
          (0, 2)   revenue_per_user    5.24      6.25             19%        [6.6%, 33%] 0.00218

        >>> # Success metrics.
        >>> metrics = {"orders_per_user", "revenue_per_user"}

        >>> # Hochberg's step-up procedure with Šidák correction,
        >>> # assuming non-negative correlation between hypotheses.
        >>> adjusted_results_fwer = tt.adjust_fwer(results, metrics)
        >>> print(adjusted_results_fwer)
        comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0422
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0422
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00869

        >>> # The adjusted confidence level alpha.
        >>> print(adjusted_results_fwer.to_string(keys=(
        ...     "comparison",
        ...     "metric",
        ...     "control",
        ...     "treatment",
        ...     "rel_effect_size",
        ...     "pvalue",
        ...     "alpha_adj",
        ... )))
        comparison           metric control treatment rel_effect_size  pvalue alpha_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118    0.0500
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211    0.0253
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213    0.0253
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.0253

        >>> # Holm's step-down procedure with Bonferroni correction,
        >>> # assuming arbitrary dependence between hypotheses.
        >>> print(tt.adjust_fwer(
        ...     results,
        ...     metrics,
        ...     arbitrary_dependence=True,
        ...     method="bonferroni",
        ... ))
        comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
            (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
            (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0634
            (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0634
            (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

        ```
    """  # noqa: E501, RUF002
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
    experiment_results: tea_tasting.experiment.ExperimentResult | Mapping[
        object, tea_tasting.experiment.ExperimentResult],
    metrics: str | set[str] | Sequence[str] | None = None,
) -> tuple[
    dict[object, tea_tasting.experiment.ExperimentResult],
    list[dict[str, object]],
]:
    if isinstance(experiment_results, tea_tasting.experiment.ExperimentResult):
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
    metric_results: Sequence[dict[str, object]],
    adjust: Callable[[float, int], tuple[float, float]],
) -> None:
    pvalue_adj_max = 1
    alpha_adj_min = 0
    m = len(metric_results)
    for i, metric_result in enumerate(
        sorted(metric_results, key=lambda d: -d["pvalue"]),  # type: ignore
    ):
        k = m - i
        pvalue: float = metric_result["pvalue"]  # type: ignore
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
    metric_results: Sequence[dict[str, object]],
    adjust: Callable[[float, int], tuple[float, float]],
) -> None:
    pvalue_adj_min = 0
    alpha_adj_max = 1
    for k, metric_result in enumerate(
        sorted(metric_results, key=lambda d: d["pvalue"]),  # type: ignore
        start=1,
    ):
        pvalue: float = metric_result["pvalue"]
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
