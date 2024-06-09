"""Metrics with resampling methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting.config
from tea_tasting.metrics.base import MetricBaseGranular
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy.typing as npt
    import pandas as pd


class BootstrapResult(NamedTuple):
    """Result of the analysis using bootstrap resampling.

    Attributes:
        control: Control statistic value.
        treatment: Treatment statistic value.
        effect_size: Absolute effect size. Difference between the two statistic values.
        effect_size_ci_lower: Lower bound of the absolute effect size
            confidence interval.
        effect_size_ci_upper: Upper bound of the absolute effect size
            confidence interval.
        rel_effect_size: Relative effect size. Difference between the two statistic
            values, divided by the control statistic value.
        rel_effect_size_ci_lower: Lower bound of the relative effect size
            confidence interval.
        rel_effect_size_ci_upper: Upper bound of the relative effect size
            confidence interval.
    """
    control: float
    treatment: float
    effect_size: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    rel_effect_size: float
    rel_effect_size_ci_lower: float
    rel_effect_size_ci_upper: float


class Bootstrap(MetricBaseGranular[BootstrapResult]):  # noqa: D101
    def __init__(  # noqa: PLR0913
        self,
        columns: str | Sequence[str],
        statistic: Callable[..., npt.NDArray[np.number[Any]]],
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        n_resamples: int = 10_000,
        method: Literal["percentile", "basic", "bca"] = "bca",
        batch: int | None = None,
        random_state: int | np.random.Generator | np.random.SeedSequence | None = None,
    ) -> None:
        """Metric for analysis of a statistic using confidence interval.

        Args:
            columns: Names of the columns to be used in the analysis.
            statistic: Statistic. It must be a vectorized callable
                that accepts a NumPy array as the first argument and returns
                the resulting statistic.
                It must also accept a keyword argument `axis` and be vectorized
                to compute the statistic along the provided axis.
            alternative: Alternative hypothesis.
            confidence_level: Confidence level for the confidence interval.
            n_resamples: The number of resamples performed to form
                the bootstrap distribution of the statistic.
            method: Whether to return the "percentile" bootstrap confidence
                interval (`"percentile"`), the "basic" (AKA "reverse") bootstrap
                confidence interval (`"basic"`), or the bias-corrected
                and accelerated bootstrap confidence interval (`"bca"`).
            batch: The number of resamples to process in each vectorized call
                to statistic. Memory usage is O(`batch * n`), where `n` is
                the sample size. Default is `None`, in which case `batch = n_resamples`
                (or `batch = max(n_resamples, n)` for method="bca").
            random_state: Pseudorandom number generator state used
                to generate resamples.

        Notes:
            If `columns` is a sequence of strings, then the sample passed
            to the statistic callable contains an extra dimension in the first axis.
            See examples below.

        Examples:
            ```python
            import numpy as np
            import tea_tasting as tt


            experiment = tt.Experiment(
                orders_per_user=tt.Bootstrap("orders", np.mean, random_state=42),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>          metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #> orders_per_user   0.530     0.573            8.0%       [-1.8%, 19%]      -
            ```

            With multiple columns:

            ```python
            def ratio_of_means(sample, axis):
                means = np.mean(sample, axis=axis)
                return means[0] / means[1]

            experiment = tt.Experiment(
                orders_per_session=tt.Bootstrap(
                    ("orders", "sessions"),
                    ratio_of_means,
                    random_state=42,
                ),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #> orders_per_session   0.266     0.289            8.8%      [-0.61%, 20%]      -
            ```
        """  # noqa: E501
        tea_tasting.utils.check_scalar(columns, "columns", typ=str | Sequence)
        if not isinstance(columns, str):
            for col in columns:
                tea_tasting.utils.check_scalar(col, "column", typ=str)
        self.columns = columns

        self.statistic = tea_tasting.utils.check_scalar(
            statistic, "statistic", typ=Callable)

        self.alternative = (
            tea_tasting.utils.auto_check(alternative, "alternative")
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )

        self.confidence_level = (
            tea_tasting.utils.auto_check(confidence_level, "confidence_level")
            if confidence_level is not None
            else tea_tasting.config.get_config("confidence_level")
        )

        self.n_resamples = tea_tasting.utils.check_scalar(
            n_resamples, "n_resamples", typ=int, gt=0)

        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, in_={"percentile", "basic", "bca"})

        self.batch = tea_tasting.utils.check_scalar(batch, "batch", typ=int | None)

        self.random_state = tea_tasting.utils.check_scalar(
            random_state,
            "random_state",
            typ=int | np.random.Generator | np.random.SeedSequence | None,
        )


    @property
    def cols(self) -> Sequence[str]:
        """Columns to be fetched for a metric analysis."""
        if isinstance(self.columns, str):
            return (self.columns,)
        return self.columns


    def analyze_dataframes(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
    ) -> BootstrapResult:
        """Analyze metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        def statistic(
            contr: npt.NDArray[np.number[Any]],
            treat: npt.NDArray[np.number[Any]],
            axis: int = -1,
        ) -> npt.NDArray[np.number[Any]]:
            contr_stat = self.statistic(contr, axis=axis)
            treat_stat = self.statistic(treat, axis=axis)

            effect_size = treat_stat - contr_stat
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_effect_size = np.divide(treat_stat, contr_stat) - 1

            return np.stack((effect_size, rel_effect_size), axis=0)

        contr = control.loc[:, self.columns].to_numpy()  # type: ignore
        treat = treatment.loc[:, self.columns].to_numpy()  # type: ignore
        stat = statistic(contr, treat, axis=0)

        result = scipy.stats.bootstrap(
            (contr, treat),
            statistic,
            n_resamples=self.n_resamples,
            batch=self.batch,
            axis=0,
            confidence_level=self.confidence_level,
            alternative=self.alternative,
            method=self.method,
            random_state=self.random_state,
        )
        ci = result.confidence_interval

        return BootstrapResult(
            control=self.statistic(contr, axis=0),  # type: ignore
            treatment=self.statistic(treat, axis=0),  # type: ignore
            effect_size=stat[0],
            effect_size_ci_lower=ci.low[0],
            effect_size_ci_upper=ci.high[0],
            rel_effect_size=stat[1],
            rel_effect_size_ci_lower=ci.low[1],
            rel_effect_size_ci_upper=ci.high[1],
        )
