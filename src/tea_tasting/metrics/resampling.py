"""Metrics analyzed using resampling methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting.config
from tea_tasting.metrics.base import MetricBaseGranular
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    import pyarrow as pa


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
    def __init__(
        self,
        columns: str | Sequence[str],
        statistic: Callable[..., npt.NDArray[np.number]],
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        n_resamples: int | None = None,
        method: Literal["percentile", "basic", "bca"] = "bca",
        batch: int | None = None,
        random_state: int | np.random.Generator | np.random.SeedSequence | None = None,
    ) -> None:
        """Metric for analysis of a statistic using bootstrap resampling.

        If `columns` is a sequence of strings, then the sample passed
        to the statistic callable contains an extra dimension in the first axis.
        See examples below.

        Args:
            columns: Names of the columns to be used in the analysis.
            statistic: Statistic. It must be a vectorized callable
                that accepts a NumPy array as the first argument and returns
                the resulting statistic.
                It must also accept a keyword argument `axis` and be vectorized
                to compute the statistic along the provided axis.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than the mean
                    in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

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

        Parameter defaults:
            Defaults for parameters `alternative`, `confidence_level`,
            and `n_resamples` can be changed using the
            `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Bootstrapping (statistics) &#8212; Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
            - [scipy.stats.bootstrap &#8212; SciPy Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html#scipy-stats-bootstrap).

        Examples:
            ```pycon
            >>> import numpy as np
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Bootstrap("orders", np.mean, random_state=42),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> print(result)
                     metric control treatment rel_effect_size rel_effect_size_ci pvalue
            orders_per_user   0.530     0.573            8.0%       [-1.8%, 19%]      -

            ```

            With multiple columns:

            ```pycon
            >>> def ratio_of_means(sample, axis):
            ...     means = np.mean(sample, axis=axis)
            ...     return means[0] / means[1]

            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.Bootstrap(
            ...         ("orders", "sessions"),
            ...         ratio_of_means,
            ...         random_state=42,
            ...     ),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> print(result)
                        metric control treatment rel_effect_size rel_effect_size_ci pvalue
            orders_per_session   0.266     0.289            8.8%      [-0.61%, 20%]      -

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

        self.n_resamples = (
            tea_tasting.utils.auto_check(n_resamples, "n_resamples")
            if n_resamples is not None
            else tea_tasting.config.get_config("n_resamples")
        )

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


    def analyze_granular(
        self,
        control: pa.Table,
        treatment: pa.Table,
    ) -> BootstrapResult:
        """Analyze metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        def statistic(
            contr: npt.NDArray[np.number],
            treat: npt.NDArray[np.number],
            axis: int = -1,
        ) -> npt.NDArray[np.number]:
            contr_stat = self.statistic(contr, axis=axis)
            treat_stat = self.statistic(treat, axis=axis)

            effect_size = treat_stat - contr_stat
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_effect_size = np.divide(treat_stat, contr_stat) - 1

            return np.stack((effect_size, rel_effect_size), axis=0)

        contr = _select_as_numpy(control, self.columns)
        treat = _select_as_numpy(treatment, self.columns)
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
            random_state=self.random_state,  # type: ignore
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


def _select_as_numpy(
    data: pa.Table,
    columns: str | Sequence[str],
) -> npt.NDArray[np.number]:
    if isinstance(columns, str):
        return data[columns].combine_chunks().to_numpy(zero_copy_only=False)
    return np.column_stack([
        data[col].combine_chunks().to_numpy(zero_copy_only=False)
        for col in columns
    ])


class Quantile(Bootstrap):  # noqa: D101
    def __init__(
        self,
        column: str,
        q: float = 0.5,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        n_resamples: int | None = None,
        method: Literal["percentile", "basic", "bca"] = "basic",
        batch: int | None = None,
        random_state: int | np.random.Generator | np.random.SeedSequence | None = None,
    ) -> None:
        """Metric for the analysis of quantiles using bootstrap resampling.

        Args:
            column: Name of the column for the quantiles to compute.
            q: Probability for the quantiles to compute.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than the mean
                    in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

            confidence_level: Confidence level for the confidence interval.
            n_resamples: The number of resamples performed to form
                the bootstrap distribution of the statistic.
            method: Whether to return the "percentile" bootstrap confidence
                interval (`"percentile"`), the "basic" (AKA "reverse") bootstrap
                confidence interval (`"basic"`), or the bias-corrected
                and accelerated bootstrap confidence interval (`"bca"`).

                Default method is "basic" which is different from default
                method "bca" in `Bootstrap`. The "bca" confidence intervals cannot
                be calculated when the bootstrap distribution is degenerate
                (e.g. all elements are identical). This is often the case for the
                quantile metrics.

            batch: The number of resamples to process in each vectorized call
                to statistic. Memory usage is O(`batch * n`), where `n` is
                the sample size. Default is `None`, in which case `batch = n_resamples`
                (or `batch = max(n_resamples, n)` for method="bca").
            random_state: Pseudorandom number generator state used
                to generate resamples.

        Parameter defaults:
            Defaults for parameters `alternative`, `confidence_level`,
            and `n_resamples` can be changed using the
            `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     revenue_per_user_p80=tt.Quantile("revenue", 0.8, random_state=42),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> print(result)
                          metric control treatment rel_effect_size rel_effect_size_ci pvalue
            revenue_per_user_p80    10.6      11.6            9.1%       [-1.2%, 21%]      -

            ```
        """  # noqa: E501
        self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
        self.q = tea_tasting.utils.check_scalar(q, "q", typ=float, ge=0, le=1)
        super().__init__(
            columns=column,
            statistic=functools.partial(np.nanquantile, q=q),
            alternative=alternative,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=method,
            batch=batch,
            random_state=random_state,
        )
