"""Metrics for the analysis of proportions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.metrics
from tea_tasting.metrics.base import AggrCols, MetricBaseAggregated
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Literal

    import ibis.expr.types
    import narwhals.typing


_MAX_EXACT_THRESHOLD = 1000


class ProportionResult(NamedTuple):
    """Result of the analysis of proportions.

    Attributes:
        control: Proportion in control.
        treatment: Proportion in treatment.
        effect_size: Absolute effect size. Difference between the two proportions.
        rel_effect_size: Relative effect size. Difference between the two proportions,
            divided by the control proportion.
        pvalue: P-value.
    """
    control: float
    treatment: float
    effect_size: float
    rel_effect_size: float
    pvalue: float


class Proportion(MetricBaseAggregated[ProportionResult]):  # noqa: D101
    def __init__(
        self,
        column: str,
        *,
        method: Literal[
            "auto",
            "barnard",
            "boschloo",
            "fisher",
            "log-likelihood",
            "norm",
            "pearson",
        ] = "auto",
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        correction: bool | None = None,
        equal_var: bool | None = None,
    ) -> None:
        """Metric for the analysis of proportions.

        Args:
            column: Metric column name. Column values should be 0 or 1.
            method: Statistical test used for calculation of p-value:

                - `"auto"`: Barnard's exact test if the total number
                    of observations is < 1000; or normal approximation otherwise.
                - `"barnard"`: Barnard's exact test using the Wald statistic.
                - `"boschloo"`: Boschloo's exact test.
                    Also known as Barnard's exact test with
                    p-value of Fisher's exact test as a statistic.
                - `"fisher"`: Fisher's exact test.
                - `"log-likelihood"`: G-test.
                - `"norm"`: Normal approximation of the binomial distribution.
                    Also known as two-sample proportion Z-test.
                - `"pearson"`: Pearson's chi-squared test.

            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than
                    the mean in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

                G-test and Pearson's chi-squared test are always two-sided.
            correction: If `True`, add continuity correction. Only for
                approximate methods: normal, G-test, and Pearson's chi-squared test.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two proportions. Only for normal approximation
                and for Barnard's exact test (in the Wald statistic).

        Parameter defaults:
            Defaults for parameters `alternative`, `correction`, and `equal_var`
            can be changed using the `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Barnard's test](https://en.wikipedia.org/wiki/Barnard%27s_test).
            - [Boschloo's test](https://en.wikipedia.org/wiki/Boschloo%27s_test).
            - [Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test).
            - [G-test](https://en.wikipedia.org/wiki/G-test).
            - [Two-proportion Z-test](https://en.wikipedia.org/wiki/Two-proportion_Z-test).
            - [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test).

        Examples:
            ```pycon
            >>> import pyarrow as pa
            >>> import pyarrow.compute as pc
            >>> import tea_tasting as tt

            >>> data = tt.make_users_data(seed=42, n_users=1000)
            >>> data = data.append_column(
            ...     "has_order",
            ...     pc.greater(data["orders"], 0).cast(pa.int64()),
            ... )
            >>> data
            pyarrow.Table
            user: int64
            variant: int64
            sessions: int64
            orders: int64
            revenue: double
            has_order: int64
            ----
            user: [[0,1,2,3,4,...,995,996,997,998,999]]
            variant: [[1,0,1,1,0,...,0,1,0,1,0]]
            sessions: [[1,2,1,2,2,...,1,2,4,2,2]]
            orders: [[0,0,0,2,1,...,1,1,0,0,2]]
            revenue: [[0,0,0,16.57,8.87,...,8.54,11.78,0,0,18.69]]
            has_order: [[0,0,0,1,1,...,1,1,0,0,1]]
            >>> experiment = tt.Experiment(
            ...     prop_users_with_orders=tt.Proportion("has_order"),
            ... )
            >>> experiment.analyze(data)
                            metric control treatment rel_effect_size rel_effect_size_ci pvalue
            prop_users_with_orders   0.300     0.356             19%             [-, -] 0.0689

            ```

            With specific statistical test:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     prop_users_with_orders=tt.Proportion(
            ...         "has_order",
            ...         method="barnard",
            ...         equal_var=True,
            ...     ),
            ... )
            >>> experiment.analyze(data)
                            metric control treatment rel_effect_size rel_effect_size_ci pvalue
            prop_users_with_orders   0.300     0.356             19%             [-, -] 0.0620

            ```
        """  # noqa: E501
        self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
        self.method = tea_tasting.utils.check_scalar(method, "method", typ=str, in_={
            "auto",
            "barnard",
            "boschloo",
            "fisher",
            "log-likelihood",
            "norm",
            "pearson",
        })
        self.alternative: Literal["two-sided", "greater", "less"] = (
            tea_tasting.utils.auto_check(alternative, "alternative")
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )
        if self.alternative != "two-sided" and method in {"log-likelihood", "pearson"}:
            raise ValueError(
                f"The {method} method supports only two-sided alternative hypothesis.")
        self.correction = (
            tea_tasting.utils.auto_check(correction, "correction")
            if correction is not None
            else tea_tasting.config.get_config("correction")
        )
        self.equal_var = (
            tea_tasting.utils.auto_check(equal_var, "equal_var")
            if equal_var is not None
            else tea_tasting.config.get_config("equal_var")
        )


    @property
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        return tea_tasting.metrics.AggrCols(
            has_count=True,
            mean_cols=(self.column,),
        )


    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> ProportionResult:
        """Analyze a metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        control = control.with_zero_div()
        treatment = treatment.with_zero_div()
        p_contr = control.mean(self.column)
        p_treat = treatment.mean(self.column)
        n_contr = control.count()
        n_treat = treatment.count()

        method = self.method
        if method == "auto":
            method = "barnard" if n_contr + n_treat < _MAX_EXACT_THRESHOLD else "norm"

        if method != "norm":
            data = np.empty(shape=(2, 2), dtype=np.int64)
            data[0, 0] = round(n_treat * p_treat)
            data[0, 1] = round(n_contr * p_contr)
            data[1, 0] = n_treat - data[0, 0]
            data[1, 1] = n_contr - data[0, 1]

        if method == "barnard":
            pvalue = scipy.stats.barnard_exact(
                data,  # type: ignore
                alternative=self.alternative,
                pooled=self.equal_var,
            ).pvalue
        elif method == "boschloo":
            pvalue = scipy.stats.boschloo_exact(
                data, alternative=self.alternative).pvalue  # type: ignore
        elif method == "fisher":
            pvalue = scipy.stats.fisher_exact(data, alternative=self.alternative).pvalue  # type: ignore
        elif method in {"log-likelihood", "pearson"}:
            if np.any(data.sum(axis=0) == 0) or np.any(data.sum(axis=1) == 0):  # type: ignore
                pvalue = float("nan")
            else:
                pvalue = scipy.stats.chi2_contingency(
                    data,  # type: ignore
                    correction=self.correction,
                    lambda_=self.method,
                ).pvalue  # type: ignore
        else:  # norm
            pvalue = _2sample_proportion_ztest(
                p_contr=p_contr,
                p_treat=p_treat,
                n_contr=n_contr,
                n_treat=n_treat,
                alternative=self.alternative,
                correction=self.correction,
                equal_var=self.equal_var,
            )

        return ProportionResult(
            control=p_contr,
            treatment=p_treat,
            effect_size=p_treat - p_contr,
            rel_effect_size=p_treat/p_contr - 1,
            pvalue=pvalue,
        )


def _2sample_proportion_ztest(
    *,
    p_contr: float,
    p_treat: float,
    n_contr: int,
    n_treat: int,
    alternative: Literal["two-sided", "greater", "less"],
    correction: bool,
    equal_var: bool,
) -> float:
    if equal_var:
        p_pooled = (p_contr*n_contr + p_treat*n_treat) / (n_contr + n_treat)
        scale = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_contr + 1/n_treat))
    else:
        scale = math.sqrt(p_contr*(1 - p_contr)/n_contr + p_treat*(1 - p_treat)/n_treat)

    distr = scipy.stats.norm(scale=scale)
    diff = p_treat - p_contr
    cc = 0.5 * (1/n_contr + 1/n_treat) if correction else 0

    if alternative == "greater":
        if correction and diff > 0:
            diff = max(diff - cc, 0)
        pvalue = distr.sf(diff)
    elif alternative == "less":
        if correction and diff < 0:
            diff = min(diff + cc, 0)
        pvalue = distr.cdf(diff)
    else:  # two-sided
        diff = abs(diff)
        if correction and diff > 0:
            diff = max(diff - cc, 0)
        pvalue = 2 * distr.sf(diff)

    return pvalue


class SampleRatioResult(NamedTuple):
    """Result of the sample ratio mismatch check.

    Attributes:
        control: Number of observations in control.
        treatment: Number of observations in treatment.
        pvalue: P-value.
    """
    control: float
    treatment: float
    pvalue: float


class SampleRatio(MetricBaseAggregated[SampleRatioResult]):  # noqa: D101
    def __init__(
        self,
        ratio: float | int | dict[object, float | int] = 1,
        *,
        method: Literal["auto", "binom", "norm"] = "auto",
        correction: bool | None = None,
    ) -> None:
        """Metric for sample ratio mismatch check.

        Args:
            ratio: Expected ratio of the number of observations in the treatment
                relative to the control.
            method: Statistical test used for calculation of p-value:

                - `"auto"`: Exact binomial test if the total number
                    of observations is < 1000; or normal approximation otherwise.
                - `"binom"`: Exact binomial test.
                - `"norm"`: Normal approximation of the binomial distribution.

            correction: If `True`, add continuity correction.
                Only for normal approximation.

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     sample_ratio=tt.SampleRatio(),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> result.with_keys(("metric", "control", "treatment", "pvalue"))
                  metric control treatment pvalue
            sample_ratio    2023      1977  0.477

            ```

            Different expected ratio:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     sample_ratio=tt.SampleRatio(0.5),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> result.with_keys(("metric", "control", "treatment", "pvalue"))
                  metric control treatment    pvalue
            sample_ratio    2023      1977 3.26e-103

            ```
        """
        if isinstance(ratio, dict):
            for val in ratio.values():
                tea_tasting.utils.auto_check(val, "ratio")
        else:
            tea_tasting.utils.auto_check(ratio, "ratio")
        self.ratio = ratio

        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, in_={"auto", "binom", "norm"})
        self.correction = (
            tea_tasting.utils.auto_check(correction, "correction")
            if correction is not None
            else tea_tasting.config.get_config("correction")
        )


    @property
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        return AggrCols(has_count=True)


    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table | dict[
            object, tea_tasting.aggr.Aggregates],
        control: object,
        treatment: object,
        variant: str | None = None,
    ) -> SampleRatioResult:
        """Perform a sample ratio mismatch check.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        tea_tasting.utils.check_scalar(variant, "variant", typ=str | None)
        aggr = tea_tasting.metrics.aggregate_by_variants(
            data,
            aggr_cols=self.aggr_cols,
            variant=variant,
        )

        k = aggr[treatment].count()
        n = k + aggr[control].count()

        r = (
            self.ratio
            if isinstance(self.ratio, float | int)
            else self.ratio[treatment] / self.ratio[control]
        )
        p = r / (1 + r)

        if (
            self.method == "binom" or
            (self.method == "auto" and n < _MAX_EXACT_THRESHOLD)
        ):
            pvalue = scipy.stats.binomtest(k=int(k), n=int(n), p=p).pvalue
        else:  # norm
            d = abs(k - n*p)
            if self.correction and d > 0:
                d = max(d - 0.5, 0)
            z = d / math.sqrt(n * p * (1 - p))
            pvalue = 2 * scipy.stats.norm.sf(z)

        return SampleRatioResult(
            control=n - k,
            treatment=k,
            pvalue=pvalue,  # type: ignore
        )


    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> SampleRatioResult:
        """Stub method for compatibility with the base class."""
        raise NotImplementedError
