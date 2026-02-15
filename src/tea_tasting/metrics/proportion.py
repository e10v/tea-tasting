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
        effect_size_ci_lower: Lower bound of the absolute effect size
            confidence interval.
            Calculated only for the normal approximation (`method="norm"`).
            `nan` for other methods.
        effect_size_ci_upper: Upper bound of the absolute effect size
            confidence interval.
            Calculated only for the normal approximation (`method="norm"`).
            `nan` for other methods.
        rel_effect_size: Relative effect size. Difference between the two proportions,
            divided by the control proportion.
        rel_effect_size_ci_lower: Lower bound of the relative effect size
            confidence interval.
            Calculated only for the normal approximation (`method="norm"`).
            `nan` for other methods.
        rel_effect_size_ci_upper: Upper bound of the relative effect size
            confidence interval.
            Calculated only for the normal approximation (`method="norm"`).
            `nan` for other methods.
        pvalue: P-value.
    """
    control: float
    treatment: float
    effect_size: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    rel_effect_size: float
    rel_effect_size_ci_lower: float
    rel_effect_size_ci_upper: float
    pvalue: float


class _ZTestResult(NamedTuple):
    pvalue: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    rel_effect_size_ci_lower: float
    rel_effect_size_ci_upper: float


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
        confidence_level: float | None = None,
        correction: bool | None = None,
        equal_var: bool = True,
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
                    Confidence intervals for absolute and relative effect size are
                    calculated only for this method.
                - `"pearson"`: Pearson's chi-squared test.

            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than
                    the mean in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

                G-test and Pearson's chi-squared test are always two-sided.
            confidence_level: Confidence level for the confidence intervals.
                Used only with the normal approximation (`method="norm"`).
            correction: If `True`, add continuity correction. Only for
                approximate methods: normal, G-test, and Pearson's chi-squared test.
                Defaults to the global config value (`True`).
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two proportions. Only for normal approximation
                and for Barnard's exact test (in the Wald statistic).
                Default is `True`. Global config is ignored as pooled variance is
                optimal for proportion tests.

        Parameter defaults:
            Defaults for parameters `alternative`, `confidence_level`, and `correction`
            can be changed using the `config_context` and `set_config` functions.
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
            prop_users_with_orders   0.300     0.356             19%       [-1.3%, 43%] 0.0693

            ```

            With specific statistical test:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     prop_users_with_orders=tt.Proportion(
            ...         "has_order",
            ...         method="barnard",
            ...         equal_var=False,
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
        self.confidence_level = (
            tea_tasting.utils.auto_check(confidence_level, "confidence_level")
            if confidence_level is not None
            else tea_tasting.config.get_config("confidence_level")
        )
        if self.alternative != "two-sided" and method in {"log-likelihood", "pearson"}:
            raise ValueError(
                f"The {method} method supports only two-sided alternative hypothesis.")
        self.correction = (
            tea_tasting.utils.auto_check(correction, "correction")
            if correction is not None
            else tea_tasting.config.get_config("correction")
        )
        self.equal_var = tea_tasting.utils.auto_check(equal_var, "equal_var")


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

        effect_size_ci_lower = float("nan")
        effect_size_ci_upper = float("nan")
        rel_effect_size_ci_lower = float("nan")
        rel_effect_size_ci_upper = float("nan")

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
            norm_result = self._2sample_proportion_ztest(
                p_contr=p_contr,
                p_treat=p_treat,
                n_contr=n_contr,
                n_treat=n_treat,
            )
            pvalue = norm_result.pvalue
            effect_size_ci_lower = norm_result.effect_size_ci_lower
            effect_size_ci_upper = norm_result.effect_size_ci_upper
            rel_effect_size_ci_lower = norm_result.rel_effect_size_ci_lower
            rel_effect_size_ci_upper = norm_result.rel_effect_size_ci_upper

        return ProportionResult(
            control=p_contr,
            treatment=p_treat,
            effect_size=p_treat - p_contr,
            effect_size_ci_lower=effect_size_ci_lower,
            effect_size_ci_upper=effect_size_ci_upper,
            rel_effect_size=p_treat/p_contr - 1,
            rel_effect_size_ci_lower=rel_effect_size_ci_lower,
            rel_effect_size_ci_upper=rel_effect_size_ci_upper,
            pvalue=pvalue,
        )


    def _2sample_proportion_ztest(
        self,
        *,
        p_contr: float,
        p_treat: float,
        n_contr: int,
        n_treat: int,
    ) -> _ZTestResult:
        distr, log_distr = self._2sample_proportion_distr(
            p_contr=p_contr,
            p_treat=p_treat,
            n_contr=n_contr,
            n_treat=n_treat,
        )
        diff = p_treat - p_contr
        p_ratio = p_treat / p_contr
        cc = 0.5 * (1/n_contr + 1/n_treat) if self.correction else 0
        log_cc = cc / p_contr if self.correction else 0

        if self.alternative == "greater":
            pvalue_diff = max(diff - cc, 0) if self.correction and diff > 0 else diff
            pvalue = distr.sf(pvalue_diff)
            q = self.confidence_level

            effect_size_ci_lower = diff + distr.isf(q) - cc
            effect_size_ci_upper = 1

            p_ratio_ci_lower = p_ratio * math.exp(log_distr.isf(q) - log_cc)
            p_ratio_ci_upper = float("inf")
        elif self.alternative == "less":
            pvalue_diff = min(diff + cc, 0) if self.correction and diff < 0 else diff
            pvalue = distr.cdf(pvalue_diff)
            q = self.confidence_level

            effect_size_ci_lower = -1
            effect_size_ci_upper = diff + distr.ppf(q) + cc

            p_ratio_ci_lower = 0
            p_ratio_ci_upper = p_ratio * math.exp(log_distr.ppf(q) + log_cc)
        else:  # two-sided
            pvalue_diff = abs(diff)
            if self.correction and pvalue_diff > 0:
                pvalue_diff = max(pvalue_diff - cc, 0)
            pvalue = 2 * distr.sf(pvalue_diff)
            q = (1 + self.confidence_level) / 2

            effect_half_ci = distr.ppf(q) + cc
            effect_size_ci_lower = diff - effect_half_ci
            effect_size_ci_upper = diff + effect_half_ci

            rel_half_ci = math.exp(log_distr.ppf(q) + log_cc)
            p_ratio_ci_lower = p_ratio / rel_half_ci
            p_ratio_ci_upper = p_ratio * rel_half_ci

        effect_size_ci_lower = max(effect_size_ci_lower, -1)
        effect_size_ci_upper = min(effect_size_ci_upper, 1)

        return _ZTestResult(
            pvalue=pvalue,
            effect_size_ci_lower=effect_size_ci_lower,
            effect_size_ci_upper=effect_size_ci_upper,
            rel_effect_size_ci_lower=p_ratio_ci_lower - 1,
            rel_effect_size_ci_upper=p_ratio_ci_upper - 1,
        )


    def _2sample_proportion_distr(
        self,
        *,
        p_contr: float,
        p_treat: float,
        n_contr: int,
        n_treat: int,
    ) -> tuple[scipy.stats.rv_frozen, scipy.stats.rv_frozen]:
        if self.equal_var:
            p_pooled = (p_contr*n_contr + p_treat*n_treat) / (n_contr + n_treat)
            scale = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_contr + 1/n_treat))
            log_scale = scale / p_pooled
            return scipy.stats.norm(scale=scale), scipy.stats.norm(scale=log_scale)

        scale = math.sqrt(
            p_contr*(1 - p_contr)/n_contr + p_treat*(1 - p_treat)/n_treat)
        log_scale = math.sqrt(
            (1 - p_contr) / (n_contr*p_contr) +
            (1 - p_treat) / (n_treat*p_treat),
        )
        return scipy.stats.norm(scale=scale), scipy.stats.norm(scale=log_scale)


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
                Defaults to the global config value (`True`).

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
