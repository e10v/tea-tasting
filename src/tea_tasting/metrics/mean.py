"""Metrics for the analysis of means."""
# ruff: noqa: PD901

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import scipy.stats

import tea_tasting.aggr
import tea_tasting.config
from tea_tasting.metrics.base import AggrCols, MetricBaseAggregated
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Literal


class MeanResult(NamedTuple):
    """Result of the analysis of means.

    Attributes:
        control: Control mean.
        treatment: Treatment mean.
        effect_size: Absolute effect size. Difference between the two means.
        effect_size_ci_lower: Lower bound of the absolute effect size
            confidence interval.
        effect_size_ci_upper: Upper bound of the absolute effect size
            confidence interval.
        rel_effect_size: Relative effect size. Difference between the two means,
            divided by the control mean.
        rel_effect_size_ci_lower: Lower bound of the relative effect size
            confidence interval.
        rel_effect_size_ci_upper: Upper bound of the relative effect size
            confidence interval.
        pvalue: P-value
        statistic: Statistic (standardized effect size).
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
    statistic: float


class RatioOfMeans(MetricBaseAggregated[MeanResult]):  # noqa: D101
    def __init__(  # noqa: PLR0913
        self,
        numer: str,
        denom: str | None = None,
        numer_covariate: str | None = None,
        denom_covariate: str | None = None,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
    ) -> None:
        """Metric for the analysis of ratios of means.

        Args:
            numer: Numerator column name.
            denom: Denominator column name.
            numer_covariate: Covariate numerator column name.
            denom_covariate: Covariate denominator column name.
            alternative: Alternative hypothesis.
            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).

        Alternative hypothesis options:
            - `"two-sided"`: the means are unequal,
            - `"greater"`: the mean in the treatment variant is greater than the mean
                in the control variant,
            - `"less"`: the mean in the treatment variant is less than the mean
                in the control variant.

        Parameter defaults:
            Defaults for the parameters `alternative`, `confidence_level`,
            `equal_var`, and `use_t` can be changed using the
            `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #> orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            ```

            With CUPED:

            ```python
            experiment = tt.Experiment(
                orders_per_session=tt.RatioOfMeans(
                    "orders",
                    "sessions",
                    "orders_covariate",
                    "sessions_covariate",
                ),
            )

            data = tt.make_users_data(seed=42, covariates=True)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci  pvalue
            #> orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229
            ```
        """  # noqa: E501
        self.numer = tea_tasting.utils.check_scalar(numer, "numer", typ=str)
        self.denom = tea_tasting.utils.check_scalar(denom, "denom", typ=str | None)
        self.numer_covariate = tea_tasting.utils.check_scalar(
            numer_covariate, "numer_covariate", typ=str | None)
        self.denom_covariate = tea_tasting.utils.check_scalar(
            denom_covariate, "denom_covariate", typ=str | None)
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
        self.equal_var = (
            tea_tasting.utils.auto_check(equal_var, "equal_var")
            if equal_var is not None
            else tea_tasting.config.get_config("equal_var")
        )
        self.use_t = (
            tea_tasting.utils.auto_check(use_t, "use_t")
            if use_t is not None
            else tea_tasting.config.get_config("use_t")
        )


    @property
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        cols = tuple(
            col for col in (
                self.numer,
                self.denom,
                self.numer_covariate,
                self.denom_covariate,
            )
            if col is not None
        )
        return AggrCols(
            has_count=True,
            mean_cols=cols,
            var_cols=cols,
            cov_cols=tuple(
                (col0, col1)
                for col0 in cols
                for col1 in cols
                if col0 < col1
            ),
        )


    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> MeanResult:
        """Analyze a metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        control = control.with_zero_div()
        treatment = treatment.with_zero_div()
        total = control + treatment
        covariate_coef = self._covariate_coef(total)
        covariate_mean = total.mean(self.numer_covariate) / total.mean(
            self.denom_covariate)
        return self._analyze_stats(
            contr_mean=self._metric_mean(control, covariate_coef, covariate_mean),
            contr_var=self._metric_var(control, covariate_coef),
            contr_count=control.count(),
            treat_mean=self._metric_mean(treatment, covariate_coef, covariate_mean),
            treat_var=self._metric_var(treatment, covariate_coef),
            treat_count=treatment.count(),
        )


    def _covariate_coef(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        covariate_var = aggr.ratio_var(self.numer_covariate, self.denom_covariate)
        if covariate_var == 0:
            return 0
        return self._covariate_cov(aggr) / covariate_var


    def _covariate_cov(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        return aggr.ratio_cov(
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        )


    def _metric_mean(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
        covariate_mean: float,
    ) -> float:
        value = aggr.mean(self.numer) / aggr.mean(self.denom)
        covariate = aggr.mean(self.numer_covariate) / aggr.mean(self.denom_covariate)
        return value - covariate_coef*(covariate - covariate_mean)


    def _metric_var(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
    ) -> float:
        var = aggr.ratio_var(self.numer, self.denom)
        covariate_var = aggr.ratio_var(self.numer_covariate, self.denom_covariate)
        covariate_cov = self._covariate_cov(aggr)
        return (
            var
            + covariate_coef * covariate_coef * covariate_var
            - 2 * covariate_coef * covariate_cov
        )


    def _analyze_stats(
        self,
        contr_mean: float,
        contr_var: float,
        contr_count: int,
        treat_mean: float,
        treat_var: float,
        treat_count: int,
    ) -> MeanResult:
        scale, distr = self._scale_and_distr(
            contr_var=contr_var,
            contr_count=contr_count,
            treat_var=treat_var,
            treat_count=treat_count,
        )
        log_scale, log_distr = self._scale_and_distr(
            contr_var=contr_var / contr_mean / contr_mean,
            contr_count=contr_count,
            treat_var=treat_var / treat_mean / treat_mean,
            treat_count=treat_count,
        )

        means_ratio = treat_mean / contr_mean
        effect_size = treat_mean - contr_mean
        statistic = effect_size / scale

        if self.alternative == "greater":
            q = self.confidence_level
            effect_size_ci_lower = effect_size + scale*distr.isf(q)
            means_ratio_ci_lower = means_ratio * math.exp(log_scale * log_distr.isf(q))
            effect_size_ci_upper = means_ratio_ci_upper = float("+inf")
            pvalue = distr.sf(statistic)
        elif self.alternative == "less":
            q = self.confidence_level
            effect_size_ci_lower = means_ratio_ci_lower = float("-inf")
            effect_size_ci_upper = effect_size + scale*distr.ppf(q)
            means_ratio_ci_upper = means_ratio * math.exp(log_scale * log_distr.ppf(q))
            pvalue = distr.cdf(statistic)
        else:  # two-sided
            q = (1 + self.confidence_level) / 2
            half_ci = scale * distr.ppf(q)
            effect_size_ci_lower = effect_size - half_ci
            effect_size_ci_upper = effect_size + half_ci

            rel_half_ci = math.exp(log_scale * log_distr.ppf(q))
            means_ratio_ci_lower = means_ratio / rel_half_ci
            means_ratio_ci_upper = means_ratio * rel_half_ci

            pvalue = 2 * distr.sf(abs(statistic))

        return MeanResult(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=effect_size,
            effect_size_ci_lower=effect_size_ci_lower,
            effect_size_ci_upper=effect_size_ci_upper,
            rel_effect_size=means_ratio - 1,
            rel_effect_size_ci_lower=means_ratio_ci_lower - 1,
            rel_effect_size_ci_upper=means_ratio_ci_upper - 1,
            pvalue=pvalue,
            statistic=statistic,
        )


    def _scale_and_distr(
        self,
        contr_var: float,
        contr_count: int,
        treat_var: float,
        treat_count: int,
    ) -> tuple[float, scipy.stats.rv_frozen]:
        if self.equal_var:
            pooled_var = (
                (contr_count - 1)*contr_var + (treat_count - 1)*treat_var
            ) / (contr_count + treat_count - 2)
            scale = math.sqrt(pooled_var/contr_count + pooled_var/treat_count)
        else:
            scale = math.sqrt(contr_var/contr_count + treat_var/treat_count)

        if self.use_t:
            if self.equal_var:
                df = contr_count + treat_count - 2
            else:
                contr_mean_var = contr_var / contr_count
                treat_mean_var = treat_var / treat_count
                df = (contr_mean_var + treat_mean_var)**2 / (
                    contr_mean_var**2 / (contr_count - 1)
                    + treat_mean_var**2 / (treat_count - 1)
                )
            distr = scipy.stats.t(df=df)
        else:
            distr = scipy.stats.norm()

        return scale, distr


class Mean(RatioOfMeans):  # noqa: D101
    def __init__(
        self,
        value: str,
        covariate: str | None = None,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
    ) -> None:
        """Metric for the analysis of means.

        Args:
            value: Metric value column name.
            covariate: Metric covariate column name.
            alternative: Alternative hypothesis.
            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).

        Alternative hypothesis options:
            - `"two-sided"`: the means are unequal,
            - `"greater"`: the mean in the treatment variant is greater than the mean
                in the control variant,
            - `"less"`: the mean in the treatment variant is less than the mean
                in the control variant.

        Parameter defaults:
            Defaults for the parameters `alternative`, `confidence_level`,
            `equal_var`, and `use_t` can be changed using the
            `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>           metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #>  orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            #> revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
            ```

            With CUPED:

            ```python
            experiment = tt.Experiment(
                orders_per_user=tt.Mean("orders", "orders_covariate"),
                revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
            )

            data = tt.make_users_data(seed=42, covariates=True)
            result = experiment.analyze(data)
            print(result)
            #>           metric control treatment rel_effect_size rel_effect_size_ci  pvalue
            #>  orders_per_user   0.523     0.581             11%        [2.9%, 20%] 0.00733
            #> revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00675
            ```
        """  # noqa: E501
        super().__init__(
            numer=value,
            denom=None,
            numer_covariate=covariate,
            denom_covariate=None,
            alternative=alternative,
            confidence_level=confidence_level,
            equal_var=equal_var,
            use_t=use_t,
        )
        self.value = value
        self.covariate = covariate
