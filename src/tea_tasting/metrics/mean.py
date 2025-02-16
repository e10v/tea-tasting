"""Metrics for the analysis of means."""
# ruff: noqa: PD901

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING, NamedTuple, overload

import scipy.optimize
import scipy.stats

import tea_tasting.aggr
import tea_tasting.config
from tea_tasting.metrics.base import (
    AggrCols,
    MetricBaseAggregated,
    MetricPowerResults,
    PowerBaseAggregated,
)
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeAlias, TypeVar


    N = TypeVar("N", bound=float | int | None)


MAX_ITER = 100


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


class MeanPowerResult(NamedTuple):
    """Power analysis results.

    Attributes:
        power: Statistical power.
        effect_size: Absolute effect size. Difference between the two means.
        rel_effect_size: Relative effect size. Difference between the two means,
            divided by the control mean.
        n_obs: Number of observations in the control and in the treatment together.
    """
    power: float
    effect_size: float
    rel_effect_size: float
    n_obs: float

MeanPowerResults: TypeAlias = MetricPowerResults[MeanPowerResult]


class RatioOfMeans(  # noqa: D101
    MetricBaseAggregated[MeanResult],
    PowerBaseAggregated[MeanPowerResults],
):
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
        alpha: float | None = None,
        ratio: float | int | None = None,
        power: float | None = None,
        effect_size: float | int | Sequence[float | int] | None = None,
        rel_effect_size: float | Sequence[float] | None = None,
        n_obs: int | Sequence[int] | None = None,
    ) -> None:
        """Metric for the analysis of ratios of means.

        Args:
            numer: Numerator column name.
            denom: Denominator column name.
            numer_covariate: Covariate numerator column name.
            denom_covariate: Covariate denominator column name.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than the mean
                    in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).
            alpha: Significance level. Only for the analysis of power.
            ratio: Ratio of the number of observations in the treatment
                relative to the control. Only for the analysis of power.
            power: Statistical power. Only for the analysis of power.
            effect_size: Absolute effect size. Difference between the two means.
                Only for the analysis of power.
            rel_effect_size: Relative effect size. Difference between the two means,
                divided by the control mean. Only for the analysis of power.
            n_obs: Number of observations in the control and in the treatment together.
                Only for the analysis of power.

        Parameter defaults:
            Defaults for parameters `alpha`, `alternative`, `confidence_level`,
            `equal_var`, `n_obs`, `power`, `ratio`, and `use_t` can be changed
            using the `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> print(result)
                        metric control treatment rel_effect_size rel_effect_size_ci pvalue
            orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762

            ```

            With CUPED:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     orders_per_session=tt.RatioOfMeans(
            ...         "orders",
            ...         "sessions",
            ...         "orders_covariate",
            ...         "sessions_covariate",
            ...     ),
            ... )
            >>> data = tt.make_users_data(seed=42, covariates=True)
            >>> result = experiment.analyze(data)
            >>> print(result)
                        metric control treatment rel_effect_size rel_effect_size_ci  pvalue
            orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     seed=42,
            ...     sessions_uplift=0,
            ...     orders_uplift=0,
            ...     revenue_uplift=0,
            ...     covariates=True,
            ... )
            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     n_obs=(10_000, 20_000),
            ... )
            >>> # Solve for effect size.
            >>> print(orders_per_session.solve_power(data))
            power effect_size rel_effect_size n_obs
              80%      0.0177            6.8% 10000
              80%      0.0125            4.8% 20000

            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     rel_effect_size=0.05,
            ... )
            >>> # Solve for the total number of observations.
            >>> print(orders_per_session.solve_power(data, "n_obs"))
            power effect_size rel_effect_size n_obs
              80%      0.0130            5.0% 18515

            >>> orders_per_session = tt.RatioOfMeans(
            ...     "orders",
            ...     "sessions",
            ...     "orders_covariate",
            ...     "sessions_covariate",
            ...     rel_effect_size=0.1,
            ... )
            >>> # Solve for power. Infer number of observations from the sample.
            >>> print(orders_per_session.solve_power(data, "power"))
            power effect_size rel_effect_size n_obs
              74%      0.0261             10%  4000

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
        self.alpha = (
            tea_tasting.utils.auto_check(alpha, "alpha")
            if alpha is not None
            else tea_tasting.config.get_config("alpha")
        )
        self.ratio = (
            tea_tasting.utils.auto_check(ratio, "ratio")
            if ratio is not None
            else tea_tasting.config.get_config("ratio")
        )
        self.power = (
            tea_tasting.utils.auto_check(power, "power")
            if power is not None
            else tea_tasting.config.get_config("power")
        )
        if effect_size is not None and rel_effect_size is not None:
            raise ValueError(
                "Both `effect_size` and `rel_effect_size` are not `None`. "
                "Only one of them should be defined.",
            )
        if isinstance(effect_size, Sequence):
            for x in effect_size:
                tea_tasting.utils.check_scalar(
                    x, "effect_size", typ=float | int,
                    gt=float("-inf"), lt=float("inf"), ne=0,
                )
        elif effect_size is not None:
            tea_tasting.utils.check_scalar(
                effect_size, "effect_size", typ=float | int,
                gt=float("-inf"), lt=float("inf"), ne=0,
            )
        self.effect_size = effect_size
        if isinstance(rel_effect_size, Sequence):
            for x in rel_effect_size:
                tea_tasting.utils.check_scalar(
                    x, "rel_effect_size", typ=float | int,
                    gt=float("-inf"), lt=float("inf"), ne=0,
                )
        elif rel_effect_size is not None:
            tea_tasting.utils.check_scalar(
                rel_effect_size, "rel_effect_size", typ=float | int,
                gt=float("-inf"), lt=float("inf"), ne=0,
            )
        self.rel_effect_size = rel_effect_size
        self.n_obs = (
            tea_tasting.utils.auto_check(n_obs, "n_obs")
            if n_obs is not None
            else tea_tasting.config.get_config("n_obs")
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


    def solve_power_from_aggregates(
        self,
        data: tea_tasting.aggr.Aggregates,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> MeanPowerResults:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """
        tea_tasting.utils.check_scalar(
            parameter,
            "parameter",
            in_={"power", "effect_size", "rel_effect_size", "n_obs"},
        )

        data = data.with_zero_div()
        covariate_coef = self._covariate_coef(data)
        covariate_mean = data.mean(self.numer_covariate) / data.mean(
            self.denom_covariate)
        metric_mean = self._metric_mean(data, covariate_coef, covariate_mean)

        power, effect_size, rel_effect_size, n_obs = self._validate_power_parameters(
            metric_mean=metric_mean,
            sample_count=data.count(),
            parameter=parameter,
        )

        result = MeanPowerResults()
        for effect_size_i, rel_effect_size_i in zip(
            effect_size,
            rel_effect_size,
            strict=True,
        ):
            for n_obs_i in n_obs:
                parameter_value = self._solve_power_from_stats(
                    sample_var=self._metric_var(data, covariate_coef),
                    sample_count=n_obs_i,
                    effect_size=effect_size_i,
                    power=power,
                )
                result.append(MeanPowerResult(
                    power=parameter_value if parameter == "power" else power,  # type: ignore
                    effect_size=(
                        parameter_value
                        if parameter in {"effect_size", "rel_effect_size"}
                        else effect_size_i
                    ),  # type: ignore
                    rel_effect_size=(
                        parameter_value / metric_mean
                        if parameter in {"effect_size", "rel_effect_size"}
                        else rel_effect_size_i
                    ),  # type: ignore
                    n_obs=(
                        math.ceil(parameter_value)
                        if parameter == "n_obs"
                        else n_obs_i
                    ),  # type: ignore
                ))

        return result


    def _validate_power_parameters(
        self,
        metric_mean: float,
        sample_count: int,
        parameter: Literal["power", "effect_size", "rel_effect_size", "n_obs"],
    ) -> tuple[
        float | None,  # power
        Sequence[float | int | None],  # effect_size
        Sequence[float | None],  # rel_effect_size
        Sequence[int | None],  # n_obs
    ]:
        n_obs = None
        effect_size = None
        rel_effect_size = None
        power = None

        if parameter in {"power", "n_obs"}:
            if self.effect_size is None and self.rel_effect_size is None:
                raise ValueError(
                    "Both `effect_size` and `rel_effect_size` are `None`. "
                    "One of them should be defined.",
                )
            effect_size = (
                self.effect_size if self.rel_effect_size is None
                else tuple(
                    rel_effect_size * metric_mean
                    for rel_effect_size in _to_seq(self.rel_effect_size)
                )
            )
            rel_effect_size = (
                self.rel_effect_size if self.effect_size is None
                else tuple(
                    effect_size / metric_mean
                    for effect_size in _to_seq(self.effect_size)
                )
            )

        if parameter in {"power", "effect_size", "rel_effect_size"}:
            n_obs = (sample_count,) if self.n_obs is None else self.n_obs

        if parameter in {"effect_size", "rel_effect_size", "n_obs"}:
            power = self.power

        return power, _to_seq(effect_size), _to_seq(rel_effect_size), _to_seq(n_obs)


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
        scale, distr, _ = self._scale_and_distr(
            contr_var=contr_var,
            contr_count=contr_count,
            treat_var=treat_var,
            treat_count=treat_count,
        )
        log_scale, log_distr, _ = self._scale_and_distr(
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


    def _solve_power_from_stats(
        self,
        sample_var: float,
        sample_count: int | None = None,
        effect_size: float | None = None,
        power: float | None = None,
    ) -> float | int:
        if power is None and effect_size is not None and sample_count is not None:
            return self._power_from_stats(
                sample_var=sample_var,
                sample_count=sample_count,
                effect_size=effect_size,
            )

        if power is not None and effect_size is None and sample_count is not None:
            def fn(x: float | int) -> float:
                return power - self._power_from_stats(
                    sample_var=sample_var,
                    sample_count=sample_count,
                    effect_size=x,
                )
            sign = -1 if self.alternative == "less" else 1
            other_bound = _find_boundary(
                fn,
                sign * 10 * math.sqrt(sample_var / sample_count),
            )
            lower_bound, upper_bound = sorted((0, other_bound))

        if power is not None and effect_size is not None and sample_count is None:
            def fn(x: float | int) -> float:
                return power - self._power_from_stats(
                    sample_var=sample_var,
                    sample_count=x,
                    effect_size=effect_size,
                )
            lower_bound = 3
            upper_bound = _find_boundary(fn, 10)

        return scipy.optimize.brentq(fn, lower_bound, upper_bound, maxiter=MAX_ITER)  # type: ignore


    def _power_from_stats(
        self,
        sample_var: float,
        sample_count: int | float,
        effect_size: float,
    ) -> float:
        _, null_distr, alt_distr = self._scale_and_distr(
            contr_var=sample_var,
            contr_count=sample_count / (1 + self.ratio),
            treat_var=sample_var,
            treat_count=sample_count * self.ratio / (1 + self.ratio),
            effect_size=effect_size,
        )
        if self.alternative == "greater":
            stat_critical = null_distr.isf(self.alpha)
            return alt_distr.sf(stat_critical)
        if self.alternative == "less":
            stat_critical = null_distr.ppf(self.alpha)
            return alt_distr.cdf(stat_critical)
        # two-sided
        stat_critical = null_distr.isf(self.alpha / 2)
        return alt_distr.cdf(-stat_critical) + alt_distr.sf(stat_critical)


    @overload
    def _scale_and_distr(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: None = None,
    ) -> tuple[float, scipy.stats.rv_frozen, None]:
        ...

    @overload
    def _scale_and_distr(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: float,
    ) -> tuple[float, scipy.stats.rv_frozen, scipy.stats.rv_frozen]:
        ...

    def _scale_and_distr(
        self,
        contr_var: float,
        contr_count: int | float,
        treat_var: float,
        treat_count: int | float,
        effect_size: float | None = None,
    ) -> tuple[float, scipy.stats.rv_frozen, scipy.stats.rv_frozen | None]:
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
            null_distr = scipy.stats.t(df=df)
            alt_distr = None if effect_size is None else scipy.stats.nct(
                df=df, nc=effect_size / scale)
        else:
            null_distr = scipy.stats.norm()
            alt_distr = None if effect_size is None else scipy.stats.norm(
                loc=effect_size / scale)

        return scale, null_distr, alt_distr


def _find_boundary(
    fn: Callable[[float | int], float],
    init: float | int,
    mult: float | int = 10,
) -> float:
    b = init
    i = 0
    while fn(b) > 0:
        b *= mult
        i += 1
        if i == MAX_ITER:
            raise RuntimeError(
                "Cannot find parameter boundaries. "
                "Maximum number of iterations is reached.",
            )
    return b


def _to_seq(x: N | Sequence[N]) -> Sequence[N]:
    if isinstance(x, Sequence):
        return x
    return (x,)


class Mean(RatioOfMeans):  # noqa: D101
    def __init__(  # noqa: PLR0913
        self,
        value: str,
        covariate: str | None = None,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
        alpha: float | None = None,
        ratio: float | int | None = None,
        power: float | None = None,
        effect_size: float | int | Sequence[float | int] | None = None,
        rel_effect_size: float | Sequence[float] | None = None,
        n_obs: int | Sequence[int] | None = None,
    ) -> None:
        """Metric for the analysis of means.

        Args:
            value: Metric value column name.
            covariate: Metric covariate column name.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the means are unequal,
                - `"greater"`: the mean in the treatment variant is greater than the mean
                    in the control variant,
                - `"less"`: the mean in the treatment variant is less than the mean
                    in the control variant.

            confidence_level: Confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If `True`,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).
            alpha: Significance level. Only for the analysis of power.
            ratio: Ratio of the number of observations in the treatment
                relative to the control. Only for the analysis of power.
            power: Statistical power. Only for the analysis of power.
            effect_size: Absolute effect size. Difference between the two means.
                Only for the analysis of power.
            rel_effect_size: Relative effect size. Difference between the two means,
                divided by the control mean. Only for the analysis of power.
            n_obs: Number of observations in the control and in the treatment together.
                Only for the analysis of power.

        Parameter defaults:
            Defaults for parameters `alpha`, `alternative`, `confidence_level`,
            `equal_var`, `n_obs`, `power`, `ratio`, and `use_t` can be changed
            using the `config_context` and `set_context` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Deng, A., Knoblich, U., & Lu, J. (2018). Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas](https://alexdeng.github.io/public/files/kdd2018-dm.pdf).
            - [Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf).

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Mean("orders"),
            ...     revenue_per_user=tt.Mean("revenue"),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> print(result)
                      metric control treatment rel_effect_size rel_effect_size_ci pvalue
             orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            With CUPED:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Mean("orders", "orders_covariate"),
            ...     revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
            ... )
            >>> data = tt.make_users_data(seed=42, covariates=True)
            >>> result = experiment.analyze(data)
            >>> print(result)
                      metric control treatment rel_effect_size rel_effect_size_ci  pvalue
             orders_per_user   0.523     0.581             11%        [2.9%, 20%] 0.00733
            revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00674

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     seed=42,
            ...     sessions_uplift=0,
            ...     orders_uplift=0,
            ...     revenue_uplift=0,
            ...     covariates=True,
            ... )
            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     n_obs=(10_000, 20_000),
            ... )
            >>> # Solve for effect size.
            >>> print(orders_per_user.solve_power(data))
            power effect_size rel_effect_size n_obs
              80%      0.0374            7.2% 10000
              80%      0.0264            5.1% 20000

            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     rel_effect_size=0.05,
            ... )
            >>> # Solve for the total number of observations.
            >>> print(orders_per_user.solve_power(data, "n_obs"))
            power effect_size rel_effect_size n_obs
              80%      0.0260            5.0% 20733

            >>> orders_per_user = tt.Mean(
            ...     "orders",
            ...     "orders_covariate",
            ...     rel_effect_size=0.1,
            ... )
            >>> # Solve for power. Infer number of observations from the sample.
            >>> print(orders_per_user.solve_power(data, "power"))
            power effect_size rel_effect_size n_obs
              69%      0.0519             10%  4000

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
            alpha=alpha,
            ratio=ratio,
            power=power,
            effect_size=effect_size,
            rel_effect_size=rel_effect_size,
            n_obs=n_obs,
        )
        self.value = value
        self.covariate = covariate
