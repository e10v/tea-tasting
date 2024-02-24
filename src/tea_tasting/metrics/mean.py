"""Analysis of means of two independent samples."""
# ruff: noqa: PD901

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting._utils
import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.metrics.base


if TYPE_CHECKING:
    from typing import Any, Literal

    import ibis.expr.types
    import pandas as pd


class MeansResult(NamedTuple):
    """Result of an analysis of means.

    Attributes:
        control: Control mean.
        treatment: Treatment mean.
        effect_size: Absolute effect size. Difference between two means.
        effect_size_ci_lower: Lower bound of the absolute effect size
            confidence interval.
        effect_size_ci_upper: Upper bound of the absolute effect size
            confidence interval.
        rel_effect_size: Relative effect size. Difference between two means,
            divided by the control mean.
        rel_effect_size_ci_lower: Lower bound of the relative effect size
            confidence interval.
        rel_effect_size_ci_upper: Upper bound of the relative effect size
            confidence P-value.
        pvalue: float
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


class RatioOfMeans(
    tea_tasting.metrics.base.MetricBaseAggregated,
    tea_tasting._utils.ReprMixin,
):
    """Compares ratios of metrics means between variants."""

    def __init__(  # noqa: PLR0913
        self,
        numer: str,
        denom: str | None = None,
        numer_covariate: str | None = None,
        denom_covariate: str | None = None,
        alternative: Literal["two-sided", "less", "greater"] | None = None,
        confidence_level: float | None = None,
        equal_var: bool | None = None,
        use_t: bool | None = None,
    ) -> None:
        """Create a ratio metric.

        Args:
            numer: Numerator column name.
            denom: Denominator column name.
            numer_covariate: Covariate numerator column name.
            denom_covariate: Covariate denominator column name.
            alternative: Default alternative hypothesis.
            confidence_level: Default confidence level for the confidence interval.
            equal_var: Defines whether equal variance is assumed. If True,
                pooled variance is used for the calculation of the standard error
                of the difference between two means.
            use_t: Defines whether to use the Student's t-distribution (True) or
                the Normal distribution (False).
        """
        self.numer = numer
        self.denom = denom
        self.numer_covariate = numer_covariate
        self.denom_covariate = denom_covariate
        self.alternative = (
            alternative
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )
        self.confidence_level = (
            confidence_level
            if confidence_level is not None
            else tea_tasting.config.get_config("confidence_level")
        )
        self.equal_var = (
            equal_var
            if equal_var is not None
            else tea_tasting.config.get_config("equal_var")
        )
        self.use_t = (
            use_t
            if use_t is not None
            else tea_tasting.config.get_config("use_t")
        )


    @property
    def aggr_cols(self) -> tea_tasting.metrics.base.AggrCols:
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
        return tea_tasting.metrics.base.AggrCols(
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


    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant_col: str | None = None,
    ) -> MeansResult:
        """Analyze metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant_col: Variant column name.

        Returns:
            Experiment result for a metric.
        """
        data = self.validate_aggregates(data, variant_col=variant_col)
        contr = data[control]
        treat = data[treatment]
        total = contr + treat
        covariate_coef = self._covariate_coef(total)
        covariate_mean = total.mean(self.numer_covariate) / total.mean(
            self.denom_covariate)
        return self._analyze_from_stats(
            contr_mean=self._metric_mean(contr, covariate_coef, covariate_mean),
            contr_var=self._metric_var(contr, covariate_coef),
            contr_count=contr.count(),
            treat_mean=self._metric_mean(treat, covariate_coef, covariate_mean),
            treat_var=self._metric_var(treat, covariate_coef),
            treat_count=treat.count(),
        )


    def _analyze_from_stats(
        self,
        contr_mean: float,
        contr_var: float,
        contr_count: int,
        treat_mean: float,
        treat_var: float,
        treat_count: int,
    ) -> MeansResult:
        scale, distr = self._scale_and_distr(
            contr_var=contr_var,
            contr_count=contr_count,
            treat_var=treat_var,
            treat_count=treat_count,
        )
        rel_scale, rel_distr = self._scale_and_distr(
            contr_var=contr_var / contr_mean / contr_mean,
            contr_count=contr_count,
            treat_var=treat_var / treat_mean / treat_mean,
            treat_count=treat_count,
        )

        means_ratio = treat_mean / contr_mean
        effect_size = treat_mean - contr_mean
        std_effect_size = effect_size / scale

        if self.alternative == "less":
            q = self.confidence_level
            effect_size_ci_lower = means_ratio_ci_lower = float("-inf")
            effect_size_ci_upper = effect_size + scale*distr.ppf(q)
            means_ratio_ci_upper = means_ratio * np.exp(rel_scale * rel_distr.ppf(q))
            pvalue = distr.cdf(std_effect_size)
        elif self.alternative == "greater":
            q = 1 - self.confidence_level
            effect_size_ci_lower = effect_size + scale*distr.ppf(q)
            means_ratio_ci_lower = means_ratio * np.exp(rel_scale * rel_distr.ppf(q))
            effect_size_ci_upper = means_ratio_ci_upper = float("+inf")
            pvalue = distr.cdf(-std_effect_size)
        else:
            q = (1 + self.confidence_level) / 2
            half_ci = scale * distr.ppf(q)
            effect_size_ci_lower = effect_size - half_ci
            effect_size_ci_upper = effect_size + half_ci

            rel_half_ci = np.exp(rel_scale * rel_distr.ppf(q))
            means_ratio_ci_lower = means_ratio / rel_half_ci
            means_ratio_ci_upper = means_ratio * rel_half_ci

            pvalue = 2 * distr.cdf(-np.abs(std_effect_size))

        return MeansResult(
            control=contr_mean,
            treatment=treat_mean,
            effect_size=effect_size,
            effect_size_ci_lower=effect_size_ci_lower,
            effect_size_ci_upper=effect_size_ci_upper,
            rel_effect_size=means_ratio - 1,
            rel_effect_size_ci_lower=means_ratio_ci_lower - 1,
            rel_effect_size_ci_upper=means_ratio_ci_upper - 1,
            pvalue=pvalue,
        )


    def _covariate_coef(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        cov = aggr.ratio_cov(
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        )

        if cov == 0:
            return 0

        return cov / aggr.ratio_var(self.numer_covariate, self.denom_covariate)


    def _metric_mean(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
        covariate_mean: float,
    ) -> float:
        return aggr.mean(self.numer)/aggr.mean(self.denom) - covariate_coef*(
            aggr.mean(self.numer_covariate)/aggr.mean(self.denom_covariate)
            - covariate_mean
        )


    def _metric_var(
        self,
        aggr: tea_tasting.aggr.Aggregates,
        covariate_coef: float,
    ) -> float:
        left_var = aggr.ratio_var(self.numer, self.denom)
        right_var = aggr.ratio_var(self.numer_covariate, self.denom_covariate)
        cov = aggr.ratio_cov(
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        )
        return left_var + covariate_coef*covariate_coef*right_var - 2*covariate_coef*cov


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
            scale = np.sqrt(pooled_var/contr_count + pooled_var/treat_count)
        else:
            scale = np.sqrt(contr_var/contr_count + treat_var/treat_count)

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


if __name__ == "__main__":
    import pandas as pd
    import tea_tasting.datasets

    data = tea_tasting.datasets.make_users_data(covariates=True, seed=2)
    cols = (
        "visits", "orders", "revenue",
        "visits_covariate", "orders_covariate", "revenue_covariate",
    )
    data = tea_tasting.aggr.read_aggregates(
        data,
        group_col="variant",
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

    with tea_tasting.config.config_context(alternative="less"):
        visits_per_user = RatioOfMeans(numer="visits")
        visits_per_user_cuped = RatioOfMeans(
            numer="visits",
            numer_covariate="visits_covariate",
        )
        orders_per_visit = RatioOfMeans(numer="orders", denom="visits")
        orders_per_visit_cuped = RatioOfMeans(
            numer="orders",
            denom="visits",
            numer_covariate="orders_covariate",
            denom_covariate="visits_covariate",
        )
        orders_per_user = RatioOfMeans(numer="orders")
        orders_per_user_cuped = RatioOfMeans(
            numer="orders",
            numer_covariate="orders_covariate",
        )
        revenue_per_user = RatioOfMeans(numer="revenue")
        revenue_per_user_cuped = RatioOfMeans(
            numer="revenue",
            numer_covariate="revenue_covariate",
        )

    metrics =  (
        (visits_per_user, "visits_per_user"),
        (visits_per_user_cuped, "visits_per_user_cuped"),
        (orders_per_visit, "orders_per_visit"),
        (orders_per_visit_cuped, "orders_per_visit_cuped"),
        (orders_per_user, "orders_per_user"),
        (orders_per_user_cuped, "orders_per_user_cuped"),
        (revenue_per_user, "revenue_per_user"),
        (revenue_per_user_cuped, "revenue_per_user_cuped"),
    )

    results = pd.DataFrame(
        {"metric": name} | metric.analyze(data, control=0, treatment=1)._asdict()
        for metric, name in metrics
    )

    print(results)
