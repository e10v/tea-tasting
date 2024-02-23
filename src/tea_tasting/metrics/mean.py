"""Analysis of means of two independent samples."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

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
        control: _description_
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
            equal_var: Defines whether to use the Welch's t-test (`False`)
                or the standard Student's t-test (`True`) by default. The standard
                Student's t-test assumes equal population variances,
                while Welch's t-test doesn't. Applicable only if `use_t` is `True`.
            use_t: Defines whether to use the Student's t-distribution (`True`) or
                the Normal distribution (`False`).
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
        covariate_coef = self._covariate_coef(contr + treat)
        return self._analyze_from_stats(
            contr_mean=contr.mean(self.numer) / contr.mean(self.denom),
            contr_var=self._metric_var(contr, covariate_coef),
            contr_count=contr.count(),
            treat_mean=treat.mean(self.numer) / treat.mean(self.denom),
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
        ...

    def _covariate_coef(self, aggr: tea_tasting.aggr.Aggregates) -> float:
        return aggr.ratio_cov(
            self.numer,
            self.denom,
            self.numer_covariate,
            self.denom_covariate,
        ) / aggr.ratio_var(
            self.numer_covariate,
            self.denom_covariate,
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
