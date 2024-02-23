"""Mean metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tea_tasting._utils
import tea_tasting.config
import tea_tasting.metrics.base


if TYPE_CHECKING:
    from typing import Literal


class RatioOfMeans(
    tea_tasting.metrics.base.MetricBaseAggregated,
    tea_tasting._utils.ReprMixin,
):
    """Compares of ratios of metrics means between variants."""

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
