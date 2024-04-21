"""Analysis of proportions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import scipy.stats

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.metrics
from tea_tasting.metrics.base import AggrCols, MetricBaseAggregated
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Any, Literal

    import ibis.expr.types
    import pandas as pd


_MAX_EXACT_THRESHOLD = 1000


class SampleRatioResult(NamedTuple):
    """Result of the sample ratio mismatch check.

    Attributes:
        control: Number of observations in control.
        treatment: Number of observations in treatment.
        pvalue: P-value
    """
    control: float
    treatment: float
    pvalue: float


class SampleRatio(MetricBaseAggregated[SampleRatioResult]):
    """Sample ratio mismatch check."""

    def __init__(
        self,
        ratio: float | int | dict[Any, float | int] = 1,
        *,
        method: Literal["auto", "binom", "norm"] = "auto",
        correction: bool = True,
    ) -> None:
        """Sample ratio mismatch check.

        Args:
            ratio: Expected ratio of the number of observation in treatment
                relative to control.
            method: Statistical test used for calculation of p-value. Options:
                "auto": Apply exact binomial test if the total number of observations
                    is < 1000, or normal approximation otherwise.
                "binom": Apply exact binomial test.
                "norm": Apply normal approximation of the binomial distribution.
            correction: If True, add continuity correction.
                Only for normal approximation.
        """
        if isinstance(ratio, dict):
            for val in ratio.values():
                tea_tasting.utils.auto_check(val, "ratio")
        else:
            tea_tasting.utils.auto_check(ratio, "ratio")
        self.ratio = ratio

        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, is_in={"auto", "binom", "norm"})
        self.correction = tea_tasting.utils.auto_check(correction, "correction")


    @property
    def aggr_cols(self) -> AggrCols:
        """Columns to aggregate for a metric analysis."""
        return AggrCols(has_count=True)


    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> SampleRatioResult:
        """Perform sample ratio mismatch check.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
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
            pvalue = scipy.stats.binomtest(k=k, n=n, p=p).pvalue
        else:  # norm
            d = k - n*p
            if self.correction and d != 0:
                d = min(d + 0.5, 0) if d < 0 else max(d - 0.5, 0)
            z = d / math.sqrt(n * p * (1 - p))
            pvalue = 2 * scipy.stats.norm.sf(abs(z))

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
        """Method stub for compatibility with the base class."""
        raise NotImplementedError
