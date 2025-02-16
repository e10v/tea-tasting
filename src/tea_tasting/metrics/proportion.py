"""Metrics for the analysis of proportions."""

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
    from typing import Literal

    import ibis.expr.types
    import narwhals.typing


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


class SampleRatio(MetricBaseAggregated[SampleRatioResult]):  # noqa: D101
    def __init__(
        self,
        ratio: float | int | dict[object, float | int] = 1,
        *,
        method: Literal["auto", "binom", "norm"] = "auto",
        correction: bool = True,
    ) -> None:
        """Metric for sample ratio mismatch check.

        Args:
            ratio: Expected ratio of the number of observations in the treatment
                relative to the control.
            method: Statistical test used for calculation of p-value:

                - `"auto"`: Apply exact binomial test if the total number
                    of observations is < 1000; or normal approximation otherwise.
                - `"binom"`: Apply exact binomial test.
                - `"norm"`: Apply normal approximation of the binomial distribution.

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
            >>> print(result.to_string(("metric", "control", "treatment", "pvalue")))
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
            >>> print(result.to_string(("metric", "control", "treatment", "pvalue")))
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
        self.correction = tea_tasting.utils.auto_check(correction, "correction")


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
        """Stub method for compatibility with the base class."""
        raise NotImplementedError
