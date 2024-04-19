"""Analysis of frequency metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
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


MAX_EXACT_THRESHOLD = 1000


class SampleRatioResult(NamedTuple):
    control: float
    treatment: float
    pvalue: float


class SampleRatio(MetricBaseAggregated[SampleRatioResult]):
    def __init__(
        self,
        ratio: float | int | dict[Any, float | int] = 1,
        *,
        method: Literal["auto", "binom", "norm"] = "auto",
        correction: bool = True,
    ) -> None:
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
        return AggrCols(has_count=True)


    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> SampleRatioResult:
        aggr = tea_tasting.metrics.aggregate_by_variants(
            data,
            aggr_cols=self.aggr_cols,
            variant=variant,
        )

        k = aggr[treatment].count()
        n = k + aggr[control].count()
        tea_tasting.utils.check_scalar(n - k, "number of observations in control", gt=0)
        tea_tasting.utils.check_scalar(k, "number of observations in treatment", gt=0)

        r = (
            self.ratio
            if isinstance(self.ratio, float | int)
            else self.ratio[treatment] / self.ratio[control]
        )
        p = r / (1 + r)

        if (
            self.method == "binom" or
            (self.method == "auto" and n < MAX_EXACT_THRESHOLD)
        ):
            pvalue = scipy.stats.binomtest(k=k, n=n, p=p).pvalue
        else:  # norm
            d = k - n*p
            if self.correction and d != 0:
                d = min(d + 0.5, 0) if d < 0 else max(d - 0.5, 0)
            z = d / np.sqrt(n * p * (1 - p))
            pvalue = 2 * scipy.stats.norm.sf(np.abs(z))

        return SampleRatioResult(
            control=n - k,
            treatment=k,
            pvalue=pvalue,  # type: ignore
        )
