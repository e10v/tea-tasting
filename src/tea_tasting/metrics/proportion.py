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
    null_ratio: float
    ratio: float
    ratio_ci_lower: float
    ratio_ci_upper: float


class SampleRatio(MetricBaseAggregated[SampleRatioResult]):
    def __init__(
        self,
        ratio: float | int | dict[Any, float | int] = 1,
        *,
        method: Literal["auto", "binom", "norm"] = "auto",
        binom_ci_method: Literal["exact", "wilson", "wilsoncc"] = "exact",
        correction: bool = True,
        confidence_level: float | None = None,
    ) -> None:
        if isinstance(ratio, dict):
            for val in ratio.values():
                tea_tasting.utils.auto_check(val, "ratio")
        else:
            tea_tasting.utils.auto_check(ratio, "ratio")
        self.ratio = ratio

        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, is_in={"auto", "binom", "norm"})
        self.binom_ci_method = tea_tasting.utils.check_scalar(
            binom_ci_method,
            "binom_ci_method",
            typ=str,
            is_in={"exact", "wilson", "wilsoncc"},
        )
        self.correction = tea_tasting.utils.auto_check(correction, "correction")
        self.confidence_level = (
            tea_tasting.utils.auto_check(confidence_level, "confidence_level")
            if confidence_level is not None
            else tea_tasting.config.get_config("confidence_level")
        )


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
            binom_result = scipy.stats.binomtest(k=k, n=n, p=p)
            pvalue = binom_result.pvalue
            prop_ci_lower, prop_ci_upper = binom_result.proportion_ci(
                confidence_level=self.confidence_level,
                method=self.binom_ci_method,
            )
        else:  # norm
            d = k - n*p
            if self.correction and d != 0:
                d = min(d + 0.5, 0) if d < 0 else max(d - 0.5, 0)
            z = d / np.sqrt(n * p * (1 - p))
            pvalue = 2 * scipy.stats.norm.sf(np.abs(z))
            s = np.sqrt(k * (n - k) / n) / n
            prop_half_ci = s * scipy.stats.norm.ppf((1 + self.confidence_level) / 2)
            prop = k / n
            prop_ci_lower = prop - prop_half_ci
            prop_ci_upper = prop + prop_half_ci

        return SampleRatioResult(
            control=n - k,
            treatment=k,
            pvalue=pvalue,  # type: ignore
            null_ratio=r,
            ratio=k / (n - k),
            ratio_ci_lower=prop_ci_lower / (1 - prop_ci_lower),
            ratio_ci_upper=prop_ci_upper / (1 - prop_ci_upper),
        )
