"""Analysis using resampling methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting.config
from tea_tasting.metrics.base import MetricBaseGranular
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    import numpy.typing as npt
    import pandas as pd


class BootstrapResult(NamedTuple):
    control: float
    treatment: float
    effect_size: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    rel_effect_size: float
    rel_effect_size_ci_lower: float
    rel_effect_size_ci_upper: float


class Bootstrap(MetricBaseGranular[BootstrapResult]):
    def __init__(
        self,
        columns: str | Sequence[str],
        statistic: Callable[..., npt.NDArray[np.number[Any]]],
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        confidence_level: float | None = None,
        n_resamples: int = 10_000,
        method: Literal["percentile", "basic", "bca"] = "bca",
        batch: int | None = None,
        random_state: int | np.random.Generator | np.random.SeedSequence | None = None,
    ) -> None:
        if not isinstance(columns, str):
            for col in columns:
                tea_tasting.utils.check_scalar(col, "column", typ=str)
        self.columns = columns

        self.statistic = tea_tasting.utils.check_scalar(
            statistic, "statistic", typ=Callable)

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

        self.n_resamples = tea_tasting.utils.check_scalar(
            n_resamples, "n_resamples", typ=int, gt=0)

        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, in_={"percentile", "basic", "bca"})

        self.batch = tea_tasting.utils.check_scalar(batch, "batch", typ=int | None)

        self.random_state = tea_tasting.utils.check_scalar(
            random_state,
            "random_state",
            typ=int | np.random.Generator | np.random.SeedSequence | None,
        )


    @property
    def cols(self) -> Sequence[str]:
        if isinstance(self.columns, str):
            return (self.columns,)
        return self.columns


    def analyze_dataframes(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
    ) -> BootstrapResult:
        def statistic(
            contr: npt.NDArray[np.number[Any]],
            treat: npt.NDArray[np.number[Any]],
            axis: int = -1,
        ) -> npt.NDArray[np.number[Any]]:
            contr_stat = self.statistic(contr, axis=axis)
            treat_stat = self.statistic(treat, axis=axis)

            effect_size = treat_stat - contr_stat
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_effect_size = np.divide(treat_stat, contr_stat) - 1

            return np.stack((effect_size, rel_effect_size), axis=0)

        contr = control.loc[:, self.columns].to_numpy()  # type: ignore
        treat = treatment.loc[:, self.columns].to_numpy()  # type: ignore
        stat = statistic(contr, treat, axis=0)

        result = scipy.stats.bootstrap(
            (contr, treat),
            statistic,
            n_resamples=self.n_resamples,
            batch=self.batch,
            axis=0,
            confidence_level=self.confidence_level,
            alternative=self.alternative,
            method=self.method,
            random_state=self.random_state,
        )
        ci = result.confidence_interval

        return BootstrapResult(
            control=self.statistic(contr, axis=0),  # type: ignore
            treatment=self.statistic(treat, axis=0),  # type: ignore
            effect_size=stat[0],
            effect_size_ci_lower=ci.low[0],
            effect_size_ci_upper=ci.high[1],
            rel_effect_size=stat[1],
            rel_effect_size_ci_lower=ci.low[1],
            rel_effect_size_ci_upper=ci.high[1],
        )
