"""Resampling statistical methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import tea_tasting.config
from tea_tasting.metrics.base import MetricBaseGranular
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import numpy.typing as npt


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
        if isinstance(columns, str):
            columns = (columns,)
        for col in columns:
            tea_tasting.utils.check_scalar(col, "column", typ=str)
        self.columns = columns

        self.statistic = statistic

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
