"""Metrics for nonparametric analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.stats

import tea_tasting.config
from tea_tasting.metrics.base import MetricBaseGranular
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Literal

    import numpy.typing as npt
    import pyarrow as pa


class MannWhitneyUResult(NamedTuple):
    """Result of the analysis using the Mann-Whitney U test.

    Attributes:
        control: ROC AUC for control. Probability that a value from control
            is greater than a value from treatment, plus half the probability
            that they are equal.
        treatment: ROC AUC for treatment. Probability that a value from treatment
            is greater than a value from control, plus half the probability
            that they are equal.
        effect_size: Absolute effect size. Difference between treatment and control
            ROC AUC values.
        pvalue: P-value.
        statistic: Mann-Whitney U statistic.
    """

    control: float
    treatment: float
    effect_size: float
    pvalue: float
    statistic: float


class MannWhitneyU(MetricBaseGranular[MannWhitneyUResult]):  # noqa: D101
    def __init__(
        self,
        column: str,
        *,
        alternative: Literal["two-sided", "greater", "less"] | None = None,
        correction: bool | None = None,
        method: Literal["auto", "asymptotic", "exact"] = "auto",
        nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
    ) -> None:
        """Metric for nonparametric analysis with the Mann-Whitney U test.

        Args:
            column: Metric column name.
            alternative: Alternative hypothesis:

                - `"two-sided"`: the distributions are not equal,
                - `"greater"`: the treatment distribution is stochastically greater
                    than the control distribution,
                - `"less"`: the treatment distribution is stochastically less than
                    the control distribution.

            correction: Whether a continuity correction (1/2) should be applied.
                Only for the asymptotic method.
                Defaults to the global config value (`True`).
            method: Method used for p-value calculation:

                - `"auto"`: exact when sample sizes are small and there are no ties;
                    asymptotic otherwise.
                - `"asymptotic"`: normal approximation with tie correction.
                - `"exact"`: exact p-value calculation.

            nan_policy: Defines how to handle `nan` values:

                - `"propagate"`: return `nan`,
                - `"omit"`: ignore `nan` values,
                - `"raise"`: raise an exception.

        Parameter defaults:
            Defaults for parameters `alternative` and `correction` can be changed using
            the `config_context` and `set_config` functions.
            See the [Global configuration](https://tea-tasting.e10v.me/api/config/)
            reference for details.

        References:
            - [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).
            - [scipy.stats.mannwhitneyu &#8212; SciPy Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).

        Examples:
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     revenue_auc=tt.MannWhitneyU("revenue"),
            ... )
            >>> data = tt.make_users_data(seed=42, n_users=1000)
            >>> result = experiment.analyze(data)
            >>> result
                 metric control treatment rel_effect_size rel_effect_size_ci pvalue
            revenue_auc   0.472     0.528               -             [-, -] 0.0698

            ```

            With specific alternative and method:

            ```pycon
            >>> experiment = tt.Experiment(
            ...     revenue_auc=tt.MannWhitneyU(
            ...         "revenue",
            ...         alternative="greater",
            ...         method="asymptotic",
            ...         correction=False,
            ...     ),
            ... )
            >>> data = tt.make_users_data(seed=42, n_users=1000)
            >>> experiment.analyze(data)
                 metric control treatment rel_effect_size rel_effect_size_ci pvalue
            revenue_auc   0.472     0.528               -             [-, -] 0.0349

            ```
        """
        self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
        self.alternative: Literal["two-sided", "greater", "less"] = (
            tea_tasting.utils.auto_check(alternative, "alternative")
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )
        self.correction = (
            tea_tasting.utils.auto_check(correction, "correction")
            if correction is not None
            else tea_tasting.config.get_config("correction")
        )
        self.method = tea_tasting.utils.check_scalar(
            method,
            "method",
            typ=str,
            in_={"auto", "asymptotic", "exact"},
        )
        self.nan_policy: Literal["propagate", "omit", "raise"]
        self.nan_policy = tea_tasting.utils.check_scalar(
            nan_policy,
            "nan_policy",
            typ=str,
            in_={"propagate", "omit", "raise"},
        )

    @property
    def cols(self) -> tuple[str]:
        """Columns to be fetched for a metric analysis."""
        return (self.column,)

    def analyze_granular(
        self,
        control: pa.Table,
        treatment: pa.Table,
    ) -> MannWhitneyUResult:
        """Analyze a metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        contr = _select_as_numpy(control, self.column)
        treat = _select_as_numpy(treatment, self.column)
        contr, treat = _handle_nan_policy(contr, treat, self.nan_policy)
        if len(contr) == 0 or len(treat) == 0:
            return MannWhitneyUResult(
                control=float("nan"),
                treatment=float("nan"),
                effect_size=float("nan"),
                pvalue=float("nan"),
                statistic=float("nan"),
            )

        result = scipy.stats.mannwhitneyu(
            treat,
            contr,
            alternative=self.alternative,
            use_continuity=self.correction,
            method=self.method,
        )
        n_pairs = len(contr) * len(treat)
        treat_auc = float(result.statistic) / n_pairs if n_pairs > 0 else float("nan")
        contr_auc = 1 - treat_auc

        return MannWhitneyUResult(
            control=contr_auc,
            treatment=treat_auc,
            effect_size=treat_auc - contr_auc,
            pvalue=float(result.pvalue),
            statistic=float(result.statistic),
        )


def _select_as_numpy(data: pa.Table, column: str) -> npt.NDArray[np.number]:
    return data[column].combine_chunks().to_numpy(zero_copy_only=False)


def _handle_nan_policy(
    control: npt.NDArray[np.number],
    treatment: npt.NDArray[np.number],
    nan_policy: Literal["propagate", "omit", "raise"],
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    if nan_policy == "omit":
        return control[~np.isnan(control)], treatment[~np.isnan(treatment)]

    if nan_policy == "raise" and (np.isnan(control).any() or np.isnan(treatment).any()):
        raise ValueError("Input contains nan.")

    return control, treatment
