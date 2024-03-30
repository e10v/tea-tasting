"""Experiments and experiment results."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd


if TYPE_CHECKING:
    from typing import Any


class ExperimentResult(NamedTuple):
    """Experiment result for a pair of variants (control, treatment)."""
    result: dict[str, NamedTuple | dict[str, Any]]

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Return result as a sequence of dictionaries, one dictionary per metric."""
        return tuple(
            {"metric": k} | (v if isinstance(v, dict) else v._asdict())
            for k, v in self.result.items()
        )

    def to_pandas(self) -> pd.DataFrame:
        """Return result as a Pandas DataFrame, one row per metric."""
        return pd.DataFrame.from_records(self.to_dicts())


class ExperimentResults(NamedTuple):
    """Experiment results for all pairs of variants (control, treatment)."""
    results: dict[tuple[Any, Any], ExperimentResult]

    def get(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> ExperimentResult:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result.
        """
        if control is None or treatment is None:
            if len(self.results) != 1:
                raise ValueError(
                    f"control is {control}, treatment is {treatment},"
                    " both must be not None.",
                )
            return next(iter(self.results.values()))

        return self.results[(control, treatment)]

    def to_dicts(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result as a sequence of dictionaries, one dictionary per metric.
        """
        return self.get(control, treatment).to_dicts()

    def to_pandas(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> pd.DataFrame:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result as a Pandas DataFrame, one row per metric.
        """
        return self.get(control, treatment).to_pandas()
