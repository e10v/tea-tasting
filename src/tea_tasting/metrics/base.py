"""Base classes for metrics."""

from __future__ import annotations

import abc
from collections import UserList
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    overload,
)

import numpy as np

import tea_tasting.aggr
import tea_tasting.data
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Literal

    import numpy.typing as npt
    import pyarrow as pa


class _NamedTupleLike(Protocol):
    def __getitem__(self, key: int, /) -> Any:
        ...

    def _asdict(self) -> Mapping[str, Any]:
        ...


type MetricResult = _NamedTupleLike | Mapping[str, Any]
type MetricPowerResult = MetricResult

MetricResultT = TypeVar("MetricResultT", bound=MetricResult)
MetricPowerResultT = TypeVar("MetricPowerResultT", bound=MetricPowerResult)


def _result_to_dict(result: MetricResult) -> dict[str, object]:
    if isinstance(result, Mapping):
        return dict(result.items())  # ty: ignore[invalid-return-type]
    return dict(result._asdict().items())


class MetricPowerResults(
    tea_tasting.utils.DictsReprMixin,
    UserList[MetricPowerResultT],
):
    """Power analysis results."""
    default_keys = ("power", "effect_size", "rel_effect_size", "n_obs")

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the results to a sequence of dictionaries."""
        return tuple(_result_to_dict(v) for v in self)

MetricPowerResultsT = TypeVar("MetricPowerResultsT", bound=MetricPowerResults)


class MetricBase[MetricResultT: MetricResult](abc.ABC, tea_tasting.utils.ReprMixin):
    """Base class for metrics."""
    @abc.abstractmethod
    def analyze(
        self,
        data: tea_tasting.data.Table,
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> MetricResultT:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """


class PowerBase[MetricPowerResultsT: MetricPowerResults](
    abc.ABC, tea_tasting.utils.ReprMixin,
):
    """Base class for the analysis of power."""
    @abc.abstractmethod
    def solve_power(
        self,
        data: tea_tasting.data.Table,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> MetricPowerResultsT:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """


class _HasAggrCols(abc.ABC):
    @property
    @abc.abstractmethod
    def aggr_cols(self) -> tea_tasting.data.AggrCols:
        """Columns to be aggregated for an analysis."""


class MetricBaseAggregated(MetricBase[MetricResultT], _HasAggrCols):
    """Base class for metrics, which are analyzed using aggregated statistics."""
    @overload
    def analyze(
        self,
        data: tea_tasting.data.AggregatesByVariant,
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        ...

    @overload
    def analyze(
        self,
        data: tea_tasting.data.Table,
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> MetricResultT:
        ...

    def analyze(
        self,
        data: tea_tasting.data.Table | tea_tasting.data.AggregatesByVariant,
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        tea_tasting.utils.check_scalar(variant, "variant", typ=str | None)
        aggr: tea_tasting.data.AggregatesByVariant
        if tea_tasting.data._is_aggregates_mapping(data):
            aggr = data
        else:
            if variant is None:
                raise ValueError(
                    "The variant parameter is required but was not provided.",
                )
            aggr = tea_tasting.data.read_aggregates(
                data,
                aggr_cols=self.aggr_cols,
                variant=variant,
            )
        return self.analyze_aggregates(
            control=aggr[control],
            treatment=aggr[treatment],
        )

    @abc.abstractmethod
    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> MetricResultT:
        """Analyze a metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """


class PowerBaseAggregated(PowerBase[MetricPowerResultsT], _HasAggrCols):
    """Base class for the analysis of power using aggregated statistics."""
    def solve_power(
        self,
        data: tea_tasting.data.Table | tea_tasting.aggr.Aggregates,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> MetricPowerResultsT:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """
        tea_tasting.utils.check_scalar(
            parameter,
            "parameter",
            in_={"power", "effect_size", "rel_effect_size", "n_obs"},
        )
        if not isinstance(data, tea_tasting.aggr.Aggregates):
            data = tea_tasting.data.read_aggregates(
                data,
                aggr_cols=self.aggr_cols,
            )
        return self.solve_power_from_aggregates(data=data, parameter=parameter)

    @abc.abstractmethod
    def solve_power_from_aggregates(
        self,
        data: tea_tasting.aggr.Aggregates,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> MetricPowerResultsT:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """

class _HasCols(abc.ABC):
    @property
    @abc.abstractmethod
    def cols(self) -> Sequence[str]:
        """Columns to be fetched for an analysis."""


class MetricBaseGranular(MetricBase[MetricResultT], _HasCols):
    """Base class for metrics, which are analyzed using granular data."""
    @overload
    def analyze(
        self,
        data: tea_tasting.data.TablesByVariant,
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        ...

    @overload
    def analyze(
        self,
        data: tea_tasting.data.Table,
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> MetricResultT:
        ...

    def analyze(
        self,
        data: tea_tasting.data.Table | tea_tasting.data.TablesByVariant,
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        tea_tasting.utils.check_scalar(variant, "variant", typ=str | None)
        if tea_tasting.data._is_tables_mapping(data):
            dfs = tea_tasting.data.read_granular(data, cols=self.cols)
        else:
            if variant is None:
                raise ValueError(
                    "The variant parameter is required but was not provided.",
                )
            dfs = tea_tasting.data.read_granular(
                data,
                cols=self.cols,
                variant=variant,
            )
        return self.analyze_granular(
            control=dfs[control],
            treatment=dfs[treatment],
        )

    @abc.abstractmethod
    def analyze_granular(
        self,
        control: pa.Table,
        treatment: pa.Table,
    ) -> MetricResultT:
        """Analyze a metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """


def _handle_nan_policy(
    control: npt.NDArray[np.number],
    treatment: npt.NDArray[np.number],
    nan_policy: Literal["propagate", "omit", "raise"],
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    if nan_policy == "omit":
        if control.ndim == 1:
            return control[~np.isnan(control)], treatment[~np.isnan(treatment)]
        return (
            control[~np.isnan(control).any(axis=1)],
            treatment[~np.isnan(treatment).any(axis=1)],
        )

    if nan_policy == "raise" and (np.isnan(control).any() or np.isnan(treatment).any()):
        raise ValueError("Input contains nan.")

    return control, treatment
