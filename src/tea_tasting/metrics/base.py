"""Base classes for metrics."""

from __future__ import annotations

import abc
from collections import UserList
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeVar,
    overload,
)

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Literal

    import narwhals.typing  # noqa: TC004
    import numpy.typing as npt


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
        data: narwhals.typing.IntoFrame,
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
        data: narwhals.typing.IntoFrame,
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


class AggrCols(NamedTuple):
    """Columns to be aggregated for a metric analysis.

    Attributes:
        has_count: If `True`, include the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.
    """
    has_count: bool = False
    mean_cols: Sequence[str] = ()
    var_cols: Sequence[str] = ()
    cov_cols: Sequence[tuple[str, str]] = ()

    def __or__(self, other: AggrCols) -> AggrCols:
        """Merge two aggregation column specifications.

        Args:
            other: Second object.

        Returns:
            Merged column specifications.
        """
        return AggrCols(
            has_count=self.has_count or other.has_count,
            mean_cols=tuple({*self.mean_cols, *other.mean_cols}),
            var_cols=tuple({*self.var_cols, *other.var_cols}),
            cov_cols=tuple({
                tea_tasting.aggr._sorted_tuple(*cols)
                for cols in tuple({*self.cov_cols, *other.cov_cols})
            }),
        )

    def __len__(self) -> int:  # ty:ignore[invalid-method-override]
        """Total length of all object attributes.

        If has_count is True then its value is 1, or 0 otherwise.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )


class _HasAggrCols(abc.ABC):
    @property
    @abc.abstractmethod
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for an analysis."""


class MetricBaseAggregated(MetricBase[MetricResultT], _HasAggrCols):
    """Base class for metrics, which are analyzed using aggregated statistics."""
    @overload
    def analyze(
        self,
        data: dict[Hashable, tea_tasting.aggr.Aggregates],
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        ...

    @overload
    def analyze(
        self,
        data: narwhals.typing.IntoFrame,
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> MetricResultT:
        ...

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | dict[Hashable, tea_tasting.aggr.Aggregates],
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
        aggr = aggregate_by_variants(
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
        data: narwhals.typing.IntoFrame | tea_tasting.aggr.Aggregates,
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
            data = tea_tasting.aggr.read_aggregates(
                data=data,
                group_col=None,
                **self.aggr_cols._asdict(),
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


def aggregate_by_variants(
    data: narwhals.typing.IntoFrame | dict[Hashable, tea_tasting.aggr.Aggregates],
    aggr_cols: AggrCols,
    variant: str | None = None,
) ->  dict[Hashable, tea_tasting.aggr.Aggregates]:
    """Aggregate experimental data by variants.

    Args:
        data: Experimental data.
        aggr_cols: Columns to be aggregated.
        variant: Variant column name.

    Returns:
        Experimental data as a dictionary of Aggregates.
    """
    if isinstance(data, dict):
        return data  # ty:ignore[invalid-return-type]

    if variant is None:
        raise ValueError("The variant parameter is required but was not provided.")

    return tea_tasting.aggr.read_aggregates(
        data=data,
        group_col=variant,
        **aggr_cols._asdict(),
    )


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
        data: dict[Hashable, pa.Table],
        control: Hashable,
        treatment: Hashable,
        variant: str | None = None,
    ) -> MetricResultT:
        ...

    @overload
    def analyze(
        self,
        data: narwhals.typing.IntoFrame,
        control: Hashable,
        treatment: Hashable,
        variant: str,
    ) -> MetricResultT:
        ...

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | dict[Hashable, pa.Table],
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
        dfs = read_granular(
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


@overload
def read_granular(
    data: narwhals.typing.IntoFrame | narwhals.typing.Frame,
    cols: Sequence[str] = (),
    variant: None = None,
) -> pa.Table:
    ...

@overload
def read_granular(
    data: dict[Hashable, pa.Table],
    cols: Sequence[str] = (),
    variant: None = None,
) -> dict[Hashable, pa.Table]:
    ...

@overload
def read_granular(
    data: narwhals.typing.IntoFrame | narwhals.typing.Frame | dict[Hashable, pa.Table],
    cols: Sequence[str],
    variant: str,
) -> dict[Hashable, pa.Table]:
    ...

def read_granular(
    data: narwhals.typing.IntoFrame | narwhals.typing.Frame | dict[Hashable, pa.Table],
    cols: Sequence[str] = (),
    variant: str | None = None,
) -> pa.Table | dict[Hashable, pa.Table]:
    """Read granular experimental data.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant: Variant column name.

    Returns:
        Experimental data as a dictionary of PyArrow Tables.
    """
    if isinstance(data, dict):
        return data

    variant_cols = () if variant is None else (variant,)
    if tea_tasting.utils._is_ibis_table(data):
        if len(cols) + len(variant_cols) > 0:
            data = data.select(*cols, *variant_cols)
        table = data.to_pyarrow()
    else:
        data = nw.from_native(data)
        if isinstance(data, nw.LazyFrame):
            data = data.collect()
        if len(cols) + len(variant_cols) > 0:
            data = data.select(*cols, *variant_cols)
        table = data.to_arrow()

    if variant is None:
        return table

    variant_array = table[variant]
    if len(cols) > 0:
        table = table.select(cols)
    return {
        var: table.filter(pc.equal(variant_array, pa.scalar(var)))  # ty:ignore[unresolved-attribute]
        for var in variant_array.unique().to_pylist()
    }
