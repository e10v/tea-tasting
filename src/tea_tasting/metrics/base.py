"""Base classes for metrics."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, Union, overload

import ibis
import ibis.expr.types
import pandas as pd

import tea_tasting.aggr
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence


# The | operator doesn't work for NamedTuple, but Union works.
MetricResult = Union[NamedTuple, dict[str, Any]]  # noqa: UP007

R = TypeVar("R", bound=MetricResult)


class MetricBase(abc.ABC, Generic[R], tea_tasting.utils.ReprMixin):
    """Base class for metrics."""
    @abc.abstractmethod
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant: str,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        ...


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
            other: Second objects.

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

    def __len__(self) -> int:
        """Total length of all object attributes.

        If has_count is True then its value is 1, or 0 otherwise.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )


class MetricBaseAggregated(MetricBase[R]):
    """Base class for metrics, which are analyzed using aggregated statistics."""
    @property
    @abc.abstractmethod
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        ...

    @overload
    def analyze(
        self,
        data: dict[Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
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
    ) -> R:
        """Analyze metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        ...


def aggregate_by_variants(
    data: pd.DataFrame | ibis.expr.types.Table | dict[Any, tea_tasting.aggr.Aggregates],
    aggr_cols: AggrCols,
    variant: str | None = None,
) ->  dict[Any, tea_tasting.aggr.Aggregates]:
    """Aggregate experimental data by variants.

    Args:
        data: Experimental data.
        aggr_cols: Columns to be aggregated.
        variant: Variant column name.

    Raises:
        ValueError: The variant parameter is required but was not provided.
        TypeError: data is not an instance of DataFrame, Table,
            or a dictionary of Aggregates.

    Returns:
        Experimental data as a dictionary of Aggregates.
    """
    if isinstance(data, dict) and all(
        isinstance(v, tea_tasting.aggr.Aggregates) for v in data.values()  # type: ignore
    ):
        return data

    if variant is None:
        raise ValueError("The variant parameter is required but was not provided.")

    if not isinstance(data, pd.DataFrame | ibis.expr.types.Table):
        raise TypeError(
            f"data is a {type(data)}, but must be an instance of"
            " DataFrame, Table, or a dictionary of Aggregates.",
        )

    return tea_tasting.aggr.read_aggregates(
        data=data,
        group_col=variant,
        **aggr_cols._asdict(),
    )


class MetricBaseGranular(MetricBase[R]):
    """Base class for metrics, which are analyzed using granular data."""
    @property
    @abc.abstractmethod
    def cols(self) -> Sequence[str]:
        """Columns to be fetched for a metric analysis."""
        ...

    @overload
    def analyze(
        self,
        data: dict[Any, pd.DataFrame],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[Any, pd.DataFrame],
        control: Any,
        treatment: Any,
        variant: str | None = None,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        dfs = read_dataframes(
            data,
            cols=self.cols,
            variant=variant,
        )
        return self.analyze_dataframes(
            control=dfs[control],
            treatment=dfs[treatment],
        )

    @abc.abstractmethod
    def analyze_dataframes(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
    ) -> R:
        """Analyze metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """
        ...


def read_dataframes(
    data: pd.DataFrame | ibis.expr.types.Table | dict[Any, pd.DataFrame],
    cols: Sequence[str],
    variant: str | None = None,
) -> dict[Any, pd.DataFrame]:
    """Read granular experimental data.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant: Variant column name.

    Raises:
        ValueError: The variant parameter is required but was not provided.
        TypeError: data is not an instance of DataFrame, Table,
            or a dictionary if DataFrames.

    Returns:
        Experimental data as a dictionary of DataFrames.
    """
    if isinstance(data, dict) and all(
        isinstance(v, pd.DataFrame) for v in data.values()  # type: ignore
    ):
        return data

    if variant is None:
        raise ValueError("The variant parameter is required but was not provided.")

    if isinstance(data, ibis.expr.types.Table):
        data = data.select(*cols, variant).to_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data is a {type(data)}, but must be an instance of"
            " DataFrame, Table, or a dictionary if DataFrames.",
        )

    return dict(tuple(data.loc[:, [*cols, variant]].groupby(variant)))
