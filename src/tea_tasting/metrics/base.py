"""Metrics base classes."""

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


class MetricBase(abc.ABC, tea_tasting.utils.ReprMixin):
    """Base class for metrics."""
    @abc.abstractmethod
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant_col: str,
    ) -> NamedTuple | dict[str, Any]:
        """Analyzes metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant_col: Variant column.

        Returns:
            Experiment results for a metric.
        """
        ...


class AggrCols(NamedTuple):
    """Columns to be aggregated for a metric analysis.

    Attributes:
        has_count: If True, include the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.
    """
    has_count: bool = False
    mean_cols: Sequence[str] = ()
    var_cols: Sequence[str] = ()
    cov_cols: Sequence[tuple[str, str]] = ()

    def __or__(self, other: AggrCols) -> AggrCols:
        """Combine columns. Exclude duplicates.

        Args:
            other: Second objects with columns to be aggregated for a metric analysis.

        Returns:
            Combined columns to be aggregated for a metric analysis.
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

        If has_count is True then its value is 1, otherwise 0.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )


# The | operator doesn't work for NamedTuple, but Union works.
R = TypeVar("R", bound=Union[NamedTuple, dict[str, Any]])  # noqa: UP007

class MetricBaseAggregated(MetricBase, Generic[R]):
    """Base class for metrics analyzed using aggregates."""
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
        variant_col: None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant_col: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant_col: str | None = None,
    ) -> R:
        """Analyze metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant_col: Variant column name.

        Returns:
            Experiment results for a metric.
        """
        aggr = aggregate_by_variants(
            data,
            aggr_cols=self.aggr_cols,
            variant_col=variant_col,
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
            Experiment results for a metric.
        """
        ...


def aggregate_by_variants(
    data: pd.DataFrame | ibis.expr.types.Table | dict[Any, tea_tasting.aggr.Aggregates],
    aggr_cols: AggrCols,
    variant_col: str | None = None,
) ->  dict[Any, tea_tasting.aggr.Aggregates]:
    """Validate aggregated experimental data.

    Reads aggregates if data is not a dictionary of Aggregates.

    Args:
        data: Experimental data.
        aggr_cols: Columns to aggregate.
        variant_col: Variant column name.

    Raises:
        ValueError: variant_col is None, while aggregated data are not provided.
        TypeError: data is not an instance of DataFrame, Table,
            or a dictionary if Aggregates.

    Returns:
        Experimental data as a dictionary of Aggregates.
    """
    if isinstance(data, pd.DataFrame):
        con = ibis.pandas.connect()
        table = con.create_table("data", data)
    else:
        table = data

    if isinstance(table, ibis.expr.types.Table):
        if variant_col is None:
            raise ValueError(
                "variant_col is None, but should be an instance of str.")
        return tea_tasting.aggr.read_aggregates(
            data=table,
            group_col=variant_col,
            **aggr_cols._asdict(),
        )

    if not isinstance(table, dict) or not all(  # type: ignore
        isinstance(v, tea_tasting.aggr.Aggregates) for v in table.values()  # type: ignore
    ):
        raise TypeError(
            f"data is a {type(data)}, but must be an instance of"
            " DataFrame, Table, or a dictionary if Aggregates.",
        )

    return table


class MetricBaseGranular(MetricBase, Generic[R]):
    """Base class for metrics analyzed using granular data."""
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
        variant_col: None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant_col: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[Any, pd.DataFrame],
        control: Any,
        treatment: Any,
        variant_col: str | None = None,
    ) -> R:
        """Analyze metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant_col: Variant column name.

        Returns:
            Experiment results for a metric.
        """
        dfs = read_dataframes(
            data,
            cols=self.cols,
            variant_col=variant_col,
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
            Experiment results for a metric.
        """
        ...


def read_dataframes(
    data: pd.DataFrame | ibis.expr.types.Table | dict[Any, pd.DataFrame],
    cols: Sequence[str],
    variant_col: str | None = None,
) -> dict[Any, pd.DataFrame]:
    """Validate granular experimental data.

    Reads granular data if data is not a dictionary of DataFrames.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant_col: Variant column name.

    Raises:
        ValueError: variant_col is None, while aggregated data are not provided.
        TypeError: data is not an instance of DataFrame, Table,
            or a dictionary if DataFrames.

    Returns:
        Experimental data as a dictionary of DataFrames.
    """
    if isinstance(data, dict) and all(
        isinstance(v, pd.DataFrame) for v in data.values()  # type: ignore
    ):
        return data

    if variant_col is None:
        raise ValueError("variant_col is None, but should be an instance of str.")

    if isinstance(data, ibis.expr.types.Table):
        data = data.select(*cols, variant_col).to_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data is a {type(data)}, but must be an instance of"
            " DataFrame, Table, or a dictionary if DataFrames.",
        )

    return dict(tuple(data.loc[:, [*cols, variant_col]].groupby(variant_col)))
