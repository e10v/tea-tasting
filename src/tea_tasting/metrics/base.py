"""Metrics base classes."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, NamedTuple, overload

import ibis
import ibis.expr.types
import pandas as pd

import tea_tasting._utils
import tea_tasting.aggr


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class AggrCols(NamedTuple):
    """Columns to be aggregated for a metric analysis.

    Attributes:
        has_count: If True, include the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.
    """
    has_count: bool
    mean_cols: Sequence[str]
    var_cols: Sequence[str]
    cov_cols: Sequence[tuple[str, str]]

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
                tea_tasting._utils.sorted_tuple(*cols)
                for cols in tuple({*self.cov_cols, *other.cov_cols})
            }),
        )


class MetricBaseAggregated(abc.ABC):
    """Metric which is analyzed using aggregates."""
    @property
    @abc.abstractmethod
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        ...

    @overload
    @abc.abstractmethod
    def analyze(
        self,
        data: dict[Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant_col: None = None,
    ) -> NamedTuple | dict[str, Any]:
        ...

    @overload
    @abc.abstractmethod
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant_col: str,
    ) -> NamedTuple | dict[str, Any]:
        ...

    @abc.abstractmethod
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant_col: str | None = None,
    ) -> NamedTuple | dict[str, Any]:
        """Analyze metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant_col: Variant column name.

        Returns:
            Experiment results for a metric.
        """
        ...

    @overload
    def validate_aggregates(
        self,
        data: dict[Any, tea_tasting.aggr.Aggregates],
        variant_col: None = None,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        ...

    @overload
    def validate_aggregates(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        variant_col: str,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        ...

    def validate_aggregates(
        self,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        variant_col: str | None = None,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        """Validate aggregated experimental data.

        Reads aggregates if data is not a dictionary of Aggregates.

        Args:
            data: Experimental data.
            variant_col: Variant column name.

        Raises:
            ValueError: variant_col is None, while aggregated data are not provided.

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
                **self.aggr_cols._asdict(),
            )

        return table


class MetricBaseGranular(abc.ABC):
    """Metric which is analyzed using granular data."""
    use_raw_data: bool = False

    @property
    @abc.abstractmethod
    def cols(self) -> Sequence[str]:
        """Columns to be fetched for a metric analysis."""
        ...

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


MetricBase = MetricBaseAggregated | MetricBaseGranular
