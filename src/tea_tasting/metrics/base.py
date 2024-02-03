"""Metrics definitions."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import ibis
import ibis.expr.types
import pandas as pd

import tea_tasting.aggr


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class AggrCols(NamedTuple):
    """Columns to be aggregated for a metric analysis."""
    has_count: bool
    mean_cols: Sequence[str]
    var_cols: Sequence[str]
    cov_cols: Sequence[tuple[str, str]]


class MetricBaseAggr(abc.ABC):
    """Metric which is analyzed using aggregates."""
    @property
    @abc.abstractmethod
    def aggr_cols(self: MetricBaseAggr) -> AggrCols:
        """Columns to be aggregated for a metric analysis."""
        ...

    @overload
    @abc.abstractmethod
    def analyze(
        self: MetricBaseAggr,
        data: dict[Any, tea_tasting.aggr.Aggregates],
        control: Any,
        treatment: Any,
        variant_col: None = None,
    ) -> NamedTuple | dict[str, Any]:
        ...

    @overload
    @abc.abstractmethod
    def analyze(
        self: MetricBaseAggr,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any,
        treatment: Any,
        variant_col: str,
    ) -> NamedTuple | dict[str, Any]:
        ...

    @abc.abstractmethod
    def analyze(
        self: MetricBaseAggr,
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
    def validate_data(
        self: MetricBaseAggr,
        data: dict[Any, tea_tasting.aggr.Aggregates],
        variant_col: None = None,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        ...

    @overload
    def validate_data(
        self: MetricBaseAggr,
        data: pd.DataFrame | ibis.expr.types.Table,
        variant_col: str,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        ...

    def validate_data(
        self: MetricBaseAggr,
        data: pd.DataFrame | ibis.expr.types.Table | dict[
            Any, tea_tasting.aggr.Aggregates],
        variant_col: str | None = None,
    ) ->  dict[Any, tea_tasting.aggr.Aggregates]:
        """Validates experimental data.

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
            aggrs = tea_tasting.aggr.read_aggregates(
                data=table,
                group_col=variant_col,
                **self.aggr_cols._asdict(),
            )
        else:
            aggrs = table

        return aggrs


class MetricBaseFull(abc.ABC):
    """Metric which is analyzed using detailed data."""
    use_raw_data: bool = False

    @property
    @abc.abstractmethod
    def cols(self: MetricBaseFull) -> Sequence[str]:
        """Columns to be fetched for a metric analysis."""
        ...

    @abc.abstractmethod
    def analyze(
        self: MetricBaseFull,
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


MetricBase = MetricBaseAggr | MetricBaseFull
