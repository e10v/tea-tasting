"""Base classes for data backend adapters."""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING

import tea_tasting.aggr


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import pyarrow as pa


_COUNT = "__count__"
_MEAN = "__mean__{}__"
_VAR = "__var__{}__"
_COV = "__cov__{}__{}__"
_DEMEAN = "__demean__{}__"


class BaseTable(abc.ABC):
    """Base class for data backend table adapters."""

    @abc.abstractmethod
    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """

    @abc.abstractmethod
    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """

    @abc.abstractmethod
    def group_by(self, by: str) -> BaseTableGroupBy:
        """Group table by a column.

        Args:
            by: Column name to group by.

        Returns:
            Grouped table adapter.
        """

    @abc.abstractmethod
    def aggregate(
        self,
        *,
        has_count: bool,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> tea_tasting.aggr.Aggregates:
        """Aggregate table data.

        Args:
            has_count: If `True`, calculate the sample size.
            mean_cols: Column names for calculation of sample means.
            var_cols: Column names for calculation of sample variances.
            cov_cols: Pairs of column names for calculation of sample covariances.

        Returns:
            Aggregated statistics.
        """


class BaseTableGroupBy(abc.ABC):
    """Base class for grouped data backend table adapters."""

    @abc.abstractmethod
    def aggregate(
        self,
        *,
        has_count: bool,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        """Aggregate grouped table data.

        Args:
            has_count: If `True`, calculate the sample size.
            mean_cols: Column names for calculation of sample means.
            var_cols: Column names for calculation of sample variances.
            cov_cols: Pairs of column names for calculation of sample covariances.

        Returns:
            Aggregated statistics by group value.
        """


def _get_aggregates(
    data: dict[str, float | int],
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=int(data[_COUNT]) if has_count else None,
        mean_={col: _float(data[_MEAN.format(col)]) for col in mean_cols},
        var_={col: _float(data[_VAR.format(col)]) for col in var_cols},
        cov_={cols: _float(data[_COV.format(*cols)]) for cols in cov_cols},
    )


def _float(value: object) -> float:
    return math.nan if value is None else float(value)  # ty: ignore[invalid-argument-type]
