"""Base classes for data backend adapters."""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, overload

import tea_tasting.aggr


if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import pyarrow as pa


_COUNT = "__count__"
_MEAN = "__mean__{}__"
_VAR = "__var__{}__"
_COV = "__cov__{}__{}__"


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
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> tea_tasting.aggr.Aggregates:
        """Aggregate table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics.
        """


class BaseTableGroupBy(abc.ABC):
    """Base class for grouped data backend table adapters."""

    @abc.abstractmethod
    def aggregate(
        self,
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        """Aggregate grouped table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics by group value.
        """


@overload
def _get_aggregates(
    data: Sequence[Mapping[str, object]],
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def _get_aggregates(
    data: Sequence[Mapping[str, object]],
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def _get_aggregates(
    data: Sequence[Mapping[str, object]],
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None = None,
) -> tea_tasting.aggr.Aggregates | dict[Hashable, tea_tasting.aggr.Aggregates]:
    if group_col is None:
        return _get_row_aggregates(_ResultRow(data[0]), aggr_cols)

    return {
        row[group_col]: _get_row_aggregates(row, aggr_cols)
        for row in (_ResultRow(row_data) for row_data in data)
    }


class _ResultRow:
    def __init__(self, data: Mapping[str, object]) -> None:
        self.data = data
        self.casefold_keys = _casefold_keys(data)

    def __getitem__(self, key: str) -> object:
        if key in self.data:
            return self.data[key]

        result_key = self.casefold_keys[key.casefold()]
        if result_key is None:
            raise KeyError(f"Ambiguous result column name: {key!r}")
        return self.data[result_key]


def _casefold_keys(data: Mapping[str, object]) -> dict[str, str | None]:
    casefold_keys: dict[str, str | None] = {}
    for key in data:
        casefold_key = key.casefold()
        if casefold_key in casefold_keys:
            casefold_keys[casefold_key] = None
        else:
            casefold_keys[casefold_key] = key
    return casefold_keys


def _get_row_aggregates(
    data: _ResultRow,
    aggr_cols: tea_tasting.aggr.AggrCols,
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=int(data[_COUNT]) if aggr_cols.has_count else None,  # ty:ignore[invalid-argument-type]
        mean_={col: _float(data[_MEAN.format(col)]) for col in aggr_cols.mean_cols},
        var_={col: _float(data[_VAR.format(col)]) for col in aggr_cols.var_cols},
        cov_={cols: _float(data[_COV.format(*cols)]) for cols in aggr_cols.cov_cols},
    )


def _float(value: object) -> float:
    return math.nan if value is None else float(value)  # ty: ignore[invalid-argument-type]
