"""Ibis data backend adapter."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from tea_tasting.backends.base import (
    _COUNT,
    _COV,
    _DEMEAN,
    _MEAN,
    _VAR,
    BaseTable,
    BaseTableGroupBy,
    _get_aggregates,
)


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import ibis.expr.types
    import pyarrow as pa

    import tea_tasting.aggr


class IbisTable(BaseTable):
    """Ibis table adapter."""

    def __init__(self, data: ibis.expr.types.Table) -> None:
        """Create an Ibis table adapter.

        Args:
            data: Ibis Table.
        """
        self.data = data

    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """
        data = self.data.select(*cols) if len(cols) > 0 else self.data
        return data.to_pyarrow()

    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """
        return self.data.select(col).distinct().to_pyarrow()[col].to_pylist()

    def group_by(self, by: str) -> IbisTableGroupBy:
        """Group table by a column.

        Args:
            by: Column name to group by.

        Returns:
            Grouped Ibis table adapter.
        """
        return IbisTableGroupBy(self.data, by)

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
        return _get_aggregates(
            _read_aggr_ibis(
                data=self.data,
                group_col=None,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
            )[0],
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )


class IbisTableGroupBy(BaseTableGroupBy):
    """Grouped Ibis table adapter."""

    def __init__(self, data: ibis.expr.types.Table, by: str) -> None:
        """Create a grouped Ibis table adapter.

        Args:
            data: Ibis Table.
            by: Column name to group by.
        """
        self.data = data
        self.by = by

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
        return {
            group_data[self.by]: _get_aggregates(
                group_data,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
            )
            for group_data in _read_aggr_ibis(
                data=self.data,
                group_col=self.by,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
            )
        }


def _read_aggr_ibis(
    data: ibis.expr.types.Table,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> list[dict[str, int | float]]:
    import ibis  # noqa: PLC0415
    import ibis.expr.operations  # noqa: PLC0415

    covar_cols = tuple({*var_cols, *itertools.chain(*cov_cols)})
    backend = ibis.get_backend(data)
    var_op = ibis.expr.operations.Variance
    cov_op = ibis.expr.operations.Covariance
    if backend.has_operation(var_op) and backend.has_operation(cov_op):
        var_expr = {
            _VAR.format(col): data[col].cast("float").var(how="sample")
            for col in var_cols
        }
        cov_expr = {
            _COV.format(left, right): data[left].cast("float").cov(
                data[right].cast("float"),
                how="sample",
            )
            for left, right in cov_cols
        }
    else:
        if len(covar_cols) > 0:
            demean_expr = {
                _DEMEAN.format(col): data[col] - data[col].cast("float").mean()
                for col in covar_cols
            }
            grouped_data = data.group_by(group_col) if group_col is not None else data
            data = grouped_data.mutate(**demean_expr)

        var_expr = {
            _VAR.format(col): (
                data[_DEMEAN.format(col)] * data[_DEMEAN.format(col)]
            ).sum() / (data.count() - 1)
            for col in var_cols
        }
        cov_expr = {
            _COV.format(left, right): (
                data[_DEMEAN.format(left)] * data[_DEMEAN.format(right)]
            ).sum() / (data.count() - 1)
            for left, right in cov_cols
        }

    count_expr = {_COUNT: data.count()} if has_count else {}
    mean_expr = {
        _MEAN.format(col): data[col].cast("float").mean()
        for col in mean_cols
    }
    all_expr = count_expr | mean_expr | var_expr | cov_expr

    grouped_data = data.group_by(group_col) if group_col is not None else data
    return grouped_data.aggregate(**all_expr).to_pyarrow().to_pylist()  # ty:ignore[invalid-argument-type]
