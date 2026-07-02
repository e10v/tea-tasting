"""Ibis data backend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tea_tasting.backends.base import (
    _COUNT,
    _COV,
    _MEAN,
    _VAR,
    BaseTable,
    BaseTableGroupBy,
    _get_aggregates,
)
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import ibis.expr.types
    import pyarrow as pa

    import tea_tasting.aggr  # noqa: TC004


_CENTERED = "__centered__{}__"
_CENTERED_LEFT = "__centered__left__{}__{}__"
_CENTERED_RIGHT = "__centered_right__{}__{}__"


class IbisTable(BaseTable):  # noqa: D101
    def __init__(
        self,
        data: ibis.expr.types.Table,
        *,
        has_var: bool | None = None,
        has_cov: bool | None = None,
        chunk_size: int | None = 100_000,
    ) -> None:
        """Ibis table adapter.

        Args:
            data: Ibis Table.
            has_var: If `True`, assume that the backend supports sample variance.
                If `None`, use the Ibis backend's operation support.
            has_cov: If `True`, assume that the backend supports sample covariance.
                If `None`, use the Ibis backend's operation support.
            chunk_size: Chunk size for fetching data. If `None`, fetch all rows.
        """
        tea_tasting.utils.check_scalar(has_var, "has_var", typ=bool | None)
        tea_tasting.utils.check_scalar(has_cov, "has_cov", typ=bool | None)
        if chunk_size is not None:
            tea_tasting.utils.check_scalar(chunk_size, "chunk_size", typ=int, gt=0)

        if has_var is None or has_cov is None:
            import ibis  # noqa: PLC0415
            import ibis.expr.operations  # noqa: PLC0415

            backend = ibis.get_backend(data)
            if has_var is None:
                has_var = backend.has_operation(ibis.expr.operations.Variance)
            if has_cov is None:
                has_cov = backend.has_operation(ibis.expr.operations.Covariance)

        self.data = data
        self.has_var = has_var
        self.has_cov = has_cov
        self.chunk_size = chunk_size

    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """
        data = self.data.select(*cols) if len(cols) > 0 else self.data
        return _to_pyarrow(data, self.chunk_size)

    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """
        return _to_pyarrow(
            self.data.select(col).distinct(),
            self.chunk_size,
        )[col].to_pylist()

    def group_by(self, by: str) -> IbisTableGroupBy:
        """Group table by a column.

        Args:
            by: Column name to group by.

        Returns:
            Grouped Ibis table adapter.
        """
        return IbisTableGroupBy(self, by)

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
        return _get_aggregates(
            _aggregate(
                data=self.data,
                aggr_cols=aggr_cols,
                group_col=None,
                has_var=self.has_var,
                has_cov=self.has_cov,
                chunk_size=self.chunk_size,
            )[0],
            aggr_cols,
        )


class IbisTableGroupBy(BaseTableGroupBy):  # noqa: D101
    def __init__(
        self,
        ibis_table: IbisTable,
        by: str,
    ) -> None:
        """Grouped Ibis table adapter.

        Args:
            ibis_table: Ibis table adapter.
            by: Column name to group by.
        """
        self.ibis_table = ibis_table
        self.by = by

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
        ibis_table = self.ibis_table
        return {
            group_data[self.by]: _get_aggregates(group_data, aggr_cols)
            for group_data in _aggregate(
                data=ibis_table.data,
                aggr_cols=aggr_cols,
                group_col=self.by,
                has_var=ibis_table.has_var,
                has_cov=ibis_table.has_cov,
                chunk_size=ibis_table.chunk_size,
            )
        }


def _aggregate(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
    *,
    has_var: bool,
    has_cov: bool,
    chunk_size: int | None,
) -> list[dict[str, int | float]]:
    mean_cols = aggr_cols.mean_cols
    var_cols = aggr_cols.var_cols
    cov_cols = aggr_cols.cov_cols
    fallback_var_cols = () if has_var else var_cols
    fallback_cov_cols = () if has_cov else cov_cols
    if len(fallback_var_cols) > 0 or len(fallback_cov_cols) > 0:
        keep_cols = set(mean_cols)
        if len(fallback_var_cols) == 0:
            keep_cols.update(var_cols)
        if len(fallback_cov_cols) == 0:
            keep_cols.update(col for cols in cov_cols for col in cols)
        if group_col is not None:
            keep_cols.add(group_col)
        data = _add_centered_cols(
            data=data,
            group_col=group_col,
            keep_cols=keep_cols,
            var_cols=fallback_var_cols,
            cov_cols=fallback_cov_cols,
        )

    count_expr = {_COUNT: data.count()} if aggr_cols.has_count else {}
    mean_expr = {
        _MEAN.format(col): data[col].cast("float").mean()
        for col in mean_cols
    }
    var_expr = {
        _VAR.format(col): _sample_var(data, col, has_var=has_var)
        for col in var_cols
    }
    cov_expr = {
        _COV.format(left, right): _sample_cov(data, left, right, has_cov=has_cov)
        for left, right in cov_cols
    }
    all_expr = count_expr | mean_expr | var_expr | cov_expr

    grouped_data = data.group_by(group_col) if group_col is not None else data
    return _to_pyarrow(grouped_data.aggregate(**all_expr), chunk_size).to_pylist()  # ty:ignore[invalid-argument-type]


def _add_centered_cols(
    data: ibis.expr.types.Table,
    group_col: str | None,
    *,
    keep_cols: set[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> ibis.expr.types.Table:
    import ibis  # noqa: PLC0415

    null = ibis.null()

    stats_expr = {}
    for col in var_cols:
        col_expr = data[col].cast("float")
        stats_expr[_CENTERED.format(col)] = col_expr - _mean(data, col_expr, group_col)

    for left, right in cov_cols:
        valid = data[left].notnull() & data[right].notnull()
        left_expr = data[left].cast("float")
        right_expr = data[right].cast("float")
        stats_expr[_CENTERED_LEFT.format(left, right)] = (
            left_expr - _mean(data, valid.ifelse(left_expr, null), group_col)
        )
        stats_expr[_CENTERED_RIGHT.format(left, right)] = (
            right_expr - _mean(data, valid.ifelse(right_expr, null), group_col)
        )
    return data.select(*keep_cols, **stats_expr)


def _mean(
    data: ibis.expr.types.Table,
    expr: ibis.expr.types.NumericValue,
    group_col: str | None,
) -> ibis.expr.types.Value:
    import ibis  # noqa: PLC0415

    return expr.mean().over(ibis.window(  # ty:ignore[unresolved-attribute]
        group_by=data[group_col] if group_col is not None else None,
    ))


def _sample_var(
    data: ibis.expr.types.Table,
    col: str,
    *,
    has_var: bool,
) -> ibis.expr.types.Value:
    if has_var:
        return data[col].cast("float").var(how="sample")
    return _fallback_sample_aggr(
        data[_CENTERED.format(col)] * data[_CENTERED.format(col)],
    )


def _sample_cov(
    data: ibis.expr.types.Table,
    left: str,
    right: str,
    *,
    has_cov: bool,
) -> ibis.expr.types.Value:
    if has_cov:
        return data[left].cast("float").cov(data[right].cast("float"), how="sample")
    return _fallback_sample_aggr(
        data[_CENTERED_LEFT.format(left, right)] *
        data[_CENTERED_RIGHT.format(left, right)],
    )


def _fallback_sample_aggr(
    centered_expr: ibis.expr.types.Column,
) -> ibis.expr.types.Value:
    import ibis  # noqa: PLC0415

    count = centered_expr.count()
    return (count > 1).ifelse(centered_expr.sum() / (count - 1), ibis.null())  # ty:ignore[unsupported-operator]


def _to_pyarrow(
    data: ibis.expr.types.Table,
    chunk_size: int | None,
) -> pa.Table:
    if chunk_size is None:
        return data.to_pyarrow()
    return data.to_pyarrow_batches(chunk_size=chunk_size).read_all()
