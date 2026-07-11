"""Ibis data backend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

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
    from collections.abc import Hashable

    import ibis.expr.types
    import pyarrow as pa

    import tea_tasting.aggr  # noqa: TC004


_CENTERED_SQUARE = "__centered_square__{}__"
_CENTERED_PRODUCT = "__centered_product__{}__{}__"


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
            chunk_size: Chunk size for fetching data. Used only in the `select` method.
                If `None`, fetch all rows.
        """
        import ibis  # noqa: PLC0415
        import ibis.expr.operations  # noqa: PLC0415

        tea_tasting.utils.check_scalar(has_var, "has_var", typ=bool | None)
        tea_tasting.utils.check_scalar(has_cov, "has_cov", typ=bool | None)
        if chunk_size is not None:
            tea_tasting.utils.check_scalar(chunk_size, "chunk_size", typ=int, gt=0)

        if has_var is None or has_cov is None:
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
        if self.chunk_size is None:
            return data.to_pyarrow()
        return data.to_pyarrow_batches(chunk_size=self.chunk_size).read_all()

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
        return _aggregate(self, aggr_cols)


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
        return _aggregate(self.ibis_table, aggr_cols, self.by)


@overload
def _aggregate(
    ibis_table: IbisTable,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def _aggregate(
    ibis_table: IbisTable,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def _aggregate(
    ibis_table: IbisTable,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None = None,
) -> tea_tasting.aggr.Aggregates | dict[Hashable, tea_tasting.aggr.Aggregates]:
    data = ibis_table.data
    fallback_var_cols = () if ibis_table.has_var else aggr_cols.var_cols
    fallback_cov_cols = () if ibis_table.has_cov else aggr_cols.cov_cols
    if len(fallback_var_cols) > 0 or len(fallback_cov_cols) > 0:
        data = _add_centered_product_cols(
            data=data,
            aggr_cols=aggr_cols,
            group_col=group_col,
            var_cols=fallback_var_cols,
            cov_cols=fallback_cov_cols,
        )

    exprs = _aggr_exprs(
        data,
        aggr_cols,
        has_var=ibis_table.has_var,
        has_cov=ibis_table.has_cov,
    )
    grouped_data = data.group_by(group_col) if group_col is not None else data
    result = grouped_data.aggregate(exprs).to_pyarrow().to_pylist()  # ty:ignore[invalid-argument-type]
    return _get_aggregates(result, aggr_cols, group_col)


def _add_centered_product_cols(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
    *,
    var_cols: tuple[str, ...],
    cov_cols: tuple[tuple[str, str], ...],
) -> ibis.expr.types.Table:
    import ibis  # noqa: PLC0415

    null = ibis.null()
    keep_cols = set(aggr_cols.mean_cols)
    if len(var_cols) == 0:
        keep_cols.update(aggr_cols.var_cols)
    if len(cov_cols) == 0:
        keep_cols.update(col for cols in aggr_cols.cov_cols for col in cols)
    if group_col is not None:
        keep_cols.add(group_col)

    exprs: list[ibis.expr.types.Value] = [data[col] for col in keep_cols]
    for col in var_cols:
        col_expr = data[col].cast("float")
        centered_col_expr = col_expr - _mean_over(data, col_expr, group_col)
        exprs.append(
            (centered_col_expr * centered_col_expr)
                .name(_CENTERED_SQUARE.format(col)),
        )

    for left, right in cov_cols:
        valid = data[left].notnull() & data[right].notnull()
        left_expr = data[left].cast("float")
        right_expr = data[right].cast("float")
        centered_left = (
            left_expr - _mean_over(data, valid.ifelse(left_expr, null), group_col)
        )
        centered_right = (
            right_expr - _mean_over(data, valid.ifelse(right_expr, null), group_col)
        )
        exprs.append(
            (centered_left * centered_right)
                .name(_CENTERED_PRODUCT.format(left, right)),
        )
    return data.select(*exprs)


def _aggr_exprs(
    data: ibis.expr.types.Table,
    aggr_cols: tea_tasting.aggr.AggrCols,
    *,
    has_var: bool,
    has_cov: bool,
) -> list[ibis.expr.types.Value]:
    exprs: list[ibis.expr.types.Value] = []
    if aggr_cols.has_count:
        exprs.append(data.count().name(_COUNT))
    exprs.extend(
        data[col].cast("float").mean().name(_MEAN.format(col))
        for col in aggr_cols.mean_cols
    )
    exprs.extend(
        _sample_var(data, col, has_var=has_var).name(_VAR.format(col))
        for col in aggr_cols.var_cols
    )
    exprs.extend(
        _sample_cov(data, left, right, has_cov=has_cov).name(_COV.format(left, right))
        for left, right in aggr_cols.cov_cols
    )
    return exprs


def _mean_over(
    data: ibis.expr.types.Table,
    expr: ibis.expr.types.Value,
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
    return _fallback_sample_aggr(data[_CENTERED_SQUARE.format(col)])


def _sample_cov(
    data: ibis.expr.types.Table,
    left: str,
    right: str,
    *,
    has_cov: bool,
) -> ibis.expr.types.Value:
    if has_cov:
        return data[left].cast("float").cov(data[right].cast("float"), how="sample")
    return _fallback_sample_aggr(data[_CENTERED_PRODUCT.format(left, right)])


def _fallback_sample_aggr(
    product_expr: ibis.expr.types.Column,
) -> ibis.expr.types.Value:
    import ibis  # noqa: PLC0415

    count = product_expr.count()
    return (count > 1).ifelse(product_expr.sum() / (count - 1), ibis.null())  # ty:ignore[unsupported-operator]
