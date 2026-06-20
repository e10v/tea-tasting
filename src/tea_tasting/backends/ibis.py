"""Ibis data backend adapter."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import tea_tasting.aggr
from tea_tasting.backends.base import BaseTable, BaseTableGroupBy


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import ibis.expr.types
    import pyarrow as pa


_COUNT = "__count__"
_MEAN = "__mean__{}__"
_VAR = "__var__{}__"
_COV = "__cov__{}__{}__"
_VAR_MEAN = "__var_mean__{}__"
_VAR_CENTERED = "__var_centered__{}__"
_COV_LEFT_MEAN = "__cov_left_mean__{}__{}__"
_COV_RIGHT_MEAN = "__cov_right_mean__{}__{}__"
_COV_CENTERED = "__cov_centered__{}__{}__"


class IbisTable(BaseTable):  # noqa: D101
    def __init__(
        self,
        data: ibis.expr.types.Table,
        *,
        has_var: bool | None = None,
        has_cov: bool | None = None,
    ) -> None:
        """Ibis table adapter.

        Args:
            data: Ibis Table.
            has_var: If `True`, assume that the backend supports sample variance.
                If `None`, use the Ibis backend's operation support.
            has_cov: If `True`, assume that the backend supports sample covariance.
                If `None`, use the Ibis backend's operation support.
        """
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
        return IbisTableGroupBy(
            self.data,
            by,
            has_var=self.has_var,
            has_cov=self.has_cov,
        )

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
            _aggregate(
                data=self.data,
                group_col=None,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
                has_var=self.has_var,
                has_cov=self.has_cov,
            )[0],
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )


class IbisTableGroupBy(BaseTableGroupBy):  # noqa: D101
    def __init__(
        self,
        data: ibis.expr.types.Table,
        by: str,
        *,
        has_var: bool,
        has_cov: bool,
    ) -> None:
        """Grouped Ibis table adapter.

        Args:
            data: Ibis Table.
            by: Column name to group by.
            has_var: If `True`, assume that the backend supports sample variance.
            has_cov: If `True`, assume that the backend supports sample covariance.
        """
        self.data = data
        self.by = by
        self.has_var = has_var
        self.has_cov = has_cov

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
            for group_data in _aggregate(
                data=self.data,
                group_col=self.by,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
                has_var=self.has_var,
                has_cov=self.has_cov,
            )
        }


def _aggregate(
    data: ibis.expr.types.Table,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
    has_var: bool,
    has_cov: bool,
) -> list[dict[str, object]]:
    fallback_var_cols = () if has_var else var_cols
    fallback_cov_cols = () if has_cov else cov_cols
    if len(fallback_var_cols) > 0 or len(fallback_cov_cols) > 0:
        keep_cols = {
            *mean_cols,
            *var_cols,
            *(col for cols in cov_cols for col in cols),
        }
        if group_col is not None:
            keep_cols.add(group_col)
        data = _add_fallback_aggr_cols(
            data=data,
            group_col=group_col,
            keep_cols=keep_cols,
            var_cols=fallback_var_cols,
            cov_cols=fallback_cov_cols,
        )

    count_expr = {_COUNT: data.count()} if has_count else {}
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
    return grouped_data.aggregate(**all_expr).to_pyarrow().to_pylist()  # ty:ignore[invalid-argument-type]


def _add_fallback_aggr_cols(
    data: ibis.expr.types.Table,
    group_col: str | None,
    *,
    keep_cols: set[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> ibis.expr.types.Table:
    import ibis  # noqa: PLC0415

    data = data.select(*keep_cols)
    null = ibis.null()

    stats_expr = {}
    for col in var_cols:
        col_expr = data[col].cast("float")
        stats_expr[_VAR_MEAN.format(col)] = _mean(data, col_expr, group_col)

    for left, right in cov_cols:
        valid = data[left].notnull() & data[right].notnull()
        left_expr = data[left].cast("float")
        right_expr = data[right].cast("float")
        stats_expr[_COV_LEFT_MEAN.format(left, right)] = (
            _mean(data, valid.ifelse(left_expr, null), group_col)
        )
        stats_expr[_COV_RIGHT_MEAN.format(left, right)] = (
            _mean(data, valid.ifelse(right_expr, null), group_col)
        )

    data = data.mutate(**stats_expr)

    centered_expr = {}
    for col in var_cols:
        col_expr = data[col].cast("float")
        diff = col_expr - data[_VAR_MEAN.format(col)]
        centered_expr[_VAR_CENTERED.format(col)] = (
            data[col].notnull().ifelse(diff * diff, null)
        )

    for left, right in cov_cols:
        left_diff = data[left].cast("float") - data[_COV_LEFT_MEAN.format(left, right)]
        right_diff = (
            data[right].cast("float") - data[_COV_RIGHT_MEAN.format(left, right)]
        )
        centered_expr[_COV_CENTERED.format(left, right)] = left_diff * right_diff

    return data.select(*keep_cols, **centered_expr)


def _mean(
    data: ibis.expr.types.Table,
    expr: ibis.expr.types.NumericValue,
    group_col: str | None,
) -> ibis.expr.types.Value:
    import ibis  # noqa: PLC0415

    if group_col is None:
        return expr.mean().as_scalar()  # ty: ignore[unresolved-attribute]
    return expr.mean().over(ibis.window(group_by=data[group_col]))  # ty:ignore[unresolved-attribute]


def _sample_var(
    data: ibis.expr.types.Table,
    col: str,
    *,
    has_var: bool,
) -> ibis.expr.types.Value:
    if has_var:
        return data[col].cast("float").var(how="sample")
    return _fallback_sample_aggr(data, _VAR_CENTERED.format(col))


def _sample_cov(
    data: ibis.expr.types.Table,
    left: str,
    right: str,
    *,
    has_cov: bool,
) -> ibis.expr.types.Value:
    if has_cov:
        return data[left].cast("float").cov(data[right].cast("float"), how="sample")
    return _fallback_sample_aggr(data, _COV_CENTERED.format(left, right))


def _fallback_sample_aggr(
    data: ibis.expr.types.Table,
    centered_alias: str,
) -> ibis.expr.types.Value:
    import ibis  # noqa: PLC0415

    centered_expr = data[centered_alias]
    count = centered_expr.count()
    return (count > 1).ifelse(centered_expr.sum() / (count - 1), ibis.null())  # ty:ignore[unsupported-operator]


def _get_aggregates(
    data: dict[str, object],
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=data[_COUNT] if has_count else None,  # ty:ignore[invalid-argument-type]
        mean_={col: _metric_value(data[_MEAN.format(col)]) for col in mean_cols},
        var_={col: _metric_value(data[_VAR.format(col)]) for col in var_cols},
        cov_={cols: _metric_value(data[_COV.format(*cols)]) for cols in cov_cols},
    )


def _metric_value(value: object) -> float:
    return math.nan if value is None else float(value)  # ty: ignore[invalid-argument-type]
