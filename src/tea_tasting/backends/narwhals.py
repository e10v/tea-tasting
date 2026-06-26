"""Narwhals data backend adapter."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import narwhals as nw
import narwhals.typing

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

    import pyarrow as pa

    import tea_tasting.aggr


class NarwhalsFrame(BaseTable):  # noqa: D101
    def __init__(
        self,
        data: narwhals.typing.IntoFrame | narwhals.typing.Frame,
    ) -> None:
        """Narwhals-compatible frame adapter.

        Args:
            data: Narwhals-compatible native frame.
        """
        self.data = data

    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """
        data = nw.from_native(self.data)
        if len(cols) > 0:
            data = data.select(*cols)
        if isinstance(data, nw.LazyFrame):
            data = data.collect()
        return data.to_arrow()

    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """
        data = nw.from_native(self.data)
        if not isinstance(data, nw.LazyFrame):
            data = data.lazy()
        return data.unique(col).collect().get_column(col).to_list()

    def group_by(self, by: str) -> NarwhalsFrameGroupBy:
        """Group table by a column.

        Args:
            by: Column name to group by.

        Returns:
            Grouped Narwhals-compatible frame adapter.
        """
        return NarwhalsFrameGroupBy(self, by)

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
            _read_aggr_narwhals(
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


class NarwhalsFrameGroupBy(BaseTableGroupBy):  # noqa: D101
    def __init__(
        self,
        narwhals_frame: NarwhalsFrame,
        by: str,
    ) -> None:
        """Grouped Narwhals-compatible frame adapter.

        Args:
            narwhals_frame: Narwhals-compatible frame adapter.
            by: Column name to group by.
        """
        self.narwhals_frame = narwhals_frame
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
            for group_data in _read_aggr_narwhals(
                data=self.narwhals_frame.data,
                group_col=self.by,
                has_count=has_count,
                mean_cols=mean_cols,
                var_cols=var_cols,
                cov_cols=cov_cols,
            )
        }


def _read_aggr_narwhals(
    data: narwhals.typing.IntoFrame | narwhals.typing.Frame,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> list[dict[str, int | float]]:
    frame = nw.from_native(data)
    if not isinstance(frame, nw.LazyFrame):
        frame = frame.lazy()

    covar_cols = tuple({*var_cols, *itertools.chain(*cov_cols)})
    if len(covar_cols) > 0:
        frame = (
            frame.with_columns(**{
                _DEMEAN.format(col): _nw_demean(col, group_col)
                for col in covar_cols
            })
            .with_columns(
                **{
                    _VAR.format(col):
                        nw.col(_DEMEAN.format(col)) * nw.col(_DEMEAN.format(col))
                    for col in var_cols
                },
                **{
                    _COV.format(left, right):
                        nw.col(_DEMEAN.format(left)) * nw.col(_DEMEAN.format(right))
                    for left, right in cov_cols
                },
            )
        )

    count_expr = (
        {_COUNT: nw.len()}
        if has_count or len(covar_cols) > 0
        else {}
    )
    mean_expr = {
        _MEAN.format(col): nw.col(col).mean()
        for col in mean_cols
    }
    var_expr = {
        _VAR.format(col): nw.col(_VAR.format(col)).mean()
        for col in var_cols
    }
    cov_expr = {
        _COV.format(left, right): nw.col(_COV.format(left, right)).mean()
        for left, right in cov_cols
    }
    all_expr = count_expr | mean_expr | var_expr | cov_expr

    aggr_data = (
        frame.select(**all_expr) if group_col is None
        else frame.group_by(group_col).agg(**all_expr)
    )
    if len(covar_cols) > 0:
        aggr_data = aggr_data.with_columns(
            **{
                _VAR.format(col): nw.col(_VAR.format(col)) / (1 - 1/nw.col(_COUNT))
                for col in var_cols
            },
            **{
                _COV.format(left, right):
                    nw.col(_COV.format(left, right)) / (1 - 1/nw.col(_COUNT))
                for left, right in cov_cols
            },
        )

    return aggr_data.collect().to_arrow().to_pylist()


def _nw_demean(col: str, group_col: str | None) -> nw.Expr:
    if group_col is None:
        return nw.col(col) - nw.col(col).mean()
    return nw.col(col) - nw.col(col).mean().over(group_col)
