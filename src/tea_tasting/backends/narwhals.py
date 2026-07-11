"""Narwhals data backend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import narwhals as nw
import narwhals.typing

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
    from typing import Literal

    import pyarrow as pa

    import tea_tasting.aggr  # noqa: TC004

    type CovarianceSupport = Literal["full", "ungrouped_only", "none", "auto"]


_CENTERED_PRODUCT = "__centered_product__{}__{}__"
_COV_COUNT = "__cov_count__{}__{}__"
_COV_SUM = "__cov_sum__{}__{}__"


IMPLEMENTATION_COV: dict[nw.Implementation, CovarianceSupport] = {
    nw.Implementation.CUDF: "ungrouped_only",
    nw.Implementation.DASK: "ungrouped_only",
    nw.Implementation.DUCKDB: "full",
    nw.Implementation.IBIS: "full",
    nw.Implementation.MODIN: "ungrouped_only",
    nw.Implementation.PANDAS: "ungrouped_only",
    nw.Implementation.POLARS: "full",
    nw.Implementation.PYARROW: "ungrouped_only",
    nw.Implementation.PYSPARK: "full",
    nw.Implementation.PYSPARK_CONNECT: "full",
    nw.Implementation.SQLFRAME: "full",
    nw.Implementation.UNKNOWN: "none",
}


class NarwhalsFrame(BaseTable):  # noqa: D101
    def __init__(
        self,
        data: narwhals.typing.IntoFrame | narwhals.typing.Frame,
        *,
        cov: CovarianceSupport = "auto",
    ) -> None:
        """Narwhals-compatible frame adapter.

        Args:
            data: Narwhals-compatible native frame.
            cov: Sample covariance support. Use `"full"` for grouped and
                ungrouped support, `"ungrouped_only"` for ungrouped support,
                `"none"` for no support, or `"auto"` to infer support from the
                data implementation.
        """
        frame = nw.from_native(data)
        self._frame = frame if isinstance(frame, nw.LazyFrame) else frame.lazy()
        self.cov: CovarianceSupport = (
            IMPLEMENTATION_COV.get(self._frame.implementation, "none")
            if cov == "auto"
            else tea_tasting.utils.check_scalar(
                cov,
                "cov",
                typ=str,
                in_={"full", "ungrouped_only", "none", "auto"},
            )
        )

    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """
        frame = self._frame.select(*cols) if len(cols) > 0 else self._frame
        return frame.collect().to_arrow()

    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """
        return self._frame.unique(col).collect().get_column(col).to_list()

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
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> tea_tasting.aggr.Aggregates:
        """Aggregate table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics.
        """
        return _aggregate(self, aggr_cols)


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
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        """Aggregate grouped table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics by group value.
        """
        return _aggregate(self.narwhals_frame, aggr_cols, self.by)


@overload
def _aggregate(
    narwhals_frame: NarwhalsFrame,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def _aggregate(
    narwhals_frame: NarwhalsFrame,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def _aggregate(
    narwhals_frame: NarwhalsFrame,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None = None,
) -> tea_tasting.aggr.Aggregates | dict[Hashable, tea_tasting.aggr.Aggregates]:
    frame = narwhals_frame._frame
    has_cov = narwhals_frame.cov == "full" or (
        narwhals_frame.cov == "ungrouped_only" and group_col is None
    )
    if not has_cov and len(aggr_cols.cov_cols) > 0:
        frame = _add_centered_product_cols(frame, aggr_cols, group_col)

    exprs = _aggr_exprs(aggr_cols, has_cov=has_cov)
    frame = (
        frame.select(*exprs)
        if group_col is None
        else frame.group_by(group_col).agg(*exprs)
    )
    if not has_cov and len(aggr_cols.cov_cols) > 0:
        frame = _add_fallback_cov(frame, aggr_cols.cov_cols)

    rows = frame.collect().to_arrow().to_pylist()
    return _get_aggregates(rows, aggr_cols, group_col)


def _add_centered_product_cols(
    frame: nw.LazyFrame,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
) -> nw.LazyFrame:
    keep_cols = set(aggr_cols.mean_cols) | set(aggr_cols.var_cols)
    if group_col is not None:
        keep_cols.add(group_col)

    exprs: list[nw.Expr] = [nw.col(col) for col in keep_cols]
    for left, right in aggr_cols.cov_cols:
        valid = ~nw.col(left).is_null() & ~nw.col(right).is_null()
        centered_cols: list[nw.Expr] = []
        for col in (left, right):
            valid_col = nw.when(valid).then(nw.col(col).cast(nw.Float64))
            mean = valid_col.mean()
            if group_col is not None:
                mean = mean.over(group_col)
            centered_cols.append(valid_col - mean)
        exprs.append(
            (centered_cols[0] * centered_cols[1]).alias(
                _CENTERED_PRODUCT.format(left, right),
            ),
        )
    return frame.select(*exprs)


def _aggr_exprs(
    aggr_cols: tea_tasting.aggr.AggrCols,
    *,
    has_cov: bool,
) -> list[nw.Expr]:
    exprs = [nw.len().alias(_COUNT)] if aggr_cols.has_count else []
    exprs.extend(
        nw.col(col).mean().alias(_MEAN.format(col))
        for col in aggr_cols.mean_cols
    )
    exprs.extend(
        nw.col(col).var(ddof=1).alias(_VAR.format(col))
        for col in aggr_cols.var_cols
    )
    if has_cov:
        exprs.extend(
            nw.cov(nw.col(left), nw.col(right), ddof=1).alias(_COV.format(left, right))
            for left, right in aggr_cols.cov_cols
        )
    else:
        for left, right in aggr_cols.cov_cols:
            centered_product = nw.col(_CENTERED_PRODUCT.format(left, right))
            exprs.extend((
                centered_product.sum().alias(_COV_SUM.format(left, right)),
                centered_product.count().alias(_COV_COUNT.format(left, right)),
            ))
    return exprs


def _add_fallback_cov(
    frame: nw.LazyFrame,
    cov_cols: tuple[tuple[str, str], ...],
) -> nw.LazyFrame:
    return frame.with_columns(*(
        nw.when(nw.col(_COV_COUNT.format(left, right)) > 1)
            .then(
                nw.col(_COV_SUM.format(left, right)) /
                (nw.col(_COV_COUNT.format(left, right)) - 1),
            )
            .alias(_COV.format(left, right))
        for left, right in cov_cols
    )).drop(*(
        name.format(left, right)
        for left, right in cov_cols
        for name in (_COV_SUM, _COV_COUNT)
    ))
