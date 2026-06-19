"""Data access helpers and backend dispatch."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, TypeGuard, overload

import narwhals.typing
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.backends
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types


type Table = (
    tea_tasting.backends.BaseTable |
    narwhals.typing.IntoFrame |
    narwhals.typing.Frame
)
type AggregatesByVariant = Mapping[Hashable, tea_tasting.aggr.Aggregates]
type TablesByVariant = Mapping[Hashable, pa.Table]


class AggrCols(tea_tasting.utils.ReprMixin):  # noqa: D101
    has_count: bool
    mean_cols: tuple[str, ...]
    var_cols: tuple[str, ...]
    cov_cols: tuple[tuple[str, str], ...]

    def __init__(
        self,
        has_count: bool = False,  # noqa: FBT001, FBT002
        mean_cols: Sequence[str] = (),
        var_cols: Sequence[str] = (),
        cov_cols: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Columns to be aggregated for a metric analysis.

        Args:
            has_count: If `True`, include the sample size.
            mean_cols: Column names for calculation of sample means.
            var_cols: Column names for calculation of sample variances.
            cov_cols: Pairs of column names for calculation of sample covariances.
        """
        self.has_count = tea_tasting.utils.check_scalar(
            has_count,
            "has_count",
            typ=bool,
        )
        self.mean_cols, self.var_cols, self.cov_cols = _validate_aggr_cols(
            mean_cols,
            var_cols,
            cov_cols,
        )

    def __or__(self, other: AggrCols) -> AggrCols:
        """Merge two aggregation column specifications.

        Args:
            other: Second object.

        Returns:
            Merged column specifications.
        """
        return AggrCols(
            has_count=self.has_count or other.has_count,
            mean_cols=(*self.mean_cols, *other.mean_cols),
            var_cols=(*self.var_cols, *other.var_cols),
            cov_cols=(*self.cov_cols, *other.cov_cols),
        )

    def __len__(self) -> int:
        """Total length of all object attributes.

        If has_count is True then its value is 1, or 0 otherwise.

        Returns:
            Total length of all object attributes.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )


@overload
def read_aggregates(
    data: AggregatesByVariant,
    aggr_cols: AggrCols,
    variant: str | None = None,
) -> AggregatesByVariant:
    ...

@overload
def read_aggregates(
    data: Table | tea_tasting.aggr.Aggregates,
    aggr_cols: AggrCols,
    variant: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def read_aggregates(
    data: Table,
    aggr_cols: AggrCols,
    variant: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def read_aggregates(
    data: Table
        | tea_tasting.aggr.Aggregates
        | AggregatesByVariant,
    aggr_cols: AggrCols,
    variant: str | None = None,
) -> tea_tasting.aggr.Aggregates | AggregatesByVariant:
    """Read aggregated statistics.

    Args:
        data: Experimental data or aggregated statistics.
        aggr_cols: Columns to be aggregated.
        variant: Variant column name.

    Returns:
        Aggregated statistics.
    """
    if _is_aggregates_mapping(data):
        return data
    if isinstance(data, tea_tasting.aggr.Aggregates):
        if variant is not None:
            raise ValueError("The variant parameter is not supported for Aggregates.")
        return data

    table = _table(data)  # ty: ignore[invalid-argument-type]
    if variant is not None:
        table = table.group_by(variant)
    return table.aggregate(
        has_count=aggr_cols.has_count,
        mean_cols=aggr_cols.mean_cols,
        var_cols=aggr_cols.var_cols,
        cov_cols=aggr_cols.cov_cols,
    )


@overload
def read_granular(
    data: Table,
    cols: Sequence[str] = (),
    variant: None = None,
) -> pa.Table:
    ...

@overload
def read_granular(
    data: TablesByVariant,
    cols: Sequence[str] = (),
    variant: str | None = None,
) -> TablesByVariant:
    ...

@overload
def read_granular(
    data: Table,
    cols: Sequence[str],
    variant: str,
) -> dict[Hashable, pa.Table]:
    ...

def read_granular(
    data: Table | TablesByVariant,
    cols: Sequence[str] = (),
    variant: str | None = None,
) -> pa.Table | TablesByVariant:
    """Read granular experimental data.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant: Variant column name.

    Returns:
        Experimental data as a PyArrow Table or as PyArrow Tables by variant.
    """
    if _is_tables_mapping(data):
        return data

    variant_cols = () if variant is None else (variant,)
    table = _table(data).select(*cols, *variant_cols)  # ty: ignore[invalid-argument-type]
    if variant is None:
        return table

    variant_array = table[variant]
    table = table.select(cols) if len(cols) > 0 else table.select([])
    return {
        var: table.filter(pc.equal(variant_array, pa.scalar(var)))  # ty:ignore[unresolved-attribute]
        for var in variant_array.unique().to_pylist()
    }


def read_variants(data: Table, variant: str) -> list[Hashable]:
    """Read unique variant values.

    Args:
        data: Experimental data.
        variant: Variant column name.

    Returns:
        Unique variant values.
    """
    return _table(data).select_col_unique(variant)


def _validate_aggr_cols(
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    mean_cols = tuple({*mean_cols})
    var_cols = tuple({*var_cols})
    cov_cols = tuple({
        tea_tasting.aggr._sorted_tuple(left, right)
        for left, right in cov_cols
    })
    return mean_cols, var_cols, cov_cols


def _table(data: Table) -> tea_tasting.backends.BaseTable:
    if isinstance(data, tea_tasting.backends.BaseTable):
        return data
    if _is_ibis_table(data):
        return tea_tasting.backends.IbisTable(data)
    return tea_tasting.backends.NarwhalsFrame(data)


def _is_aggregates_mapping(data: object) -> TypeGuard[AggregatesByVariant]:
    return (
        isinstance(data, Mapping)
        and all(
            isinstance(value, tea_tasting.aggr.Aggregates)
            for value in data.values()
        )
    )


def _is_tables_mapping(data: object) -> TypeGuard[TablesByVariant]:
    return (
        isinstance(data, Mapping)
        and all(isinstance(value, pa.Table) for value in data.values())
    )


def _is_ibis_table(data: object) -> TypeGuard[ibis.expr.types.Table]:
    try:
        import ibis.expr.types  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name == "ibis":
            return False
        raise

    return isinstance(data, ibis.expr.types.Table)
