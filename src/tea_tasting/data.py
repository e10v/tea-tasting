"""Data access helpers and backend dispatch."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, TypeGuard, overload

import narwhals.typing
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.backends


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


@overload
def read_aggregates(
    data: AggregatesByVariant,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: str | None = None,
) -> AggregatesByVariant:
    ...

@overload
def read_aggregates(
    data: Table | tea_tasting.aggr.Aggregates,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: None = None,
) -> tea_tasting.aggr.Aggregates:
    ...

@overload
def read_aggregates(
    data: Table,
    aggr_cols: tea_tasting.aggr.AggrCols,
    variant: str,
) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    ...

def read_aggregates(
    data: Table
        | tea_tasting.aggr.Aggregates
        | AggregatesByVariant,
    aggr_cols: tea_tasting.aggr.AggrCols,
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
    return table.aggregate(aggr_cols)


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
