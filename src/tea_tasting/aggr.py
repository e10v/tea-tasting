"""Classes for working with aggregates: count, mean, var, cov."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import tea_tasting._utils


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ibis.expr.types import Table
    from typing_extensions import Self


_COUNT = "_count"
_MEAN = "_mean__{}"
_MEAN_OF_SQ = "_mean_of_sq__{}"
_MEAN_OF_MUL = "_mean_of_mul__{}__{}"


class Aggregates:
    _count: int | None
    _mean: dict[str, float | int]
    _var: dict[str, float | int]
    _cov: dict[tuple[str, str], float | int]

    def __init__(
        self: Self,
        count: int | None,
        mean: dict[str, float | int],
        var: dict[str, float | int],
        cov: dict[tuple[str, str], float | int],
    ) -> None:
        self._count = count
        self._mean = mean
        self._var = var
        self._cov = cov

    def __repr__(self: Self) -> str:
        return (
            f"Aggregates(count={self._count!r}, mean={self._mean!r}, "
            f"var={self._var!r}, cov={self._cov!r})"
        )

    def count(self: Self) -> int:
        if self._count is None:
            raise RuntimeError("Count is not defined.")
        return self._count

    def mean(self: Self, key: str | None) -> float | int:
        if key is None:
            return 1
        return self._mean[key]

    def var(self: Self, key: str | None) -> float | int:
        if key is None:
            return 0
        return self._var[key]

    def cov(self: Self, left: str | None, right: str | None) -> float | int:
        if left is None or right is None:
            return 0
        return self._cov[tea_tasting._utils.sorted_tuple(left, right)]


def read_aggregates(
    data: Table,
    group_col: str,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> dict[Any, Aggregates]:
    has_count, mean_cols, var_cols, cov_cols = _validate_aggr_cols(
        has_count, mean_cols, var_cols, cov_cols)

    count_expr = {_COUNT: data.count()} if has_count else {}
    mean_expr = {_MEAN.format(col): data[col].mean() for col in mean_cols}  # type: ignore
    mean_of_sq_expr = {
        _MEAN_OF_SQ.format(col): (data[col] * data[col]).mean()  # type: ignore
        for col in var_cols
    }
    mean_of_mul_expr = {
        _MEAN_OF_MUL.format(left, right): (data[left] * data[right]).mean()  # type: ignore
        for left, right in cov_cols
    }

    aggr_data = data.group_by(group_col).aggregate(
        **count_expr,
        **mean_expr,
        **mean_of_sq_expr,
        **mean_of_mul_expr,
    )

    result: dict[Any, Aggregates] = {}

    for group, group_data in aggr_data.to_pandas().groupby(group_col):
        s = group_data.iloc[0]
        count = s[_COUNT]
        bessel_factor = count / (count - 1)
        mean = {col: s[_MEAN.format(col)] for col in mean_cols}

        var = {
            col: (s[_MEAN_OF_SQ.format(col)] - s[_MEAN.format(col)]**2) * bessel_factor
            for col in var_cols
        }

        cov = {
            (left, right): (
                s[_MEAN_OF_MUL.format(left, right)] -
                s[_MEAN.format(left)]*s[_MEAN.format(right)]
            ) * bessel_factor
            for left, right in cov_cols
        }

        result[group] = Aggregates(
            count=count,
            mean=mean,
            var=var,
            cov=cov,
        )

    return result


def _validate_aggr_cols(
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    has_count = has_count or len(var_cols) > 0 or len(cov_cols) > 0
    mean_cols = tuple({*mean_cols, *var_cols, *itertools.chain(*cov_cols)})
    var_cols = tuple(set(var_cols))
    cov_cols = tuple({
        tea_tasting._utils.sorted_tuple(left, right)
        for left, right in cov_cols
    })
    return has_count, mean_cols, var_cols, cov_cols
