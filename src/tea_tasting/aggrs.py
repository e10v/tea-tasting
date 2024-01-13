"""Classes for working with aggregates: count, mean, var, cov."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import tea_tasting._utils


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from ibis.expr.types import Table
    from typing_extensions import Self


_COUNT = "_count"
_MEAN = "_mean__{}"
_MEAN_SQ = "_mean_sq__{}"
_MEAN_MUL = "_mean_mul__{}__{}"


class Aggregates:
    _count: int
    _mean: dict[str, float | int]
    _var: dict[str, float | int]
    _cov: dict[tuple[str, str], float | int]

    def __init__(
        self: Self,
        count: int,
        mean: dict[str, float | int],
        var: dict[str, float | int],
        cov: dict[tuple[str, str], float | int],
    ) -> None:
        self._count = count
        self._mean = mean
        self._var = var
        self._cov = cov


    def count(self: Self) -> int:
        return self._count


    def mean(self: Self, key: str) -> float | int:
        return self._mean[key]


    def var(self: Self, key: str) -> float | int:
        return self._var[key]


    def cov(
        self: Self,
        left: str,
        right: str,
    ) -> float | int:
        return self._cov[tea_tasting._utils.sorted_tuple(left, right)]


class ExperimentAggregates:
    _variant_aggregates: dict[Hashable, Aggregates]

    def __init__(
        self: Self,
        data: Table,
        variant_col: str,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> None:
        count_expr = {_COUNT: data.count()}

        mean_expr = {
            _MEAN.format(col): data[col].mean()
            for col in tuple({*mean_cols, *var_cols, *itertools.chain(*cov_cols)})
        }

        mean_sq_expr = {
            _MEAN_SQ.format(col): (data[col] * data[col]).mean()
            for col in tuple(set(var_cols))
        }

        uniq_cov_cols = tuple({
            tea_tasting._utils.sorted_tuple(left, right)
            for left, right in cov_cols
        })
        mean_mul_expr = {
            _MEAN_MUL.format(left, right): (data[left] * data[right]).mean()
            for left, right in uniq_cov_cols
        }

        aggr_data = data.group_by(variant_col).aggregate(
            **count_expr,
            **mean_expr,
            **mean_sq_expr,
            **mean_mul_expr,
        )

        self._variant_aggregates: dict[Hashable, Aggregates] = {}

        for variant, variant_data in aggr_data.to_pandas().groupby(variant_col):
            s = variant_data.iloc[0]
            count = s[_COUNT]
            mean = {col: s[_MEAN.format(col)] for col in mean_cols}
            coef = count / (count - 1)
            var = {
                col: (s[_MEAN_SQ.format(col)] - s[_MEAN.format(col)]**2) * coef
                for col in var_cols
            }
            cov = {
                (left, right): (
                    s[_MEAN_MUL.format(left, right)] -
                    s[_MEAN.format(left)]*s[_MEAN.format(right)]
                ) * coef
                for left, right in uniq_cov_cols
            }
            self._variant_aggregates[variant] = Aggregates(
                count=count,
                mean=mean,
                var=var,
                cov=cov,
            )


    def __getitem__(self: Self, key: str) -> Aggregates:
        return self._variant_aggregates[key]
