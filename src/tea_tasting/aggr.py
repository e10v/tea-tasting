"""Module for working with aggregated statistics: count, mean, var, cov."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, overload

import ibis.expr.types
import pandas as pd

import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


_COUNT = "_count"
_MEAN = "_mean__{}"
_VAR = "_var__{}"
_COV = "_cov__{}__{}"
_DEMEAN = "_demean__{}"


class Aggregates(tea_tasting.utils.ReprMixin):
    """Aggregated statistics."""
    count_: int | None
    mean_: dict[str, float | int]
    var_: dict[str, float | int]
    cov_: dict[tuple[str, str], float | int]

    def __init__(
        self,
        count_: int | None = None,
        mean_: dict[str, float | int] = {},  # noqa: B006
        var_: dict[str, float | int] = {},  # noqa: B006
        cov_: dict[tuple[str, str], float | int] = {},  # noqa: B006
    ) -> None:
        """Create an object with aggregated statistics.

        Args:
            count_: Sample size.
            mean_: Variables sample means.
            var_: Variables sample variances.
            cov_: Pairs of variables sample covariances.
        """
        self.count_ = count_
        self.mean_ = mean_
        self.var_ = var_
        self.cov_ = {
            _sorted_tuple(left, right): value
            for (left, right), value in cov_.items()
        }

    def filter(
        self,
        has_count: bool,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> Aggregates:
        """Filter aggregated statistics.

        Args:
            has_count: If True, keep sample size in the resulting object.
            mean_cols: Sample means variable names.
            var_cols: Sample variances variable names.
            cov_cols: Sample covariances variable names.

        Returns:
            Filtered aggregated statistics.
        """
        mean_cols, var_cols, cov_cols = _validate_aggr_cols(
            mean_cols, var_cols, cov_cols)

        return Aggregates(
            count_=self.count() if has_count else None,
            mean_={col: self.mean(col) for col in mean_cols},
            var_={col: self.var(col) for col in var_cols},
            cov_={cols: self.cov(*cols) for cols in cov_cols},
        )

    def count(self) -> int:
        """Sample size.

        Raises:
            RuntimeError: Count is None (it wasn't defined at init).

        Returns:
            Number of observations.
        """
        if self.count_ is None:
            raise RuntimeError("Count is None.")
        return self.count_

    def mean(self, key: str | None) -> float | int:
        """Sample mean.

        Args:
            key: Variable name.

        Returns:
            Sample mean.
        """
        if key is None:
            return 1
        return self.mean_[key]

    def var(self, key: str | None) -> float | int:
        """Sample variance.

        Args:
            key: Variable name.

        Returns:
            Sample variance.
        """
        if key is None:
            return 0
        return self.var_[key]

    def cov(self, left: str | None, right: str | None) -> float | int:
        """Sample covariance.

        Args:
            left: First variable name.
            right: Second variable name.

        Returns:
            Sample covariance.
        """
        if left is None or right is None:
            return 0
        return self.cov_[_sorted_tuple(left, right)]

    def ratio_var(
        self,
        numer: str | None,
        denom: str | None,
    ) -> float | int:
        """Sample variance of the ratio of two variables using the Delta method.

        References:
            [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).

        Args:
            numer: Numerator variable name.
            denom: Denominator variable name.

        Returns:
            Sample variance of the ratio of two variables.
        """
        denom_mean_sq = self.mean(denom) * self.mean(denom)
        return (
            self.var(numer)
            - 2 * self.cov(numer, denom) * self.mean(numer) / self.mean(denom)
            + self.var(denom) * self.mean(numer) * self.mean(numer) / denom_mean_sq
        ) / denom_mean_sq

    def ratio_cov(
        self,
        left_numer: str | None,
        left_denom: str | None,
        right_numer: str | None,
        right_denom: str | None,
    ) -> float | int:
        """Sample covariance of the ratios of variables using the Delta method.

        References:
            [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).

        Args:
            left_numer: First numerator variable name.
            left_denom: First denominator variable name.
            right_numer: Second numerator variable name.
            right_denom: Second denominator variable name.

        Returns:
            Sample covariance of the ratios of variables.
        """
        left_ratio_of_means = self.mean(left_numer) / self.mean(left_denom)
        right_ratio_of_means = self.mean(right_numer) / self.mean(right_denom)
        return (
            self.cov(left_numer, right_numer)
            - self.cov(left_numer, right_denom) * right_ratio_of_means
            - self.cov(left_denom, right_numer) * left_ratio_of_means
            + self.cov(left_denom, right_denom)
                * left_ratio_of_means * right_ratio_of_means
        ) / self.mean(left_denom) / self.mean(right_denom)

    def __add__(self, other: Aggregates) -> Aggregates:
        """Calculate aggregated statistics of the concatenation of two samples.

        Samples are assumed to be independent.

        Args:
            other: Aggregated statistics of the second sample.

        Returns:
            Aggregated statistics of the concatenation of two samples.
        """
        return Aggregates(
            count_=self.count() + other.count() if self.count_ is not None else None,
            mean_={col: _add_mean(self, other, col) for col in self.mean_},
            var_={col: _add_var(self, other, col) for col in self.var_},
            cov_={cols: _add_cov(self, other, cols) for cols in self.cov_},
        )


def _add_mean(left: Aggregates, right: Aggregates, col: str) -> float | int:
    sum_ = left.count()*left.mean(col) + right.count()*right.mean(col)
    count = left.count() + right.count()
    return sum_ / count

def _add_var(left: Aggregates, right: Aggregates, col: str) -> float | int:
    left_n = left.count()
    right_n = right.count()
    total_n = left_n + right_n
    diff_of_means = left.mean(col) - right.mean(col)
    return (
        left.var(col) * (left_n - 1)
        + right.var(col) * (right_n - 1)
        + diff_of_means * diff_of_means * left_n * right_n / total_n
    ) / (total_n - 1)

def _add_cov(left: Aggregates, right: Aggregates, cols: tuple[str, str]) -> float | int:
    left_n = left.count()
    right_n = right.count()
    total_n = left_n + right_n
    diff_of_means0 = left.mean(cols[0]) - right.mean(cols[0])
    diff_of_means1 = left.mean(cols[1]) - right.mean(cols[1])
    return (
        left.cov(*cols) * (left_n - 1)
        + right.cov(*cols) * (right_n - 1)
        + diff_of_means0 * diff_of_means1 * left_n * right_n / total_n
    ) / (total_n - 1)


@overload
def read_aggregates(
    data: ibis.expr.types.Table | pd.DataFrame,
    group_col: str,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> dict[Any, Aggregates]:
    ...

@overload
def read_aggregates(
    data: ibis.expr.types.Table | pd.DataFrame,
    group_col: None,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> Aggregates:
    ...

def read_aggregates(
    data: ibis.expr.types.Table | pd.DataFrame,
    group_col: str | None,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> dict[Any, Aggregates] | Aggregates:
    """Read aggregated statistics from an Ibis Table.

    Args:
        data: Ibis Table.
        group_col: Column name to group by before aggregation.
        has_count: If True, calculate the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.

    Returns:
        Aggregated statistics from the Ibis Table.
    """
    if isinstance(data, pd.DataFrame):
        con = ibis.pandas.connect()
        data = con.create_table("data", data)

    mean_cols, var_cols, cov_cols = _validate_aggr_cols(mean_cols, var_cols, cov_cols)

    demean_cols = tuple({*var_cols, *itertools.chain(*cov_cols)})
    if len(demean_cols) > 0:
        demean_expr = {
            _DEMEAN.format(col): data[col] - data[col].mean()  # type: ignore
            for col in demean_cols
        }
        grouped_data = data.group_by(group_col) if group_col is not None else data
        data = grouped_data.mutate(**demean_expr)

    count_expr = {_COUNT: data.count()} if has_count else {}
    mean_expr = {_MEAN.format(col): data[col].mean() for col in mean_cols}  # type: ignore
    var_expr = {
        _VAR.format(col): (
            data[_DEMEAN.format(col)] * data[_DEMEAN.format(col)]
        ).sum().cast("float") / (data.count() - 1)  # type: ignore
        for col in var_cols
    }
    cov_expr = {
        _COV.format(left, right): (
            data[_DEMEAN.format(left)] * data[_DEMEAN.format(right)]
        ).sum().cast("float") / (data.count() - 1)  # type: ignore
        for left, right in cov_cols
    }

    grouped_data = data.group_by(group_col) if group_col is not None else data
    aggr_data = grouped_data.aggregate(
        **count_expr,
        **mean_expr,
        **var_expr,
        **cov_expr,
    ).to_pandas()

    if group_col is None:
        return _get_aggregates(
            aggr_data,
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )

    return {
        group: _get_aggregates(
            group_data,
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )
        for group, group_data in aggr_data.groupby(group_col)
    }


def _get_aggregates(
    data: pd.DataFrame,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> Aggregates:
    s = data.iloc[0]
    return Aggregates(
        count_=s[_COUNT] if has_count else None,
        mean_={col: s[_MEAN.format(col)] for col in mean_cols},
        var_={col: s[_VAR.format(col)] for col in var_cols},
        cov_={cols: s[_COV.format(*cols)] for cols in cov_cols},
    )


def _validate_aggr_cols(
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    mean_cols = tuple({*mean_cols})
    var_cols = tuple({*var_cols})
    cov_cols = tuple({
        _sorted_tuple(left, right)
        for left, right in cov_cols
    })
    return mean_cols, var_cols, cov_cols


def _sorted_tuple(left: str, right: str) -> tuple[str, str]:
    if right < left:
        return right, left
    return left, right
