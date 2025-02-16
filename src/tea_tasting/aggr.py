"""Module for working with aggregated statistics: count, mean, var, cov."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, overload

import ibis.expr.operations
import ibis.expr.types
import narwhals as nw

import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence

    import narwhals.typing  # noqa: TC004


_COUNT = "_count"
_MEAN = "_mean__{}"
_VAR = "_var__{}"
_COV = "_cov__{}__{}"
_DEMEAN = "_demean__{}"


class Aggregates(tea_tasting.utils.ReprMixin):  # noqa: D101
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
        """Aggregated statistics.

        Args:
            count_: Sample size (number of observations).
            mean_: Dictionary of sample means with variable names as keys.
            var_: Dictionary of sample variances with variable names as keys.
            cov_: Dictionary of sample covariances with pairs of variable names as keys.
        """
        self.count_ = count_
        self.mean_ = mean_
        self.var_ = var_
        self.cov_ = {_sorted_tuple(*k): v for k, v in cov_.items()}

    def with_zero_div(self) -> Aggregates:
        """Return aggregates that do not raise an error on division by zero.

        Division by zero returns:

        - `inf` if numerator is greater than `0`,
        - `nan` if numerator is equal to or less than `0`.
        """
        return Aggregates(
            count_=None if self.count_ is None else tea_tasting.utils.Int(self.count_),
            mean_={k: tea_tasting.utils.numeric(v) for k, v in self.mean_.items()},
            var_={k: tea_tasting.utils.numeric(v) for k, v in self.var_.items()},
            cov_={k: tea_tasting.utils.numeric(v) for k, v in self.cov_.items()},
        )

    def count(self) -> int:
        """Sample size (number of observations).

        Returns:
            Sample size (number of observations).
        """
        if self.count_ is None:
            raise RuntimeError("Count is None.")
        return self.count_

    def mean(self, name: str | None) -> float | int:
        """Sample mean.

        Assume the variable is a constant `1` if the variable name is `None`.

        Args:
            name: Variable name.

        Returns:
            Sample mean.
        """
        if name is None:
            return 1
        return self.mean_[name]

    def var(self, name: str | None) -> float | int:
        """Sample variance.

        Assume the variable is a constant if the variable name is `None`.

        Args:
            name: Variable name.

        Returns:
            Sample variance.
        """
        if name is None:
            return 0
        return self.var_[name]

    def cov(self, left: str | None, right: str | None) -> float | int:
        """Sample covariance.

        Assume the variable is a constant if the variable name is `None`.

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

        Args:
            numer: Numerator variable name.
            denom: Denominator variable name.

        Returns:
            Sample variance of the ratio of two variables.

        References:
            - [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            - [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).
        """
        numer_mean_sq = self.mean(numer) * self.mean(numer)
        denom_mean_sq = self.mean(denom) * self.mean(denom)
        return (
            self.var(numer)
            - 2 * self.cov(numer, denom) * self.mean(numer) / self.mean(denom)
            + self.var(denom) * numer_mean_sq / denom_mean_sq
        ) / denom_mean_sq

    def ratio_cov(
        self,
        left_numer: str | None,
        left_denom: str | None,
        right_numer: str | None,
        right_denom: str | None,
    ) -> float | int:
        """Sample covariance of the ratios of variables using the Delta method.

        Args:
            left_numer: First numerator variable name.
            left_denom: First denominator variable name.
            right_numer: Second numerator variable name.
            right_denom: Second denominator variable name.

        Returns:
            Sample covariance of the ratios of variables.

        References:
            - [Delta method](https://en.wikipedia.org/wiki/Delta_method).
            - [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).
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
        """Calculate the aggregated statistics of the concatenation of two samples.

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
    data: ibis.expr.types.Table | narwhals.typing.IntoFrame,
    group_col: str,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> dict[object, Aggregates]:
    ...

@overload
def read_aggregates(
    data: ibis.expr.types.Table | narwhals.typing.IntoFrame,
    group_col: None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> Aggregates:
    ...

def read_aggregates(
    data: ibis.expr.types.Table | narwhals.typing.IntoFrame,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> dict[object, Aggregates] | Aggregates:
    """Extract aggregated statistics.

    Args:
        data: Granular data.
        group_col: Column name to group by before aggregation.
            If `None`, total aggregates are calculated.
        has_count: If `True`, calculate the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.

    Returns:
        Aggregated statistics.
    """
    mean_cols, var_cols, cov_cols = _validate_aggr_cols(mean_cols, var_cols, cov_cols)

    if isinstance(data, ibis.expr.types.Table):
        aggr_data = _read_aggr_ibis(
            data=data,
            group_col=group_col,
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )
    else:
        aggr_data = _read_aggr_narwhals(
            data=data,
            group_col=group_col,
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )

    if group_col is None:
        return _get_aggregates(
            aggr_data[0],
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )

    return {
        group_data[group_col]: _get_aggregates(
            group_data,
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )
        for group_data in aggr_data
    }


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


def _read_aggr_ibis(
    data: ibis.expr.types.Table,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> list[dict[str, int | float]]:
    covar_cols = tuple({*var_cols, *itertools.chain(*cov_cols)})
    backend = ibis.get_backend(data)
    var_op = ibis.expr.operations.Variance
    cov_op = ibis.expr.operations.Covariance
    if backend.has_operation(var_op) and backend.has_operation(cov_op):
        var_expr = {
            _VAR.format(col): data[col].cast("float").var(how="sample")  # type: ignore
            for col in var_cols
        }
        cov_expr = {
            _COV.format(left, right): data[left].cast("float").cov(  # type: ignore
                data[right].cast("float"),  # type: ignore
                how="sample",
            )
            for left, right in cov_cols
        }
    else:
        # Use demeaned values if backend doesn't have var and cov functions.
        if len(covar_cols) > 0:
            demean_expr = {
                _DEMEAN.format(col): data[col] - data[col].cast("float").mean()  # type: ignore
                for col in covar_cols
            }
            grouped_data = data.group_by(group_col) if group_col is not None else data  # type: ignore
            data = grouped_data.mutate(**demean_expr)  # type: ignore

        var_expr = {
            _VAR.format(col): (
                data[_DEMEAN.format(col)] * data[_DEMEAN.format(col)]
            ).sum() / (data.count() - 1)  # type: ignore
            for col in var_cols
        }
        cov_expr = {
            _COV.format(left, right): (
                data[_DEMEAN.format(left)] * data[_DEMEAN.format(right)]
            ).sum() / (data.count() - 1)  # type: ignore
            for left, right in cov_cols
        }

    count_expr = {_COUNT: data.count()} if has_count else {}
    mean_expr = {_MEAN.format(col): data[col].cast("float").mean() for col in mean_cols}  # type: ignore
    all_expr = count_expr | mean_expr | var_expr | cov_expr

    grouped_data = data.group_by(group_col) if group_col is not None else data  # type: ignore
    return grouped_data.aggregate(**all_expr).to_pyarrow().to_pylist()  # type: ignore


def _read_aggr_narwhals(
    data: narwhals.typing.IntoFrame,
    group_col: str | None,
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> list[dict[str, int | float]]:
    data = nw.from_native(data)
    if not isinstance(data, nw.LazyFrame):
        data = data.lazy()

    covar_cols = tuple({*var_cols, *itertools.chain(*cov_cols)})
    if len(covar_cols) > 0:
        data = (
            data.with_columns(**{
                _DEMEAN.format(col): _demean_nw_col(col, group_col)
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

    count_expr = {_COUNT: nw.len()} if has_count or len(covar_cols) > 0 else {}
    mean_expr = {_MEAN.format(col): nw.col(col).mean() for col in mean_cols}
    var_expr = {_VAR.format(col): nw.col(_VAR.format(col)).mean() for col in var_cols}
    cov_expr = {
        _COV.format(left, right): nw.col(_COV.format(left, right)).mean()
        for left, right in cov_cols
    }
    all_expr = count_expr | mean_expr | var_expr | cov_expr

    aggr_data = (
        data.select(**all_expr) if group_col is None
        else data.group_by(group_col).agg(**all_expr)
    )
    if len(covar_cols) > 0:
        aggr_data = aggr_data.with_columns(
            **{
                _VAR.format(col): nw.col(_VAR.format(col)) / (1 - 1/nw.col(_COUNT))
                for col in var_cols
            },
            **{
                _COV.format(left, right): nw.col(_COV.format(left, right)) /
                    (1 - 1/nw.col(_COUNT))
                for left, right in cov_cols
            },
        )

    return aggr_data.collect().to_arrow().to_pylist()


def _demean_nw_col(col: str, group_col: str | None) -> nw.Expr:
    if group_col is None:
        return nw.col(col) - nw.col(col).mean()
    return nw.col(col) - nw.col(col).mean().over(group_col)


def _get_aggregates(
    data: dict[str, float | int],
    *,
    has_count: bool,
    mean_cols: Sequence[str],
    var_cols: Sequence[str],
    cov_cols: Sequence[tuple[str, str]],
) -> Aggregates:
    return Aggregates(
        count_=data[_COUNT] if has_count else None,  # type: ignore
        mean_={col: data[_MEAN.format(col)] for col in mean_cols},
        var_={col: data[_VAR.format(col)] for col in var_cols},
        cov_={cols: data[_COV.format(*cols)] for cols in cov_cols},
    )
