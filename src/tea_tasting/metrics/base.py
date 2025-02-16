"""Base classes for metrics."""

from __future__ import annotations

import abc
from collections import UserList
from typing import (
    TYPE_CHECKING,
    Generic,
    NamedTuple,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import ibis
import ibis.expr.types
import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import narwhals.typing  # noqa: TC004


# The | operator doesn't work for NamedTuple, but Union works.
MetricResult: TypeAlias = Union[NamedTuple, dict[str, object]]  # noqa: UP007
MetricPowerResult: TypeAlias = Union[NamedTuple, dict[str, object]]  # noqa: UP007

R = TypeVar("R", bound=MetricResult)
P = TypeVar("P", bound=MetricPowerResult)


class MetricPowerResults(UserList[P], tea_tasting.utils.DictsReprMixin):
    """Power analysis results."""
    default_keys = ("power", "effect_size", "rel_effect_size", "n_obs")

    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """"Convert the results to a sequence of dictionaries."""
        return tuple((v if isinstance(v, dict) else v._asdict()) for v in self)

S = TypeVar("S", bound=MetricPowerResults)  # type: ignore


class MetricBase(abc.ABC, Generic[R], tea_tasting.utils.ReprMixin):
    """Base class for metrics."""
    @abc.abstractmethod
    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object,
        treatment: object,
        variant: str,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """


class PowerBase(abc.ABC, Generic[S], tea_tasting.utils.ReprMixin):
    """Base class for the analysis of power."""
    @abc.abstractmethod
    def solve_power(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> S:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """


class AggrCols(NamedTuple):
    """Columns to be aggregated for a metric analysis.

    Attributes:
        has_count: If `True`, include the sample size.
        mean_cols: Column names for calculation of sample means.
        var_cols: Column names for calculation of sample variances.
        cov_cols: Pairs of column names for calculation of sample covariances.
    """
    has_count: bool = False
    mean_cols: Sequence[str] = ()
    var_cols: Sequence[str] = ()
    cov_cols: Sequence[tuple[str, str]] = ()

    def __or__(self, other: AggrCols) -> AggrCols:
        """Merge two aggregation column specifications.

        Args:
            other: Second objects.

        Returns:
            Merged column specifications.
        """
        return AggrCols(
            has_count=self.has_count or other.has_count,
            mean_cols=tuple({*self.mean_cols, *other.mean_cols}),
            var_cols=tuple({*self.var_cols, *other.var_cols}),
            cov_cols=tuple({
                tea_tasting.aggr._sorted_tuple(*cols)
                for cols in tuple({*self.cov_cols, *other.cov_cols})
            }),
        )

    def __len__(self) -> int:
        """Total length of all object attributes.

        If has_count is True then its value is 1, or 0 otherwise.
        """
        return (
            int(self.has_count)
            + len(self.mean_cols)
            + len(self.var_cols)
            + len(self.cov_cols)
        )


class _HasAggrCols(abc.ABC):
    @property
    @abc.abstractmethod
    def aggr_cols(self) -> AggrCols:
        """Columns to be aggregated for an analysis."""


class MetricBaseAggregated(MetricBase[R], _HasAggrCols):
    """Base class for metrics, which are analyzed using aggregated statistics."""
    @overload
    def analyze(
        self,
        data: dict[object, tea_tasting.aggr.Aggregates],
        control: object,
        treatment: object,
        variant: str | None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object,
        treatment: object,
        variant: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table | dict[
            object, tea_tasting.aggr.Aggregates],
        control: object,
        treatment: object,
        variant: str | None = None,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        aggr = aggregate_by_variants(
            data,
            aggr_cols=self.aggr_cols,
            variant=variant,
        )
        return self.analyze_aggregates(
            control=aggr[control],
            treatment=aggr[treatment],
        )

    @abc.abstractmethod
    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> R:
        """Analyze metric in an experiment using aggregated statistics.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """


class PowerBaseAggregated(PowerBase[S], _HasAggrCols):
    """Base class for the analysis of power using aggregated statistics."""
    def solve_power(
        self,
        data: (
            narwhals.typing.IntoFrame |
            ibis.expr.types.Table |
            tea_tasting.aggr.Aggregates
        ),
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> S:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """
        if not isinstance(data, tea_tasting.aggr.Aggregates):
            data = tea_tasting.aggr.read_aggregates(
                data=data,
                group_col=None,
                **self.aggr_cols._asdict(),
            )
        return self.solve_power_from_aggregates(data=data, parameter=parameter)

    @abc.abstractmethod
    def solve_power_from_aggregates(
        self,
        data: tea_tasting.aggr.Aggregates,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> S:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """


def aggregate_by_variants(
    data: (
        narwhals.typing.IntoFrame |
        ibis.expr.types.Table |
        dict[object, tea_tasting.aggr.Aggregates]
    ),
    aggr_cols: AggrCols,
    variant: str | None = None,
) ->  dict[object, tea_tasting.aggr.Aggregates]:
    """Aggregate experimental data by variants.

    Args:
        data: Experimental data.
        aggr_cols: Columns to be aggregated.
        variant: Variant column name.

    Returns:
        Experimental data as a dictionary of Aggregates.
    """
    if isinstance(data, dict) and all(
        isinstance(v, tea_tasting.aggr.Aggregates) for v in data.values()  # type: ignore
    ):
        return data

    if variant is None:
        raise ValueError("The variant parameter is required but was not provided.")

    return tea_tasting.aggr.read_aggregates(
        data=data,  # type: ignore
        group_col=variant,
        **aggr_cols._asdict(),
    )


class _HasCols(abc.ABC):
    @property
    @abc.abstractmethod
    def cols(self) -> Sequence[str]:
        """Columns to be fetched for an analysis."""


class MetricBaseGranular(MetricBase[R], _HasCols):
    """Base class for metrics, which are analyzed using granular data."""
    @overload
    def analyze(
        self,
        data: dict[object, pa.Table],
        control: object,
        treatment: object,
        variant: str | None = None,
    ) -> R:
        ...

    @overload
    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object,
        treatment: object,
        variant: str,
    ) -> R:
        ...

    def analyze(
        self,
        data: (
            narwhals.typing.IntoFrame |
            ibis.expr.types.Table |
            dict[object, pa.Table]
        ),
        control: object,
        treatment: object,
        variant: str | None = None,
    ) -> R:
        """Analyze a metric in an experiment.

        Args:
            data: Experimental data.
            control: Control variant.
            treatment: Treatment variant.
            variant: Variant column name.

        Returns:
            Analysis result.
        """
        dfs = read_granular(
            data,
            cols=self.cols,
            variant=variant,
        )
        return self.analyze_granular(
            control=dfs[control],
            treatment=dfs[treatment],
        )

    @abc.abstractmethod
    def analyze_granular(
        self,
        control: pa.Table,
        treatment: pa.Table,
    ) -> R:
        """Analyze metric in an experiment using granular data.

        Args:
            control: Control data.
            treatment: Treatment data.

        Returns:
            Analysis result.
        """


def read_granular(
    data: narwhals.typing.IntoFrame | ibis.expr.types.Table | dict[object, pa.Table],
    cols: Sequence[str],
    variant: str | None = None,
) -> dict[object, pa.Table]:
    """Read granular experimental data.

    Args:
        data: Experimental data.
        cols: Columns to read.
        variant: Variant column name.

    Returns:
        Experimental data as a dictionary of PyArrow Tables.
    """
    if isinstance(data, dict) and all(
        isinstance(v, pa.Table) for v in data.values()
    ):
        return data

    if variant is None:
        raise ValueError("The variant parameter is required but was not provided.")

    if isinstance(data, ibis.expr.types.Table):
        table = data.select(*cols, variant).to_pyarrow()
    else:
        data = nw.from_native(data)
        if not isinstance(data, nw.LazyFrame):
            data = data.lazy()
        table = data.select(*cols, variant).collect().to_arrow()

    variant_col = table[variant]
    table = table.select(cols)
    return {
        var: table.filter(pc.equal(variant_col, pa.scalar(var)))  # type: ignore
        for var in variant_col.unique().to_pylist()
    }
