"""Experiments and experiment results."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

import tea_tasting.aggr
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Any

    import ibis.expr.types


class ExperimentResult(NamedTuple):
    """Experiment result for a pair of variants."""
    result: dict[str, NamedTuple | dict[str, Any]]

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Return result as a sequence of dictionaries, one dictionary per metric."""
        return tuple(
            {"metric": k} | (v if isinstance(v, dict) else v._asdict())
            for k, v in self.result.items()
        )

    def to_pandas(self) -> pd.DataFrame:
        """Return result as a Pandas DataFrame, one row per metric."""
        return pd.DataFrame.from_records(self.to_dicts())


class ExperimentResults(NamedTuple):
    """Experiment results for all pairs of variants (control, treatment)."""
    results: dict[tuple[Any, Any], ExperimentResult]

    def keys(self) -> tuple[tuple[Any, Any], ...]:
        """Returns a tuple of pairs of variants (control, treatment)."""
        return tuple(self.results.keys())

    def get(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> ExperimentResult:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result.
        """
        if control is None or treatment is None:
            if len(self.results) != 1:
                raise ValueError(
                    f"control is {control}, treatment is {treatment},"
                    " both must be not None.",
                )
            return next(iter(self.results.values()))

        return self.results[(control, treatment)]

    def to_dicts(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result as a sequence of dictionaries, one dictionary per metric.
        """
        return self.get(control, treatment).to_dicts()

    def to_pandas(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> pd.DataFrame:
        """Return result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Raises:
            ValueError: Either control or treatment is None while
                there are more than two variants in the experiment.

        Returns:
            Experiment result as a Pandas DataFrame, one row per metric.
        """
        return self.get(control, treatment).to_pandas()


class Experiment(tea_tasting.utils.ReprMixin):
    """Defines and analyzes the experiment."""

    def __init__(
        self,
        metrics: dict[str, tea_tasting.metrics.MetricBase],
        variant_col: str = "variant",
        control: Any = None,
    ) -> None:
        """Define an experiment.

        Args:
            metrics: A dictionary  metrics with metric names as keys.
            variant_col: Variant column name.
            control: Control variant. If None, all pairs of variants are analyzed,
                with variant with the minimal ID as the control.
        """
        tea_tasting.utils.check_scalar(metrics, "metrics", typ=dict)
        for name, metric in metrics.items():
            tea_tasting.utils.check_scalar(name, "metric_name", typ=str)
            tea_tasting.utils.check_scalar(
                metric, "metric", typ=tea_tasting.metrics.MetricBase)

        self.metrics = metrics
        self.variant_col = tea_tasting.utils.check_scalar(
            variant_col, "variant_col", typ=str)
        self.control = control


    def analyze(self, data: pd.DataFrame | ibis.expr.types.Table) -> ExperimentResults:
        """Analyze the experiment.

        Args:
            data: Experimental data.

        Returns:
            Experiment results.
        """
        aggregated_data, granular_data = self._read_data(data)

        if aggregated_data is not None:
            variants = aggregated_data.keys()
        elif granular_data is not None:
            variants = granular_data.keys()
        else:
            variants = self._read_variants(data)

        if self.control is not None:
            variant_pairs = (
                (self.control, treatment)
                for treatment in variants
                if treatment != self.control
            )
        else:
            variant_pairs = (
                (control, treatment)
                for control in variants
                for treatment in variants
                if control < treatment
            )

        results: dict[tuple[Any, Any], ExperimentResult] = {}
        for control, treatment in variant_pairs:
            result: dict[str, NamedTuple | dict[str, Any]] = {}
            for name, metric in self.metrics.items():
                result |= {name: self._analyze_metric(
                    data=data,
                    aggr_data=aggregated_data,
                    granular_data=granular_data,
                    metric=metric,
                    control=control,
                    treatment=treatment,
                )}
            results |= {(control, treatment): ExperimentResult(result)}

        return ExperimentResults(results)


    def _analyze_metric(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        aggr_data: dict[Any, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[Any, pd.DataFrame] | None,
        metric: tea_tasting.metrics.MetricBase,
        control: Any,
        treatment: Any,
    ) -> NamedTuple | dict[str, Any]:
        if (
            isinstance(metric, tea_tasting.metrics.MetricBaseAggregated)
            and aggr_data is not None
        ):
            return metric.analyze(aggr_data, control, treatment)

        if (
            isinstance(metric, tea_tasting.metrics.MetricBaseGranular)
            and granular_data is not None
        ):
            return metric.analyze(granular_data, control, treatment)

        return metric.analyze(data, control, treatment, self.variant_col)


    def _read_data(self, data: pd.DataFrame | ibis.expr.types.Table) -> tuple[
        dict[Any, tea_tasting.aggr.Aggregates] | None,
        dict[Any, pd.DataFrame] | None,
    ]:
        aggr_cols = tea_tasting.metrics.AggrCols()
        gran_cols = set()
        for metric in self.metrics.values():
            if isinstance(metric, tea_tasting.metrics.MetricBaseAggregated):
                aggr_cols |= metric.aggr_cols
            if isinstance(metric, tea_tasting.metrics.MetricBaseGranular):
                gran_cols |= set(metric.cols)

        aggregated_data = tea_tasting.metrics.aggregate_by_variants(
            data,
            aggr_cols=aggr_cols,
            variant_col=self.variant_col,
        ) if len(aggr_cols) > 0 else None

        granular_data = tea_tasting.metrics.read_dataframes(
            data,
            cols=tuple(gran_cols),
            variant_col=self.variant_col,
        ) if len(gran_cols) > 0 else None

        return aggregated_data, granular_data


    def _read_variants(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
    ) -> pd.Series[Any]:  # type: ignore
        if isinstance(data, pd.DataFrame):
            return data.loc[:, self.variant_col].drop_duplicates()

        return (
            data.select(self.variant_col)
            .distinct()
            .to_pandas()
            .loc[:, self.variant_col]
        )
