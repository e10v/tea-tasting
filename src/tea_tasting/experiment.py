"""Experiments and experiment results."""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, Any

import pandas as pd

import tea_tasting.aggr
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    import ibis.expr.types


class ExperimentResult(UserDict[str, tea_tasting.metrics.MetricResultBase]):
    """Experiment result for a pair of variants."""

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Convert the result to a sequence of dictionaries."""
        return tuple(
            {"metric": k} | (v if isinstance(v, dict) else v._asdict())
            for k, v in self.items()
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert the result to a Pandas DataFrame."""
        return pd.DataFrame.from_records(self.to_dicts())


class ExperimentResults(UserDict[tuple[Any, Any], ExperimentResult]):
    """Experiment results for all pairs of variants (control, treatment)."""

    def __getitem__(self, key: tuple[Any, Any]) -> ExperimentResult:
        """Get the result for a pair of variants (control, treatment).

        Both the control and the treatment can be None if there are two variants
        in the experiment.
        """
        if key[0] is None or key[1] is None:
            if len(self.data) != 1:
                raise ValueError("Key values must be not None.")
            return next(iter(self.data.values()))
        return super().__getitem__(key)

    def to_dicts(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> tuple[dict[str, Any], ...]:
        """Convert the result to a sequence of dictionaries for a pair of variants.

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
        return self[control, treatment].to_dicts()

    def to_pandas(
        self,
        control: Any = None,
        treatment: Any = None,
    ) -> pd.DataFrame:
        """Convert the result to a Pandas DataFrame for a pair of variants.

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
        return self[control, treatment].to_pandas()


class Experiment(tea_tasting.utils.ReprMixin):
    """Defines and analyzes the experiment."""

    def __init__(
        self,
        metrics: dict[str, tea_tasting.metrics.MetricBase[Any]],
        variant: str = "variant",
    ) -> None:
        """Define an experiment.

        Args:
            metrics: A dictionary  metrics with metric names as keys.
            variant: Variant column name.
        """
        tea_tasting.utils.check_scalar(metrics, "metrics", typ=dict)
        for name, metric in metrics.items():
            tea_tasting.utils.check_scalar(name, "metric_name", typ=str)
            tea_tasting.utils.check_scalar(
                metric, "metric", typ=tea_tasting.metrics.MetricBase)

        self.metrics = metrics
        self.variant = tea_tasting.utils.check_scalar(
            variant, "variant", typ=str)


    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any = None,
    ) -> ExperimentResults:
        """Analyze the experiment.

        Args:
            data: Experimental data.
            control: Control variant. If None, all pairs of variants are analyzed,
                with variant with the minimal ID as the control.

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

        if control is not None:
            variant_pairs = (
                (control, treatment)
                for treatment in variants
                if treatment != control
            )
        else:
            variant_pairs = (
                (control, treatment)
                for control in variants
                for treatment in variants
                if control < treatment
            )

        results = ExperimentResults()
        for control, treatment in variant_pairs:
            result = ExperimentResult()
            for name, metric in self.metrics.items():
                result |= {name: self._analyze_metric(
                    metric=metric,
                    data=data,
                    aggr_data=aggregated_data,
                    granular_data=granular_data,
                    control=control,
                    treatment=treatment,
                )}
            results |= {(control, treatment): result}

        return results


    def _analyze_metric(
        self,
        metric: tea_tasting.metrics.MetricBase[Any],
        data: pd.DataFrame | ibis.expr.types.Table,
        aggr_data: dict[Any, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[Any, pd.DataFrame] | None,
        control: Any,
        treatment: Any,
    ) -> tea_tasting.metrics.MetricResultBase:
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

        return metric.analyze(data, control, treatment, self.variant)


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
            variant=self.variant,
        ) if len(aggr_cols) > 0 else None

        granular_data = tea_tasting.metrics.read_dataframes(
            data,
            cols=tuple(gran_cols),
            variant=self.variant,
        ) if len(gran_cols) > 0 else None

        return aggregated_data, granular_data


    def _read_variants(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
    ) -> pd.Series[Any]:  # type: ignore
        if isinstance(data, pd.DataFrame):
            return data.loc[:, self.variant].drop_duplicates()

        return (
            data.select(self.variant)
            .distinct()
            .to_pandas()
            .loc[:, self.variant]
        )
