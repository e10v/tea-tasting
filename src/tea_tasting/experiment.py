"""Experiments."""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, Any, overload

import pandas as pd

import tea_tasting.aggr
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    import ibis.expr.types


def _default_formatter(data: dict[str, Any], name: str) -> str:
    if name.endswith("_ci"):
        ci_lower = _default_formatter(data, name + "_lower")
        ci_upper = _default_formatter(data, name + "_upper")
        return f"[{ci_lower}, {ci_upper}]"

    sig, pct = (2, True) if name.startswith("rel_") else (3, False)
    return tea_tasting.utils.format_num(data.get(name), sig=sig, pct=pct)


class ExperimentResult(UserDict[str, tea_tasting.metrics.MetricResult]):
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

    def to_pretty(
        self,
        names: Sequence[str] = (
            "control",
            "treatment",
            "rel_effect_size",
            "rel_effect_size_ci",
            "pvalue",
        ),
        formatter: Callable[[dict[str, Any], str], str] = _default_formatter,
    ) -> pd.DataFrame:
        """Convert the result to a Pandas Dataframe with formatted values.

        Metric result attribute values are converted to strings in a "pretty" format.

        Args:
            names: Metric attribute  attribute names. If an attribute is not defined
                for a metric it's assumed to be None.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            Pandas Dataframe with formatted values.
        """
        records: list[dict[str, Any]] = []
        for key, val in self.items():
            data = val if isinstance(val, dict) else val._asdict()
            formatted_values = {name: formatter(data, name) for name in names}
            records.append({"metric": key} | formatted_values)
        return pd.DataFrame.from_records(records)

    def to_string(
        self,
        names: Sequence[str] = (
            "control",
            "treatment",
            "rel_effect_size",
            "rel_effect_size_ci",
            "pvalue",
        ),
        formatter: Callable[[dict[str, Any], str], str] = _default_formatter,
    ) -> str:
        """Convert the result to a string.

        Metric result attribute values are converted to strings in a "pretty" format.

        Args:
            names: Metric attribute  attribute names. If an attribute is not defined
                for a metric it's assumed to be None.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            A string with formatted values.
        """
        return self.to_pretty(names=names, formatter=formatter).to_string(index=False)

    def __str__(self) -> str:
        """Result string representation."""
        return self.to_string()


ExperimentResults = dict[tuple[Any, Any], ExperimentResult]


class Experiment(tea_tasting.utils.ReprMixin):  # noqa: D101
    def __init__(
        self,
        metrics: dict[str, tea_tasting.metrics.MetricBase[Any]] | None = None,
        variant: str = "variant",
        **kw_metrics: tea_tasting.metrics.MetricBase[Any],
    ) -> None:
        """Experiment.

        Args:
            metrics: Dictionary of metrics with metric names as keys.
            variant: Variant column name.
            kw_metrics: Metrics with metric names as parameter names.
        """
        if metrics is None:
            metrics = {}
        metrics = metrics | kw_metrics

        tea_tasting.utils.check_scalar(metrics, "metrics", typ=dict)
        tea_tasting.utils.check_scalar(len(metrics), "len(metrics)", gt=0)
        for name, metric in metrics.items():
            tea_tasting.utils.check_scalar(name, "metric_name", typ=str)
            tea_tasting.utils.check_scalar(
                metric, name, typ=tea_tasting.metrics.MetricBase)

        self.metrics = metrics
        self.variant = tea_tasting.utils.check_scalar(
            variant, "variant", typ=str)


    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any = None,
        all_variants: Literal[False] = False,
    ) -> ExperimentResult:
        ...

    @overload
    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any = None,
        all_variants: Literal[True] = True,
    ) -> ExperimentResults:
        ...

    def analyze(
        self,
        data: pd.DataFrame | ibis.expr.types.Table,
        control: Any = None,
        all_variants: bool = False,
    ) -> ExperimentResult | ExperimentResults:
        """Analyze the experiment.

        Args:
            data: Experimental data.
            control: Control variant. If None, the variant with the minimal ID
                is used as a control.
            all_variants: If True, analyze all pairs of variants. Otherwise,
                analyze only one pair of variants.

        Returns:
            Experiment result.
        """
        aggregated_data, granular_data = self._read_data(data)

        if aggregated_data is not None:
            variants = aggregated_data.keys()
        elif granular_data is not None:
            variants = granular_data.keys()
        else:
            variants = self._read_variants(data)

        if control is not None:
            variant_pairs = tuple(
                (control, treatment)
                for treatment in variants
                if treatment != control
            )
        else:
            variant_pairs = tuple(
                (control, treatment)
                for control in variants
                for treatment in variants
                if control < treatment
            )

        if len(variant_pairs) != 1 and not all_variants:
            raise ValueError(
                "all_variants is False, but there are more than one pair of variants.")

        results = ExperimentResults()
        for control, treatment in variant_pairs:
            result = ExperimentResult()
            for name, metric in self.metrics.items():
                result |= {name: self._analyze_metric(
                    metric=metric,
                    data=data,
                    aggregated_data=aggregated_data,
                    granular_data=granular_data,
                    control=control,
                    treatment=treatment,
                )}

            if not all_variants:
                return result

            results |= {(control, treatment): result}

        return results


    def _analyze_metric(
        self,
        metric: tea_tasting.metrics.MetricBase[Any],
        data: pd.DataFrame | ibis.expr.types.Table,
        aggregated_data: dict[Any, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[Any, pd.DataFrame] | None,
        control: Any,
        treatment: Any,
    ) -> tea_tasting.metrics.MetricResult:
        if (
            isinstance(metric, tea_tasting.metrics.MetricBaseAggregated)
            and aggregated_data is not None
        ):
            return metric.analyze(aggregated_data, control, treatment)

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
