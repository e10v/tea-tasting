"""Experiment and experiment result."""

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
    """Experiment result for a pair of variants.

    Examples:
        ```python
        import tea_tasting as tt


        experiment = tt.Experiment(
            sessions_per_user=tt.Mean("sessions"),
            orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            orders_per_user=tt.Mean("orders"),
            revenue_per_user=tt.Mean("revenue"),
        )

        data = tt.make_users_data(seed=42)
        result = experiment.analyze(data)
        print(result)
        #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
        #>  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
        #> orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
        #>    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
        #>   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
        ```
    """  # noqa: E501

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Convert the result to a sequence of dictionaries.

        Examples:
            ```python
            import pprint

            import tea_tasting as tt


            experiment = tt.Experiment(
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            pprint.pprint(result.to_dicts())
            #> ({'control': 0.5304003954522986,
            #>   'effect_size': 0.04269014577177832,
            #>   'effect_size_ci_lower': -0.010800201598205564,
            #>   'effect_size_ci_upper': 0.0961804931417622,
            #>   'metric': 'orders_per_user',
            #>   'pvalue': 0.11773177998716244,
            #>   'rel_effect_size': 0.08048664016431273,
            #>   'rel_effect_size_ci_lower': -0.019515294044062048,
            #>   'rel_effect_size_ci_upper': 0.19068800612788883,
            #>   'statistic': 1.5647028839586694,
            #>   'treatment': 0.5730905412240769},
            #>  {'control': 5.2410786458606005,
            #>   'effect_size': 0.4890530110046889,
            #>   'effect_size_ci_lower': -0.13265634499246826,
            #>   'effect_size_ci_upper': 1.110762367001846,
            #>   'metric': 'revenue_per_user',
            #>   'pvalue': 0.123097417367404,
            #>   'rel_effect_size': 0.09331151925967429,
            #>   'rel_effect_size_ci_lower': -0.023744208691729107,
            #>   'rel_effect_size_ci_upper': 0.22440254776265856,
            #>   'statistic': 1.5422307220453677,
            #>   'treatment': 5.730131656865289})
            ```
        """
        return tuple(
            {"metric": k} | (v if isinstance(v, dict) else v._asdict())
            for k, v in self.items()
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert the result to a Pandas DataFrame.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result.to_pandas())
            #>                metric   control  ...    pvalue  statistic
            #> 0   sessions_per_user  1.996045  ...  0.674021  -0.420667
            #> 1  orders_per_session  0.265726  ...  0.076238   1.773406
            #> 2     orders_per_user  0.530400  ...  0.117732   1.564703
            #> 3    revenue_per_user  5.241079  ...  0.123097   1.542231
            #>
            #> [4 rows x 11 columns]
            ```
        """
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
            names: Metric attribute names. If an attribute is not defined
                for a metric it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            Pandas Dataframe with formatted values.

        Default formatting rules:
            - If a name starts with `"rel_"` consider it a percentage value.
                Round percentage values to 2 significant digits, multiply by `100`
                and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result.to_pretty(names=(
                "control",
                "treatment",
                "effect_size",
                "effect_size_ci",
            )))
            #>                metric control treatment effect_size      effect_size_ci
            #> 0   sessions_per_user    2.00      1.98     -0.0132   [-0.0750, 0.0485]
            #> 1  orders_per_session   0.266     0.289      0.0233  [-0.00246, 0.0491]
            #> 2     orders_per_user   0.530     0.573      0.0427   [-0.0108, 0.0962]
            #> 3    revenue_per_user    5.24      5.73       0.489      [-0.133, 1.11]
            ```
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
            names: Metric attribute names. If an attribute is not defined
                for a metric it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            A string with formatted values.

        Default formatting rules:
            - If a name starts with `"rel_"` consider it a percentage value.
                Round percentage values to 2 significant digits, multiply by `100`
                and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result.to_string(names=(
                "control",
                "treatment",
                "effect_size",
                "effect_size_ci",
            )))
            #>             metric control treatment effect_size     effect_size_ci
            #>  sessions_per_user    2.00      1.98     -0.0132  [-0.0750, 0.0485]
            #> orders_per_session   0.266     0.289      0.0233 [-0.00246, 0.0491]
            #>    orders_per_user   0.530     0.573      0.0427  [-0.0108, 0.0962]
            #>   revenue_per_user    5.24      5.73       0.489     [-0.133, 1.11]
            ```
        """
        return self.to_pretty(names=names, formatter=formatter).to_string(index=False)

    def to_html(
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
        """Convert the result to HTML.

        Metric result attribute values are converted to strings in a "pretty" format.

        Args:
            names: Metric attribute names. If an attribute is not defined
                for a metric it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            A table with results rendered as HTML.

        Default formatting rules:
            - If a name starts with `"rel_"` consider it a percentage value.
                Round percentage values to 2 significant digits, multiply by `100`
                and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result.to_html())
            #> <table border="1" class="dataframe">
            #>   <thead>
            #>     <tr style="text-align: right;">
            #>       <th>metric</th>
            #>       <th>control</th>
            #>       <th>treatment</th>
            #>       <th>rel_effect_size</th>
            #>       <th>rel_effect_size_ci</th>
            #>       <th>pvalue</th>
            #>     </tr>
            #>   </thead>
            #>   <tbody>
            #>     <tr>
            #>       <td>orders_per_user</td>
            #>       <td>0.530</td>
            #>       <td>0.573</td>
            #>       <td>8.0%</td>
            #>       <td>[-2.0%, 19%]</td>
            #>       <td>0.118</td>
            #>     </tr>
            #>     <tr>
            #>       <td>revenue_per_user</td>
            #>       <td>5.24</td>
            #>       <td>5.73</td>
            #>       <td>9.3%</td>
            #>       <td>[-2.4%, 22%]</td>
            #>       <td>0.123</td>
            #>     </tr>
            #>   </tbody>
            #> </table>
            ```
        """
        return self.to_pretty(names=names, formatter=formatter).to_html(index=False)

    def __str__(self) -> str:
        """Result string representation."""
        return self.to_string()

    def _repr_html_(self) -> str:
        """Result HTML representation."""
        return self.to_html()


ExperimentResults = dict[tuple[Any, Any], ExperimentResult]


class Experiment(tea_tasting.utils.ReprMixin):  # noqa: D101
    def __init__(
        self,
        metrics: dict[str, tea_tasting.metrics.MetricBase[Any]] | None = None,
        variant: str = "variant",
        **kw_metrics: tea_tasting.metrics.MetricBase[Any],
    ) -> None:
        """Experiment definition: metrics and variant column.

        Args:
            metrics: Dictionary of metrics with metric names as keys.
            variant: Variant column name.
            kw_metrics: Metrics with metric names as parameter names.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #>  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            #> orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            #>    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            #>   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
            ```

            Using the first argument `metrics` which accepts metrics if a form of dictionary:

            ```python
            experiment = tt.Experiment({
                "sessions per user": tt.Mean("sessions"),
                "orders per session": tt.RatioOfMeans("orders", "sessions"),
                "orders per user": tt.Mean("orders"),
                "revenue per user": tt.Mean("revenue"),
            })

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #>  sessions per user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            #> orders per session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            #>    orders per user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            #>   revenue per user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
            ```
        """  # noqa: E501
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
            control: Control variant. If `None`, the variant with the minimal ID
                is used as a control.
            all_variants: If `True`, analyze all pairs of variants. Otherwise,
                analyze only one pair of variants.

        Returns:
            Experiment result.

        Examples:
            ```python
            import tea_tasting as tt


            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

            data = tt.make_users_data(seed=42)
            result = experiment.analyze(data)
            print(result)
            #>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
            #>  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            #> orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            #>    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            #>   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
            ```
        """  # noqa: E501
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
