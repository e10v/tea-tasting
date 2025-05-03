"""Experiment and experiment result."""

from __future__ import annotations

from collections import UserDict, UserList
import functools
import itertools
from typing import TYPE_CHECKING, Any, overload

import ibis.expr.types
import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Concatenate, Literal, TypeAlias, TypeVar

    import narwhals.typing  # noqa: TC004


    T = TypeVar("T")
    MapLike: TypeAlias = Callable[Concatenate[Callable[..., T], ...], Iterable[T]]
    ProgressFn: TypeAlias = Callable[Concatenate[Iterable[T], ...], Iterable[T]]
    DataGenerator: TypeAlias =  Callable[
        ..., narwhals.typing.IntoFrame | ibis.expr.types.Table]


class ExperimentResult(
    tea_tasting.utils.DictsReprMixin,
    UserDict[str, tea_tasting.metrics.MetricResult],
):
    """Experiment result for a pair of variants."""
    default_keys = (
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "rel_effect_size_ci",
        "pvalue",
    )

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the result to a sequence of dictionaries.

        Examples:
            ```pycon
            >>> import pprint
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     orders_per_user=tt.Mean("orders"),
            ...     revenue_per_user=tt.Mean("revenue"),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> pprint.pprint(result.to_dicts())
            ({'control': 0.5304003954522986,
              'effect_size': 0.04269014577177832,
              'effect_size_ci_lower': -0.010800201598205515,
              'effect_size_ci_upper': 0.09618049314176216,
              'metric': 'orders_per_user',
              'pvalue': np.float64(0.11773177998716214),
              'rel_effect_size': 0.08048664016431273,
              'rel_effect_size_ci_lower': -0.019515294044061937,
              'rel_effect_size_ci_upper': 0.1906880061278886,
              'statistic': 1.5647028839586707,
              'treatment': 0.5730905412240769},
             {'control': 5.241028175976273,
              'effect_size': 0.4890831037404775,
              'effect_size_ci_lower': -0.13261881482742033,
              'effect_size_ci_upper': 1.1107850223083753,
              'metric': 'revenue_per_user',
              'pvalue': np.float64(0.1230698855425058),
              'rel_effect_size': 0.09331815958981626,
              'rel_effect_size_ci_lower': -0.02373770894855798,
              'rel_effect_size_ci_upper': 0.22440926894909308,
              'statistic': 1.5423440700784083,
              'treatment': 5.73011127971675})

            ```
        """
        return tuple(
            {"metric": k} | (v if isinstance(v, dict) else v._asdict())
            for k, v in self.items()
        )  # type: ignore


class ExperimentResults(
    tea_tasting.utils.DictsReprMixin,
    UserDict[tuple[object, object], ExperimentResult],
):
    """Experiment results for multiple pairs of variants."""
    default_keys = (
        "variants",
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "rel_effect_size_ci",
        "pvalue",
    )

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the results to a sequence of dictionaries."""
        return tuple(
            {"variants": str(variants)} | metric_result
            for variants, experiment_result in self.items()
            for metric_result in experiment_result.to_dicts()
        )


class SimulationResults(tea_tasting.utils.DictsReprMixin, UserList[ExperimentResult]):
    """Simulation results.

    Simulations are not enumerated for better performance.
    """
    default_keys = (
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "rel_effect_size_ci",
        "pvalue",
    )
    _pagination = True

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the results to a sequence of dictionaries."""
        return tuple(itertools.chain.from_iterable(
            experiment_result.to_dicts()
            for experiment_result in self
        ))

    @tea_tasting.utils._cache_method
    def __str__(self) -> str:
        """Object string representation."""
        return str(self.to_arrow())


class ExperimentPowerResult(
    tea_tasting.utils.DictsReprMixin,
    UserDict[str, tea_tasting.metrics.MetricPowerResults[Any]],
):
    """Result of the analysis of power in a experiment."""
    default_keys = ("metric", "power", "effect_size", "rel_effect_size", "n_obs")

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the result to a sequence of dictionaries."""
        dicts = ()
        for metric, results in self.items():
            dicts = (*dicts, *({"metric": metric} | d for d in results.to_dicts()))
        return dicts


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
            ```pycon
            >>> import tea_tasting as tt

            >>> experiment = tt.Experiment(
            ...     sessions_per_user=tt.Mean("sessions"),
            ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            ...     orders_per_user=tt.Mean("orders"),
            ...     revenue_per_user=tt.Mean("revenue"),
            ... )
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> result
                        metric control treatment rel_effect_size rel_effect_size_ci pvalue
             sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
               orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
              revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            Using the first argument `metrics` which accepts metrics in a form of dictionary:

            ```pycon
            >>> experiment = tt.Experiment({
            ...     "sessions per user": tt.Mean("sessions"),
            ...     "orders per session": tt.RatioOfMeans("orders", "sessions"),
            ...     "orders per user": tt.Mean("orders"),
            ...     "revenue per user": tt.Mean("revenue"),
            ... })
            >>> data = tt.make_users_data(seed=42)
            >>> result = experiment.analyze(data)
            >>> result
                        metric control treatment rel_effect_size rel_effect_size_ci pvalue
             sessions per user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            orders per session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
               orders per user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
              revenue per user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     seed=42,
            ...     sessions_uplift=0,
            ...     orders_uplift=0,
            ...     revenue_uplift=0,
            ...     covariates=True,
            ... )
            >>> with tt.config_context(n_obs=(10_000, 20_000)):
            ...     experiment = tt.Experiment(
            ...         sessions_per_user=tt.Mean("sessions", "sessions_covariate"),
            ...         orders_per_session=tt.RatioOfMeans(
            ...             numer="orders",
            ...             denom="sessions",
            ...             numer_covariate="orders_covariate",
            ...             denom_covariate="sessions_covariate",
            ...         ),
            ...         orders_per_user=tt.Mean("orders", "orders_covariate"),
            ...         revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
            ...     )
            >>> power_result = experiment.solve_power(data)
            >>> power_result
                        metric power effect_size rel_effect_size n_obs
             sessions_per_user   80%      0.0458            2.3% 10000
             sessions_per_user   80%      0.0324            1.6% 20000
            orders_per_session   80%      0.0177            6.8% 10000
            orders_per_session   80%      0.0125            4.8% 20000
               orders_per_user   80%      0.0374            7.2% 10000
               orders_per_user   80%      0.0264            5.1% 20000
              revenue_per_user   80%       0.488            9.2% 10000
              revenue_per_user   80%       0.345            6.5% 20000

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
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object = None,
        *,
        all_variants: Literal[False] = False,
    ) -> ExperimentResult:
        ...

    @overload
    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object = None,
        *,
        all_variants: Literal[True],
    ) -> ExperimentResults:
        ...

    def analyze(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        control: object = None,
        *,
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
        """
        aggregated_data, granular_data = self._read_data(data)

        if aggregated_data is not None:
            variants = aggregated_data.keys()
        elif granular_data is not None:
            variants = granular_data.keys()
        else:
            variants = self._read_variants(data)
        variants = sorted(variants)  # type: ignore

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
        for contr, treat in variant_pairs:
            result = ExperimentResult()
            for name, metric in self.metrics.items():
                result |= {name: self._analyze_metric(
                    metric=metric,
                    data=data,
                    aggregated_data=aggregated_data,
                    granular_data=granular_data,
                    control=contr,
                    treatment=treat,
                )}

            if not all_variants:
                return result

            results |= {(contr, treat): result}

        return results


    def _analyze_metric(
        self,
        metric: tea_tasting.metrics.MetricBase[Any],
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        aggregated_data: dict[object, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[object, pa.Table] | None,
        control: object,
        treatment: object,
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


    def _read_data(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
    ) -> tuple[
        dict[object, tea_tasting.aggr.Aggregates] | None,
        dict[object, pa.Table] | None,
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

        granular_data = tea_tasting.metrics.read_granular(
            data,
            cols=tuple(gran_cols),
            variant=self.variant,
        ) if len(gran_cols) > 0 else None

        return aggregated_data, granular_data


    def _read_variants(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
    ) -> list[object]:
        if isinstance(data, ibis.expr.types.Table):
            return (
                data.select(self.variant)
                .distinct()
                .to_pyarrow()[self.variant]
                .to_pylist()
            )

        data = nw.from_native(data)
        if not isinstance(data, nw.LazyFrame):
            data = data.lazy()
        return data.unique(self.variant).collect().get_column(self.variant).to_list()


    def solve_power(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table,
        parameter: Literal[
            "power", "effect_size", "rel_effect_size", "n_obs"] = "rel_effect_size",
    ) -> ExperimentPowerResult:
        """Solve for a parameter of the power of a test.

        Args:
            data: Sample data.
            parameter: Parameter name.

        Returns:
            Power analysis result.
        """
        aggr_cols = tea_tasting.metrics.AggrCols()
        for metric in self.metrics.values():
            if isinstance(metric, tea_tasting.metrics.PowerBaseAggregated):
                aggr_cols |= metric.aggr_cols

        aggr_data = tea_tasting.aggr.read_aggregates(
            data,
            group_col=None,
            **aggr_cols._asdict(),
        ) if len(aggr_cols) > 0 else tea_tasting.aggr.Aggregates()

        result = ExperimentPowerResult()
        for name, metric in self.metrics.items():
            if isinstance(metric, tea_tasting.metrics.PowerBaseAggregated):
                result |= {name: metric.solve_power(aggr_data, parameter=parameter)}
            elif isinstance(metric, tea_tasting.metrics.PowerBase):
                result |= {name: metric.solve_power(data, parameter=parameter)}

        return result


    def simulate(
        self,
        data: narwhals.typing.IntoFrame | ibis.expr.types.Table | DataGenerator,  # type: ignore
        n_simulations: int = 10_000,
        *,
        seed: int | np.random.Generator | np.random.SeedSequence | None = None,
        ratio: float | int = 1,
        treat: Callable[[pa.Table], pa.Table] | None = None,
        map_: MapLike[Any] = map,
        progress: ProgressFn[Any] | type[Iterable[Any]] | None = None,
    ) -> SimulationResults:
        """Simulate the experiment analysis multiple times.

        Args:
            data: Experimental data or a callable that generates the data.
            n_simulations: Number of simulations.
            seed: Random seed.
            ratio: Ratio of the number of users in treatment relative to control.
            treat: Treatment function that takes a PyArrow Table as an input
                and returns an updated PyArrow Table.
            map_: Map-like function to run simulations.
            progress: tqdm-like class or function to show the progress of simulations.

        Returns:
            Simulation results.
        """
        if not callable(data):
            gran_cols: set[str] = set()
            for metric in self.metrics.values():
                if isinstance(metric, tea_tasting.metrics.MetricBaseAggregated):
                    aggr_cols = metric.aggr_cols
                    gran_cols |= (
                        set(aggr_cols.mean_cols) |
                        set(aggr_cols.var_cols) |
                        set(itertools.chain.from_iterable(aggr_cols.cov_cols))
                    )
                elif isinstance(metric, tea_tasting.metrics.MetricBaseGranular):
                    gran_cols |= set(metric.cols)
                else:
                    gran_cols = set()
                    break
            cols = tuple(gran_cols)
            data: pa.Table = tea_tasting.metrics.read_granular(data, cols)
            if self.variant in data.column_names:
                data = data.drop_columns(self.variant)

        sim = functools.partial(
            _simulate_once,
            experiment=self,
            data=data,
            ratio=ratio,
            treat=treat,
        )

        results = map_(sim, np.random.default_rng(seed).spawn(n_simulations))
        if progress is not None:
            results = progress(results)  # type: ignore
        return SimulationResults(results)


def _simulate_once(
    rng: np.random.Generator,
    experiment: Experiment,
    data: pa.Table | DataGenerator,  # type: ignore
    ratio: float | int,
    treat: Callable[[pa.Table], pa.Table] | None,
) -> ExperimentResult:
    if callable(data):
        data: pa.Table = tea_tasting.metrics.read_granular(data(seed=rng))  # type: ignore

    if experiment.variant not in data.column_names:
        data = data.append_column(
            experiment.variant,
            [rng.binomial(n=1, p=ratio / (1 + ratio), size=data.num_rows)],
        )

    if treat is not None:
        variant_array = data[experiment.variant]
        contr_data = data.filter(pc.equal(variant_array, pa.scalar(0)))  # type: ignore
        treat_data = treat(data.filter(pc.equal(variant_array, pa.scalar(1))))  # type: ignore
        if not contr_data.schema.equals(treat_data.schema):
            schema = pa.unify_schemas(
                [contr_data.schema, treat_data.schema],
                promote_options="permissive",
            )
            contr_data = contr_data.select(schema.names).cast(schema)
            treat_data = treat_data.select(schema.names).cast(schema)
        data = pa.concat_tables((contr_data, treat_data))

    return experiment.analyze(data)
