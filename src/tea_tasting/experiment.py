"""Experiment and experiment results."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Hashable
import functools
import itertools
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.aggr
import tea_tasting.data
import tea_tasting.metrics
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Concatenate, Literal, Protocol


    type MapLike[T] = Callable[Concatenate[Callable[..., T], ...], Iterable[T]]
    type DataGenerator[T] =  Callable[
        ...,
        tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
    ]

    class _ProgressLike(Protocol):
        def update(self, __n: int, /) -> object:
            ...

    class _ProgressContextLike(Protocol):
        def __enter__(self) -> _ProgressLike:
            ...

        def __exit__(self, *args: object) -> object:
            ...

    class TqdmLike(Protocol):
        """Progress factory compatible with `tqdm.tqdm`."""

        def __call__(
            self,
            *args: object,
            total: int,
            **kwargs: object,
        ) -> _ProgressContextLike:
            """Create a progress context manager."""
            ...



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
    default_text_keys = ("metric",)

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
            >>> data = tt.make_users_data(rng=42)
            >>> result = experiment.analyze(data)
            >>> pprint.pprint(result.to_dicts())
            ({'control': 0.5304003954522986,
              'effect_size': 0.04269014577177832,
              'effect_size_ci_lower': -0.010800201598205522,
              'effect_size_ci_upper': 0.09618049314176216,
              'metric': 'orders_per_user',
              'pvalue': 0.11773177998716207,
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
              'pvalue': 0.12306988554250592,
              'rel_effect_size': 0.09331815958981626,
              'rel_effect_size_ci_lower': -0.02373770894855798,
              'rel_effect_size_ci_upper': 0.22440926894909308,
              'statistic': 1.5423440700784083,
              'treatment': 5.73011127971675})

            ```
        """
        return tuple(
            {"metric": k} | tea_tasting.metrics.base._result_to_dict(v)
            for k, v in self.items()
        )


class ExperimentResults(
    tea_tasting.utils.DictsReprMixin,
    UserDict[tuple[Hashable, Hashable], ExperimentResult],
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
    default_text_keys = ("variants", "metric")

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
    default_text_keys = ("metric",)
    default_max_rows: int = 10

    @tea_tasting.utils._cache_method
    def to_dicts(self) -> tuple[dict[str, object], ...]:
        """Convert the results to a sequence of dictionaries."""
        return tuple(itertools.chain.from_iterable(
            experiment_result.to_dicts()
            for experiment_result in self
        ))


class ExperimentPowerResult(
    tea_tasting.utils.DictsReprMixin,
    UserDict[str, tea_tasting.metrics.MetricPowerResults[Any]],
):
    """Result of power analysis in an experiment."""
    default_keys = ("metric", "power", "effect_size", "rel_effect_size", "n_obs")
    default_text_keys = ("metric",)

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
            >>> data = tt.make_users_data(rng=42)
            >>> result = experiment.analyze(data)
            >>> result
            metric             control treatment rel_effect_size rel_effect_size_ci pvalue
            sessions_per_user     2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            orders_per_user      0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            revenue_per_user      5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            Using the first argument `metrics`, which accepts metrics in the form of a dictionary:

            ```pycon
            >>> experiment = tt.Experiment({
            ...     "sessions per user": tt.Mean("sessions"),
            ...     "orders per session": tt.RatioOfMeans("orders", "sessions"),
            ...     "orders per user": tt.Mean("orders"),
            ...     "revenue per user": tt.Mean("revenue"),
            ... })
            >>> data = tt.make_users_data(rng=42)
            >>> result = experiment.analyze(data)
            >>> result
            metric             control treatment rel_effect_size rel_effect_size_ci pvalue
            sessions per user     2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
            orders per session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
            orders per user      0.530     0.573            8.0%       [-2.0%, 19%]  0.118
            revenue per user      5.24      5.73            9.3%       [-2.4%, 22%]  0.123

            ```

            Power analysis:

            ```pycon
            >>> data = tt.make_users_data(
            ...     rng=42,
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
            metric             power effect_size rel_effect_size n_obs
            sessions_per_user    80%      0.0458            2.3% 10000
            sessions_per_user    80%      0.0324            1.6% 20000
            orders_per_session   80%      0.0177            6.8% 10000
            orders_per_session   80%      0.0125            4.8% 20000
            orders_per_user      80%      0.0374            7.2% 10000
            orders_per_user      80%      0.0264            5.1% 20000
            revenue_per_user     80%       0.488            9.2% 10000
            revenue_per_user     80%       0.345            6.5% 20000

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
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
        control: Hashable | None = None,
        *,
        all_variants: Literal[False] = False,
    ) -> ExperimentResult:
        ...

    @overload
    def analyze(
        self,
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
        control: Hashable | None = None,
        *,
        all_variants: Literal[True],
    ) -> ExperimentResults:
        ...

    def analyze(
        self,
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
        control: Hashable | None = None,
        *,
        all_variants: bool = False,
    ) -> ExperimentResult | ExperimentResults:
        """Analyze the experiment.

        Args:
            data: Experimental data or aggregated data by variants.
            control: Control variant. If `None`, the variant with the minimal ID
                is used as a control.
            all_variants: If `True`, analyze all pairs of variants. Otherwise,
                analyze only one pair of variants.

        Returns:
            Experiment result.
        """
        tea_tasting.utils.check_scalar(all_variants, "all_variants", typ=bool)
        aggregated_data, granular_data = self._read_data(data)
        variants = self._read_variants(data, aggregated_data, granular_data)
        variant_pairs = self._get_variant_pairs(variants, control)

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


    def _read_data(
        self,
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
    ) -> tuple[
        dict[Hashable, tea_tasting.aggr.Aggregates] | None,
        dict[Hashable, pa.Table] | None,
    ]:
        if isinstance(data, dict):
            for name, metric in self.metrics.items():
                if not isinstance(metric, tea_tasting.metrics.MetricBaseAggregated):
                    raise TypeError(
                        "Aggregated data was provided, but metric "
                        f"{name!r} is not based on aggregated statistics.",
                    )
            return data, None  # ty:ignore[invalid-return-type]

        aggr_cols = tea_tasting.data.AggrCols()
        gran_cols = set()
        for metric in self.metrics.values():
            if isinstance(metric, tea_tasting.metrics.MetricBaseAggregated):
                aggr_cols |= metric.aggr_cols
            if isinstance(metric, tea_tasting.metrics.MetricBaseGranular):
                gran_cols |= set(metric.cols)

        aggregated_data = tea_tasting.data.read_aggregates(
            data,
            aggr_cols=aggr_cols,
            variant=self.variant,
        ) if len(aggr_cols) > 0 else None

        granular_data = tea_tasting.data.read_granular(
            data,
            cols=tuple(gran_cols),
            variant=self.variant,
        ) if len(gran_cols) > 0 else None

        return aggregated_data, granular_data


    def _read_variants(
        self,
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
        aggregated_data: dict[Hashable, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[Hashable, pa.Table] | None,
    ) -> list[Hashable]:
        if aggregated_data is not None:
            variants = aggregated_data.keys()
        elif granular_data is not None:
            variants = granular_data.keys()
        else:
            variants = tea_tasting.data.read_variants(data, self.variant)  # ty:ignore[invalid-argument-type]
        return sorted(variants)


    def _get_variant_pairs(
        self,
        variants: list[Hashable],
        control: Hashable | None,
    ) -> tuple[tuple[Hashable, Hashable], ...]:
        if control is not None:
            return tuple(
                (control, treatment)
                for treatment in variants
                if treatment != control
            )

        return tuple(
            (control, treatment)
            for control in variants
            for treatment in variants
            if control < treatment  # ty:ignore[unsupported-operator]
        )


    def _analyze_metric(
        self,
        metric: tea_tasting.metrics.MetricBase[Any],
        data: tea_tasting.data.Table | dict[Hashable, tea_tasting.aggr.Aggregates],
        aggregated_data: dict[Hashable, tea_tasting.aggr.Aggregates] | None,
        granular_data: dict[Hashable, pa.Table] | None,
        control: Hashable,
        treatment: Hashable,
    ) -> tea_tasting.metrics.MetricResult:
        if (
            isinstance(metric, tea_tasting.metrics.MetricBaseAggregated)
            and aggregated_data is not None
        ):
            return metric.analyze(aggregated_data, control, treatment)  # ty:ignore[invalid-return-type]

        if (
            isinstance(metric, tea_tasting.metrics.MetricBaseGranular)
            and granular_data is not None
        ):
            return metric.analyze(granular_data, control, treatment)  # ty:ignore[invalid-return-type]

        return metric.analyze(data, control, treatment, self.variant)  # ty:ignore[invalid-argument-type]


    def solve_power(
        self,
        data: tea_tasting.data.Table,
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
        tea_tasting.utils.check_scalar(
            parameter,
            "parameter",
            in_={"power", "effect_size", "rel_effect_size", "n_obs"},
        )
        aggr_cols = tea_tasting.data.AggrCols()
        for metric in self.metrics.values():
            if isinstance(metric, tea_tasting.metrics.PowerBaseAggregated):
                aggr_cols |= metric.aggr_cols

        aggr_data = tea_tasting.data.read_aggregates(
            data,
            aggr_cols=aggr_cols,
        ) if len(aggr_cols) > 0 else tea_tasting.aggr.Aggregates()

        result = ExperimentPowerResult()
        for name, metric in self.metrics.items():
            if isinstance(metric, tea_tasting.metrics.PowerBaseAggregated):
                result |= {name: metric.solve_power(aggr_data, parameter=parameter)}  # ty:ignore[unsupported-operator]
            elif isinstance(metric, tea_tasting.metrics.PowerBase):
                result |= {name: metric.solve_power(data, parameter=parameter)}  # ty:ignore[unsupported-operator, invalid-argument-type]

        return result


    def simulate(
        self,
        data: tea_tasting.data.Table | DataGenerator,
        n_simulations: int = 10_000,
        *,
        rng: int | np.random.Generator | np.random.SeedSequence | None = None,
        ratio: float | int = 1,
        treat: Callable[[pa.Table], pa.Table] | None = None,
        map_: MapLike = map,
        batch_size: int = 1,
        progress: TqdmLike | None = None,
    ) -> SimulationResults:
        """Simulate the experiment analysis multiple times.

        Args:
            data: Experimental data or a callable that generates the data.
            n_simulations: Number of simulations.
            rng: Pseudorandom number generator or seed.
            ratio: Ratio of the number of users in treatment relative to control.
            treat: Treatment function that takes a PyArrow Table as an input
                and returns an updated PyArrow Table.
            map_: Map-like function to run simulations.
            batch_size: Number of simulations to run in each batch.
            progress: Progress class or function, such as `tqdm.tqdm` or
                `marimo.status.progress_bar`, that returns a context manager whose
                `__enter__` method returns an object with an `update(n)` method.

        Returns:
            Simulation results.
        """
        tea_tasting.utils.check_scalar(n_simulations, "n_simulations", typ=int, gt=0)
        tea_tasting.utils.check_scalar(batch_size, "batch_size", typ=int, gt=0)
        tea_tasting.utils.auto_check(ratio, "ratio")
        rng = tea_tasting.utils.auto_check(rng, "rng")

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
            data: pa.Table = tea_tasting.data.read_granular(data, cols)
            if self.variant in data.column_names:
                data = data.drop_columns(self.variant)

        sim_batch = functools.partial(
            _simulate_batch,
            experiment=self,
            data=data,
            ratio=ratio,
            treat=treat,
        )

        rngs = np.random.default_rng(rng).spawn(n_simulations)
        batches = itertools.batched(rngs, batch_size)
        batch_results = map_(sim_batch, batches)
        if progress is not None:
            results = SimulationResults()
            with progress(total=n_simulations) as progress_bar:
                for batch_result in batch_results:
                    results.extend(batch_result)
                    progress_bar.update(len(batch_result))
            return results
        return SimulationResults(itertools.chain.from_iterable(batch_results))


def _simulate_batch(
    rngs: Iterable[np.random.Generator],
    experiment: Experiment,
    data: pa.Table | DataGenerator,
    ratio: float | int,
    treat: Callable[[pa.Table], pa.Table] | None,
) -> tuple[ExperimentResult, ...]:
    return tuple(
        _simulate_once(
            rng=rng,
            experiment=experiment,
            data=data,
            ratio=ratio,
            treat=treat,
        )
        for rng in rngs
    )


def _simulate_once(
    rng: np.random.Generator,
    experiment: Experiment,
    data: pa.Table | DataGenerator,
    ratio: float | int,
    treat: Callable[[pa.Table], pa.Table] | None,
) -> ExperimentResult:
    raw_data: (
        tea_tasting.data.Table |
        dict[Hashable, tea_tasting.aggr.Aggregates] |
        pa.Table
    ) = data if isinstance(data, pa.Table) else data(rng=rng)

    if isinstance(raw_data, dict):
        if ratio != 1:
            raise ValueError(
                "The ratio parameter is not supported when callable data "
                "generates aggregated statistics.",
            )
        if treat is not None:
            raise ValueError(
                "The treat parameter is not supported when callable data "
                "generates aggregated statistics.",
            )
        return experiment.analyze(raw_data)

    table: pa.Table = (
        raw_data
        if isinstance(raw_data, pa.Table)
        else tea_tasting.data.read_granular(raw_data)
    )

    if experiment.variant not in table.column_names:
        table = table.append_column(
            experiment.variant,
            [rng.binomial(n=1, p=ratio / (1 + ratio), size=table.num_rows)],
        )

    if treat is not None:
        variant_array = table[experiment.variant]
        contr_data = table.filter(pc.equal(variant_array, pa.scalar(0)))  # ty:ignore[unresolved-attribute]
        treat_data = treat(table.filter(pc.equal(variant_array, pa.scalar(1))))  # ty:ignore[unresolved-attribute]
        if not contr_data.schema.equals(treat_data.schema):
            schema = pa.unify_schemas(
                [contr_data.schema, treat_data.schema],
                promote_options="permissive",
            )
            contr_data = contr_data.select(schema.names).cast(schema)
            treat_data = treat_data.select(schema.names).cast(schema)
        table = pa.concat_tables((contr_data, treat_data))

    return experiment.analyze(table)
