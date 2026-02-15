# Custom metrics

## Intro

tea-tasting supports Student's t-test, Z-test, and [some other statistical tests](api/metrics/index.md) out of the box. However, you might want to analyze an experiment using other statistical criteria. In this case, you can define a custom metric with a statistical test of your choice.

In tea-tasting, there are two types of metrics:

- Metrics that require only aggregated statistics for the analysis.
- Metrics that require granular data for the analysis.

This guide explains how to define a custom metric for each type.

First, let's import all the required modules and prepare the data:

```pycon
>>> from typing import Literal, NamedTuple
>>> import numpy as np
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> import scipy.stats
>>> import tea_tasting as tt
>>> import tea_tasting.aggr
>>> import tea_tasting.config
>>> import tea_tasting.metrics
>>> import tea_tasting.utils

>>> data = tt.make_users_data(rng=42)
>>> data = data.append_column(
...     "has_order",
...     pc.greater(data["orders"], 0).cast(pa.int64()),
... )
>>> data
pyarrow.Table
user: int64
variant: int64
sessions: int64
orders: int64
revenue: double
has_order: int64
----
user: [[0,1,2,3,4,...,3995,3996,3997,3998,3999]]
variant: [[1,0,1,1,0,...,0,0,0,0,0]]
sessions: [[2,2,2,2,1,...,2,2,3,1,5]]
orders: [[1,1,1,1,1,...,0,0,0,0,2]]
revenue: [[9.17,6.43,7.94,15.93,7.14,...,0,0,0,0,17.16]]
has_order: [[1,1,1,1,1,...,0,0,0,0,1]]

```

This guide uses PyArrow as the data backend, but it's valid for other backends as well. See the [guide on data backends](data-backends.md) for more details.

## Metrics based on aggregated statistics

Let's define a metric that performs a proportion test, [G-test](https://en.wikipedia.org/wiki/G-test) or [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test), on a binary column (with values `0` or `1`).

The first step is defining a result class. It should be a named tuple or a dictionary.

```pycon
>>> class ProportionResult(NamedTuple):
...     control: float
...     treatment: float
...     effect_size: float
...     rel_effect_size: float
...     pvalue: float
...     statistic: float
... 

```

The second step is defining the metric class itself. A metric based on aggregated statistics should be a subclass of [`MetricBaseAggregated`](api/metrics/base.md#tea_tasting.metrics.base.MetricBaseAggregated). `MetricBaseAggregated` is a generic class with the result class as a type variable.

The metric should have the following methods and properties defined:

- Method `__init__` checks and saves metric parameters.
- Property `aggr_cols` returns columns to be aggregated for analysis for each type of statistic.
- Method `analyze_aggregates` analyzes the metric using aggregated statistics.

Let's define the metric and discuss each method in detail:

```pycon
>>> class Proportion(tea_tasting.metrics.MetricBaseAggregated[ProportionResult]):
...     def __init__(
...         self,
...         column: str,
...         *,
...         correction: bool = True,
...         method: Literal["g-test", "pearson"] = "g-test",
...     ) -> None:
...         self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
...         self.correction = tea_tasting.utils.auto_check(correction, "correction")
...         self.method = tea_tasting.utils.check_scalar(
...             method, "method", typ=str, in_={"g-test", "pearson"})
...     @property
...     def aggr_cols(self) -> tea_tasting.metrics.AggrCols:
...         return tea_tasting.metrics.AggrCols(
...             has_count=True,
...             mean_cols=(self.column,),
...         )
...     def analyze_aggregates(
...         self,
...         control: tea_tasting.aggr.Aggregates,
...         treatment: tea_tasting.aggr.Aggregates,
...     ) -> ProportionResult:
...         observed = np.empty(shape=(2, 2), dtype=np.int64)
...         observed[0, 0] = round(control.count() * control.mean(self.column))
...         observed[1, 0] = control.count() - observed[0, 0]
...         observed[0, 1] = round(treatment.count() * treatment.mean(self.column))
...         observed[1, 1] = treatment.count() - observed[0, 1]
...         res = scipy.stats.chi2_contingency(
...             observed=observed,
...             correction=self.correction,
...             lambda_=int(self.method == "pearson"),
...         )
...         return ProportionResult(
...             control=control.mean(self.column),
...             treatment=treatment.mean(self.column),
...             effect_size=treatment.mean(self.column) - control.mean(self.column),
...             rel_effect_size=treatment.mean(self.column)/control.mean(self.column) - 1,
...             pvalue=res.pvalue,
...             statistic=res.statistic,
...         )
... 

```

Method `__init__` saves metric parameters to be used in the analysis. You can use utility functions [`check_scalar`](api/utils.md#tea_tasting.utils.check_scalar) and [`auto_check`](api/utils.md#tea_tasting.utils.auto_check) to check parameter values.

Property `aggr_cols` returns an instance of [`AggrCols`](api/metrics/base.md#tea_tasting.metrics.base.AggrCols). Analysis of proportion requires the number of rows (`has_count=True`) and the average value for the column of interest (`mean_cols=(self.column,)`) for each variant.

Method `analyze_aggregates` accepts two parameters: `control` and `treatment` data as instances of class [`Aggregates`](api/aggr.md#tea_tasting.aggr.Aggregates). They contain values for statistics and columns specified in `aggr_cols`.

Method `analyze_aggregates` returns an instance of `ProportionResult`, defined earlier, with the analysis result.

Now we can analyze the proportion of users who created at least one order during the experiment. For comparison, let's also add a metric that performs a Z-test on the same column.

```pycon
>>> experiment_prop = tt.Experiment(
...     prop_users_with_orders=Proportion("has_order"),
...     mean_users_with_orders=tt.Mean("has_order", use_t=False),
... )
>>> experiment_prop.analyze(data)
                metric control treatment rel_effect_size rel_effect_size_ci pvalue
prop_users_with_orders   0.345     0.384             11%             [-, -] 0.0117
mean_users_with_orders   0.345     0.384             11%        [2.5%, 21%] 0.0106

```

## Metrics based on granular data

Now let's define a metric that performs the Mann-Whitney U test. While it's possible to use the aggregated sum of ranks for the test, this example uses granular data for analysis.

The result class:

```pycon
>>> class MannWhitneyUResult(NamedTuple):
...     pvalue: float
...     statistic: float
... 

```

A metric that analyzes granular data should be a subclass of [`MetricBaseGranular`](api/metrics/base.md#tea_tasting.metrics.base.MetricBaseGranular). `MetricBaseGranular` is a generic class with the result class as a type variable.

The metric should have the following methods and properties defined:

- Method `__init__` checks and saves metric parameters.
- Property `cols` returns columns to be fetched for an analysis.
- Method `analyze_granular` analyzes the metric using granular data.

```pycon
>>> class MannWhitneyU(tea_tasting.metrics.MetricBaseGranular[MannWhitneyUResult]):
...     def __init__(
...         self,
...         column: str,
...         *,
...         correction: bool = True,
...         alternative: Literal["two-sided", "less", "greater"] | None = None,
...     ) -> None:
...         self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
...         self.correction = tea_tasting.utils.auto_check(correction, "correction")
...         self.alternative = (
...             tea_tasting.utils.auto_check(alternative, "alternative")
...             if alternative is not None
...             else tea_tasting.config.get_config("alternative")
...         )
...     @property
...     def cols(self) -> tuple[str]:
...         return (self.column,)
...     def analyze_granular(
...         self,
...         control: pa.Table,
...         treatment: pa.Table,
...     ) -> MannWhitneyUResult:
...         res = scipy.stats.mannwhitneyu(
...             treatment[self.column].combine_chunks().to_numpy(zero_copy_only=False),
...             control[self.column].combine_chunks().to_numpy(zero_copy_only=False),
...             use_continuity=self.correction,
...             alternative=self.alternative,
...         )
...         return MannWhitneyUResult(
...             pvalue=res.pvalue,
...             statistic=res.statistic,
...         )
... 

```

Property `cols` should return a sequence of strings.

Method `analyze_granular` accepts two parameters: control and treatment data as PyArrow Tables. Even with a [data backend](data-backends.md) other than PyArrow, tea-tasting retrieves the data and converts it into a PyArrow Table.

Method `analyze_granular` returns an instance of `MannWhitneyUResult`, defined earlier, with the analysis result.

Now we can perform the Mann-Whitney U test:

```pycon
>>> experiment_mwu = tt.Experiment(
...     mwu_orders=MannWhitneyU("orders"),
...     mwu_revenue=MannWhitneyU("revenue"),
... )
>>> result_mwu = experiment_mwu.analyze(data)
>>> result_mwu.with_keys(("metric", "pvalue", "statistic"))
     metric pvalue statistic
 mwu_orders 0.0263   2069092
mwu_revenue 0.0300   2068060

```

## Analyzing two types of metrics together

It's also possible to analyze both types of metrics in one experiment:

```pycon
>>> experiment = tt.Experiment(
...     prop_users_with_orders=Proportion("has_order"),
...     mean_users_with_orders=tt.Mean("has_order"),
...     mwu_orders=MannWhitneyU("orders"),
...     mwu_revenue=MannWhitneyU("revenue"),
... )
>>> experiment.analyze(data)
                metric control treatment rel_effect_size rel_effect_size_ci pvalue
prop_users_with_orders   0.345     0.384             11%             [-, -] 0.0117
mean_users_with_orders   0.345     0.384             11%        [2.5%, 21%] 0.0106
            mwu_orders       -         -               -             [-, -] 0.0263
           mwu_revenue       -         -               -             [-, -] 0.0300

```

In this case, tea-tasting performs two queries on the experimental data:

- With aggregated statistics required for analysis of metrics of type `MetricBaseAggregated`.
- With detailed data with columns required for analysis of metrics of type `MetricBaseGranular`.

## Recommendations

Follow these recommendations when defining custom metrics:

- Use parameter and attribute names consistent with the ones that are already defined in tea-tasting. For example, use `pvalue` instead of `p_value` or `correction` instead of `use_continuity`.
- End confidence interval boundary names with `"_ci_lower"` and `"_ci_upper"`.
- During initialization, save parameter values in metric attributes using the same names. For example, use `self.correction = correction` instead of `self.use_continuity = correction`.
- Use global settings as default values for standard parameters, such as `alternative` or `confidence_level`. See the [reference](api/config.md#tea_tasting.config.config_context) for the full list of standard parameters. You can also define and use your own global parameters.
