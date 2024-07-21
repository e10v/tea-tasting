# Custom metrics

## Intro

**tea-tasting** supports Student's t-test, Z-test, and [some other statistical tests](api/metrics/index.md) out of the box. However you might want to analyze an experiment using other statistical criteria. In this case you can define a custom metric with statistical test of your choice.

There are two types of metrics in **tea-tasting**:

- Metrics that require only aggregated statistics for the analysis.
- Metrics that require granular data for the analysis.

This guide explains how to define a custom metric of each type.

First, let's import all required modules and prepare data:

```python
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import scipy.stats
import tea_tasting as tt
import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.metrics
import tea_tasting.utils


data = tt.make_users_data(seed=42)
data["has_order"] = data.orders.gt(0).astype(int)
print(data)
#>       user  variant  sessions  orders    revenue  has_order
#> 0        0        1         2       1   9.166147          1
#> 1        1        0         2       1   6.434079          1
#> 2        2        1         2       1   7.943873          1
#> 3        3        1         2       1  15.928675          1
#> 4        4        0         1       1   7.136917          1
#> ...    ...      ...       ...     ...        ...        ...
#> 3995  3995        0         2       0   0.000000          0
#> 3996  3996        0         2       0   0.000000          0
#> 3997  3997        0         3       0   0.000000          0
#> 3998  3998        0         1       0   0.000000          0
#> 3999  3999        0         5       2  17.162459          1
#>
#> [4000 rows x 6 columns]
```

This guide uses Pandas as a data backend. But it's valid for other backends as well. See the [guide on data backends](data-backends.md) for more details.

## Metrics based on aggregated statistics

Let's define a metric that performs a proportion test, [G-test](https://en.wikipedia.org/wiki/G-test) or [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test), on a binary column (with values `0` or `1`).

First step is defining a result class. It should be a named tuple or a dictionary.

```python
class ProportionResult(NamedTuple):
    control: float
    treatment: float
    effect_size: float
    rel_effect_size: float
    pvalue: float
    statistic: float
```

Second step is defining the metric class itself. Metric based on aggregated statistics should be a subclass of [`MetricBaseAggregated`](api/metrics/base.md#tea_tasting.metrics.base.MetricBaseAggregated). `MetricBaseAggregated` is a generic class with the result class as a type variable.

Metric should have the following methods properties defined:

- Method `__init__` checks and saves metric parameters.
- Property `aggr_cols` returns columns to be aggregated for analysis for each type of statistic.
- Method `analyze_aggregates` analyzes the metric using aggregated statistics.

Let's define the metric and discuss each method in details:

```python
class Proportion(tea_tasting.metrics.MetricBaseAggregated[ProportionResult]):
    def __init__(
        self,
        column: str,
        *,
        correction: bool = True,
        method: Literal["g-test", "pearson"] = "g-test",
    ) -> None:
        self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
        self.correction = tea_tasting.utils.auto_check(correction, "correction")
        self.method = tea_tasting.utils.check_scalar(
            method, "method", typ=str, in_={"g-test", "pearson"})

    @property
    def aggr_cols(self) -> tea_tasting.metrics.AggrCols:
        return tea_tasting.metrics.AggrCols(
            has_count=True,
            mean_cols=(self.column,),
        )

    def analyze_aggregates(
        self,
        control: tea_tasting.aggr.Aggregates,
        treatment: tea_tasting.aggr.Aggregates,
    ) -> ProportionResult:
        observed = np.empty(shape=(2, 2), dtype=np.int64)
        observed[0, 0] = round(control.count() * control.mean(self.column))
        observed[1, 0] = control.count() - observed[0, 0]
        observed[0, 1] = round(treatment.count() * treatment.mean(self.column))
        observed[1, 1] = treatment.count() - observed[0, 1]
        res = scipy.stats.chi2_contingency(
            observed=observed,
            correction=self.correction,
            lambda_=int(self.method == "pearson"),
        )
        return ProportionResult(
            control=control.mean(self.column),
            treatment=treatment.mean(self.column),
            effect_size=treatment.mean(self.column) - control.mean(self.column),
            rel_effect_size=treatment.mean(self.column)/control.mean(self.column) - 1,
            pvalue=res.pvalue,
            statistic=res.statistic,
        )
```

Method `__init__` save metric parameters to be used in the analysis. You can use utility functions [`check_scalar`](api/utils.md#tea_tasting.utils.check_scalar) and [`auto_check`](api/utils.md#tea_tasting.utils.auto_check) to check parameter values.

Property `aggr_cols` returns an instance of [`AggrCols`](api/metrics/base.md#tea_tasting.metrics.base.AggrCols). Analysis of proportion requires the number of rows (`has_count=True`) and the average value for the column of interest (`mean_cols=(self.column,)`) for each variant.

Method `analyze_aggregates` accepts two parameters: `control` and `treatment` data as instances of class [`Aggregates`](api/aggr.md#tea_tasting.aggr.Aggregates). They contain values for statistics and columns specified in `aggr_cols`.

Method `analyze_aggregates` returns an instance of `ProportionResult`, defined earlier, with analysis result.

Now we can analyze the proportion of users which created at least one order during the experiment. For comparison, let's also add a metric that performs Z-test on the same column.

```python
experiment_prop = tt.Experiment(
    prop_users_with_orders=Proportion("has_order"),
    mean_users_with_orders=tt.Mean("has_order", use_t=False),
)
print(experiment_prop.analyze(data))
#>                 metric control treatment rel_effect_size rel_effect_size_ci pvalue
#> prop_users_with_orders   0.345     0.384             11%             [-, -] 0.0117
#> mean_users_with_orders   0.345     0.384             11%        [2.5%, 21%] 0.0106
```

## Metrics based on granular data

```python
class MannWhitneyUResult(NamedTuple):
    pvalue: float
    statistic: float
```

```python
class MannWhitneyU(tea_tasting.metrics.MetricBaseGranular[MannWhitneyUResult]):
    def __init__(
        self,
        column: str,
        *,
        correction: bool = True,
        alternative: Literal["two-sided", "less", "greater"] | None = None,
    ) -> None:
        self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
        self.correction = tea_tasting.utils.auto_check(correction, "correction")
        self.alternative = (
            tea_tasting.utils.auto_check(alternative, "alternative")
            if alternative is not None
            else tea_tasting.config.get_config("alternative")
        )

    @property
    def cols(self) -> tuple[str]:
        return (self.column,)

    def analyze_dataframes(
        self,
        control: pd.DataFrame,
        treatment: pd.DataFrame,
    ) -> MannWhitneyUResult:
        res = scipy.stats.mannwhitneyu(
            treatment[self.column],
            control[self.column],
            use_continuity=self.correction,
            alternative=self.alternative,
        )
        return MannWhitneyUResult(
            pvalue=res.pvalue,
            statistic=res.statistic,
        )
```

```python
experiment_mwu = tt.Experiment(
    mwu_orders=MannWhitneyU("orders"),
    mwu_revenue=MannWhitneyU("revenue"),
)
result = experiment_mwu.analyze(data)
print(result.to_string(("metric", "pvalue", "statistic")))
#>      metric pvalue statistic
#>  mwu_orders 0.0263   2069092
#> mwu_revenue 0.0300   2068063
```

## Analyzing two types of metrics together

```python
experiment = tt.Experiment(
    prop_users_with_orders=Proportion("has_order"),
    mean_users_with_orders=tt.Mean("has_order"),
    mwu_orders=MannWhitneyU("orders"),
    mwu_revenue=MannWhitneyU("revenue"),
)
print(experiment.analyze(data))
#>                 metric control treatment rel_effect_size rel_effect_size_ci pvalue
#> prop_users_with_orders   0.345     0.384             11%             [-, -] 0.0117
#> mean_users_with_orders   0.345     0.384             11%        [2.5%, 21%] 0.0106
#>             mwu_orders       -         -               -             [-, -] 0.0263
#>            mwu_revenue       -         -               -             [-, -] 0.0300
```

## Recommendations
