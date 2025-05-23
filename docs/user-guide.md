# User guide

## Installation

```bash
uv pip install tea-tasting
```

Install Pandas or Polars to serialize analysis results as a Pandas DataFrame or a Polars DataFrame, respectively. These packages are not installed with tea-tasting by default.

## Basic usage

Begin with this simple example to understand the basic functionality:

```pycon
>>> import tea_tasting as tt

>>> data = tt.make_users_data(seed=42)
>>> experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
... )
>>> result = experiment.analyze(data)
>>> result
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```

In the following sections, each step of this process is explained in detail.

### Input data

The [`make_users_data`](api/datasets.md#tea_tasting.datasets.make_users_data) function creates synthetic data for demonstration purposes. This data mimics what you might encounter in an A/B test for an online store. Each row represents an individual user, with the following columns:

- `user`: The unique identifier for each user.
- `variant`: The specific variant (e.g., 0 or 1) assigned to each user in the A/B test.
- `sessions`: The total number of user's sessions.
- `orders`: The total number of user's orders.
- `revenue`: The total revenue generated by the user.

By default, `make_users_data` returns a PyArrow Table:

```pycon
>>> data
pyarrow.Table
user: int64
variant: int64
sessions: int64
orders: int64
revenue: double
----
user: [[0,1,2,3,4,...,3995,3996,3997,3998,3999]]
variant: [[1,0,1,1,0,...,0,0,0,0,0]]
sessions: [[2,2,2,2,1,...,2,2,3,1,5]]
orders: [[1,1,1,1,1,...,0,0,0,0,2]]
revenue: [[9.17,6.43,7.94,15.93,7.14,...,0,0,0,0,17.16]]

```

You can control return type using the `return_type` parameter. The other possible output types are Pandas DataFrame and Polars DataFrame. They require Pandas or Polars packages respectively.

tea-tasting can process data in the form of an Ibis Table or a DataFrame supported by Narwhals:

- [Ibis](https://github.com/ibis-project/ibis) is a DataFrame API to various data backends. It supports many backends including BigQuery, ClickHouse, DuckDB, PostgreSQL, Snowflake, Spark etc. You can write an SQL query, [wrap](https://ibis-project.org/how-to/extending/sql#backend.sql) it as an Ibis Table and pass it to tea-tasting.
- [Narwhals](https://github.com/narwhals-dev/narwhals) is a compatibility layer between dataframe libraries. It supports cuDF, Dask, Modin, pandas, Polars, PyArrow dataframes. You can use any of these dataframes as an input to tea-tasting.

Many statistical tests, such as the Student's t-test or the Z-test, require only aggregated data for analysis. For these tests, tea-tasting retrieves only aggregated statistics like mean and variance instead of downloading all detailed data. See more details in the [guide on data backends](data-backends.md).

tea-tasting assumes that:

- Data is grouped by randomization units, such as individual users.
- There is a column indicating the variant of the A/B test (typically labeled as A, B, etc.).
- All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the table.

### A/B test definition

The [`Experiment`](api/experiment.md#tea_tasting.experiment.Experiment) class defines parameters of an A/B test: metrics and a variant column name. There are two ways to define metrics:

- Using keyword parameters, with metric names as parameter names, and metric definitions as parameter values, as in example above.
- Using the first argument `metrics` which accepts metrics in a form of dictionary with metric names as keys and metric definitions as values.

By default, tea-tasting assumes that the A/B test variant is stored in a column named `"variant"`. You can change it using the `variant` parameter of the `Experiment` class.

Example usage:

```pycon
>>> new_experiment = tt.Experiment(
...     {
...         "sessions per user": tt.Mean("sessions"),
...         "orders per session": tt.RatioOfMeans("orders", "sessions"),
...         "orders per user": tt.Mean("orders"),
...         "revenue per user": tt.Mean("revenue"),
...     },
...     variant="variant",
... )

```

### Metrics

Metrics are instances of metric classes which define how metrics are calculated. Those calculations include calculation of effect size, confidence interval, p-value and other statistics.

Use the [`Mean`](api/metrics/mean.md#tea_tasting.metrics.mean.Mean) class to compare averages between variants of an A/B test. For example, average number of orders per user, where user is a randomization unit of an experiment. Specify the column containing the metric values using the first parameter `value`.

Use the [`RatioOfMeans`](api/metrics/mean.md#tea_tasting.metrics.mean.RatioOfMeans) class to compare ratios of averages between variants of an A/B test. For example, average number of orders per average number of sessions. Specify the columns containing the numerator and denominator values using parameters `numer` and `denom`.

Use the following parameters of `Mean` and `RatioOfMeans` to customize the analysis:

- `alternative`: Alternative hypothesis. The following options are available:
    - `"two-sided"` (default): the means are unequal.
    - `"greater"`: the mean in the treatment variant is greater than the mean in the control variant.
    - `"less"`: the mean in the treatment variant is less than the mean in the control variant.
- `confidence_level`: Confidence level of the confidence interval. Default is `0.95`.
- `equal_var`: Defines whether equal variance is assumed. If `True`, pooled variance is used for the calculation of the standard error of the difference between two means. Default is `False`.
- `use_t`: Defines whether to use the Student's t-distribution (`True`) or the Normal distribution (`False`). Default is `True`.

Example usage:

```pycon
>>> another_experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions", alternative="greater"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions", confidence_level=0.9),
...     orders_per_user=tt.Mean("orders", equal_var=True),
...     revenue_per_user=tt.Mean("revenue", use_t=False),
... )

```

Look for other supported metrics in the [Metrics](api/metrics/index.md) reference.

You can change default values of these four parameters using the [global settings](#global-settings).

### Analyzing and retrieving experiment results

After defining an experiment and metrics, you can analyze the experiment data using the [`analyze`](api/experiment.md#tea_tasting.experiment.Experiment.analyze) method of the `Experiment` class. This method takes data as an input and returns an `ExperimentResult` object with experiment result.

```pycon
>>> new_result = experiment.analyze(data)

```

By default, tea-tasting assumes that the variant with the lowest ID is a control. Change default behavior using the `control` parameter:

```pycon
>>> result_with_non_default_control = experiment.analyze(data, control=1)

```

[`ExperimentResult`](api/experiment.md#tea_tasting.experiment.ExperimentResult) is a mapping. Get a metric's analysis result using metric name as a key.

```pycon
>>> import pprint

>>> pprint.pprint(result["orders_per_user"]._asdict())
{'control': 0.5304003954522986,
 'effect_size': 0.04269014577177832,
 'effect_size_ci_lower': -0.010800201598205515,
 'effect_size_ci_upper': 0.09618049314176216,
 'pvalue': np.float64(0.11773177998716214),
 'rel_effect_size': 0.08048664016431273,
 'rel_effect_size_ci_lower': -0.019515294044061937,
 'rel_effect_size_ci_upper': 0.1906880061278886,
 'statistic': 1.5647028839586707,
 'treatment': 0.5730905412240769}

```

Fields in result depend on metrics. For `Mean` and `RatioOfMeans`, the [fields include](api/metrics/mean.md#tea_tasting.metrics.mean.MeanResult):

- `metric`: Metric name.
- `control`: Mean or ratio of means in the control variant.
- `treatment`: Mean or ratio of means in the treatment variant.
- `effect_size`: Absolute effect size. Difference between two means.
- `effect_size_ci_lower`: Lower bound of the absolute effect size confidence interval.
- `effect_size_ci_upper`: Upper bound of the absolute effect size confidence interval.
- `rel_effect_size`: Relative effect size. Difference between two means, divided by the control mean.
- `rel_effect_size_ci_lower`: Lower bound of the relative effect size confidence interval.
- `rel_effect_size_ci_upper`: Upper bound of the relative effect size confidence interval.
- `pvalue`: P-value
- `statistic`: Statistic (standardized effect size).

[`ExperimentResult`](api/experiment.md#tea_tasting.experiment.ExperimentResult) provides the following methods to serialize and view the experiment result:

- `to_dicts`: Convert the result to a sequence of dictionaries.
- `to_arrow`: Convert the result to a PyArrow Table.
- `to_pandas`: Convert the result to a Pandas DataFrame. Requires Pandas to be installed.
- `to_polars`: Convert the result to a Polars DataFrame. Requires Polars to be installed.
- `to_pretty_dicts`: Convert the result to a sequence of dictionaries with formatted values (as strings).
- `to_string`: Convert the result to a string.
- `to_html`: Convert the result to HTML.

`result` is the same as `print(result.to_string())`. `ExperimentResult` provides also the `_repr_html_` method that renders it as an HTML table in IPython and Jupyter, and the `_mime_` method that renders it as a table in marimo notebooks.

```pycon
>>> result
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```

By default, methods `to_pretty_dicts`, `to_string`, and `to_html` return a predefined list of attributes. This list can be customized:

```pycon
>>> result.with_keys((
...     "metric",
...     "control",
...     "treatment",
...     "effect_size",
...     "effect_size_ci",
... ))
            metric control treatment effect_size     effect_size_ci
 sessions_per_user    2.00      1.98     -0.0132  [-0.0750, 0.0485]
orders_per_session   0.266     0.289      0.0233 [-0.00246, 0.0491]
   orders_per_user   0.530     0.573      0.0427  [-0.0108, 0.0962]
  revenue_per_user    5.24      5.73       0.489     [-0.133, 1.11]

```

Or:

```pycon
>>> print(result.to_string(keys=(
...     "metric",
...     "control",
...     "treatment",
...     "effect_size",
...     "effect_size_ci",
... )))
            metric control treatment effect_size     effect_size_ci
 sessions_per_user    2.00      1.98     -0.0132  [-0.0750, 0.0485]
orders_per_session   0.266     0.289      0.0233 [-0.00246, 0.0491]
   orders_per_user   0.530     0.573      0.0427  [-0.0108, 0.0962]
  revenue_per_user    5.24      5.73       0.489     [-0.133, 1.11]

```

## More features

### Variance reduction with CUPED/CUPAC

tea-tasting supports variance reduction with CUPED/CUPAC, within both [`Mean`](api/metrics/mean.md#tea_tasting.metrics.mean.Mean) and [`RatioOfMeans`](api/metrics/mean.md#tea_tasting.metrics.mean.RatioOfMeans) classes.

Example usage:

```pycon
>>> data_cuped = tt.make_users_data(seed=42, covariates=True)
>>> experiment_cuped = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions", "sessions_covariate"),
...     orders_per_session=tt.RatioOfMeans(
...         numer="orders",
...         denom="sessions",
...         numer_covariate="orders_covariate",
...         denom_covariate="sessions_covariate",
...     ),
...     orders_per_user=tt.Mean("orders", "orders_covariate"),
...     revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
... )
>>> result_cuped = experiment_cuped.analyze(data_cuped)
>>> result_cuped
            metric control treatment rel_effect_size rel_effect_size_ci  pvalue
 sessions_per_user    2.00      1.98          -0.68%      [-3.2%, 1.9%]   0.603
orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229
   orders_per_user   0.523     0.581             11%        [2.9%, 20%] 0.00733
  revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00674

```

Set the `covariates` parameter of the `make_users_data` functions to `True` to add the following columns with pre-experimental data:

- `sessions_covariate`: Number of sessions before the experiment.
- `orders_covariate`: Number of orders before the experiment.
- `revenue_covariate`: Revenue before the experiment.

Define the metrics' covariates:

- In `Mean`, specify the covariate using the `covariate` parameter.
- In `RatioOfMeans`, specify the covariates for the numerator and denominator using the `numer_covariate` and `denom_covariate` parameters, respectively.

### Sample ratio mismatch check

The [`SampleRatio`](api/metrics/proportion.md#tea_tasting.metrics.proportion.SampleRatio) class in tea-tasting detects mismatches in the sample ratios of different variants of an A/B test.

Example usage:

```pycon
>>> experiment_sample_ratio = tt.Experiment(
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
...     sample_ratio=tt.SampleRatio(),
... )
>>> result_sample_ratio = experiment_sample_ratio.analyze(data)
>>> result_sample_ratio
          metric control treatment rel_effect_size rel_effect_size_ci pvalue
 orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
    sample_ratio    2023      1977               -             [-, -]  0.477

```

By default, `SampleRatio` expects equal number of observations across all variants. To specify a different ratio, use the `ratio` parameter. It accepts two types of values:

- Ratio of the number of observation in treatment relative to control, as a positive number. Example: `SampleRatio(0.5)`.
- A dictionary with variants as keys and expected ratios as values. Example: `SampleRatio({"A": 2, "B": 1})`.

The `method` parameter determines the statistical test to apply:

- `"auto"`: Apply exact binomial test if the total number of observations is less than 1000, or normal approximation otherwise.
- `"binom"`: Apply exact binomial test.
- `"norm"`: Apply normal approximation of the binomial distribution.

The [result](api/metrics/proportion.md#tea_tasting.metrics.proportion.SampleRatioResult) of the sample ratio mismatch includes the following attributes:

- `metric`: Metric name.
- `control`: Number of observations in control.
- `treatment`: Number of observations in treatment.
- `pvalue`: P-value

### Global settings

In tea-tasting, you can change defaults for the following parameters:

- `alternative`: Alternative hypothesis.
- `confidence_level`: Confidence level of the confidence interval.
- `equal_var`: If `False`, assume unequal population variances in calculation of the standard deviation and the number of degrees of freedom. Otherwise, assume equal population variance and calculate pooled standard deviation.
- `n_resamples`: The number of resamples performed to form the bootstrap distribution of a statistic.
- `use_t`: If `True`, use Student's t-distribution in p-value and confidence interval calculations. Otherwise use Normal distribution.
- And [more](api/config.md#tea_tasting.config.config_context).

Use [`get_config`](api/config.md#tea_tasting.config.get_config) with the option name as a parameter to get a global option value:

```pycon
>>> tt.get_config("equal_var")
False

```

Use [`get_config`](api/config.md#tea_tasting.config.get_config) without parameters to get a dictionary of global options:

```pycon
>>> global_config = tt.get_config()

```

Use [`set_config`](api/config.md#tea_tasting.config.set_config) to set a global option value:

```pycon
>>> tt.set_config(equal_var=True, use_t=False)
>>> experiment_with_config = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
... )
>>> tt.set_config(equal_var=False, use_t=True)
>>> orders_per_user = experiment_with_config.metrics["orders_per_user"]
>>> print(
...     f"orders_per_user.equal_var: {orders_per_user.equal_var}\n"
...     f"orders_per_user.use_t: {orders_per_user.use_t}"
... )
orders_per_user.equal_var: True
orders_per_user.use_t: False

```

Use [`config_context`](api/config.md#tea_tasting.config.config_context) to temporarily set a global option value within a context:

```pycon
>>> with tt.config_context(equal_var=True, use_t=False):
...     experiment_within_context = tt.Experiment(
...         sessions_per_user=tt.Mean("sessions"),
...         orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...         orders_per_user=tt.Mean("orders"),
...         revenue_per_user=tt.Mean("revenue"),
...     )
... 
>>> orders_per_user_context = experiment_with_config.metrics["orders_per_user"]
>>> print(
...     f"global_config.equal_var: {tt.get_config('equal_var')}\n"
...     f"global_config.use_t: {tt.get_config('use_t')}\n\n"
...     f"orders_per_user_context.equal_var: {orders_per_user_context.equal_var}\n"
...     f"orders_per_user_context.use_t: {orders_per_user_context.use_t}"
... )
global_config.equal_var: False
global_config.use_t: True
<BLANKLINE>
orders_per_user_context.equal_var: True
orders_per_user_context.use_t: False

```

### More than two variants

/// admonition | Note

This guide uses [Polars](https://github.com/pola-rs/polars) as an example data backend. Install Polars in addition to tea-tasting to reproduce the examples:

```bash
uv pip install polars
```

///

In tea-tasting, it's possible to analyze experiments with more than two variants. However, the variants will be compared in pairs through two-sample statistical tests.

Example usage:

```pycon
>>> import polars as pl

>>> data_three_variants = pl.concat((
...     tt.make_users_data(seed=42, return_type="polars"),
...     tt.make_users_data(seed=21, return_type="polars")
...         .filter(pl.col("variant").eq(1))
...         .with_columns(variant=pl.lit(2, pl.Int64)),
... ))
>>> results = experiment.analyze(data_three_variants, control=0, all_variants=True)
>>> results
variants             metric control treatment rel_effect_size rel_effect_size_ci pvalue
  (0, 1)  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
  (0, 1) orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
  (0, 1)    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  (0, 1)   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
  (0, 2)  sessions_per_user    2.00      2.02           0.98%      [-2.1%, 4.1%]  0.532
  (0, 2) orders_per_session   0.266     0.273            2.8%       [-6.6%, 13%]  0.575
  (0, 2)    orders_per_user   0.530     0.550            3.8%       [-6.0%, 15%]  0.465
  (0, 2)   revenue_per_user    5.24      5.41            3.1%       [-8.1%, 16%]  0.599

```

How variant pairs are determined:

- Specified control variant: If a specific variant is set as `control`, as in the example above, it is then compared against each of the other variants.
- Default control variant: When the `control` parameter of the `analyze` method is set to `None`, tea-tasting automatically compares each variant pair. The variant with the lowest ID in each pair is a control.

Example usage without specifying a control variant:

```pycon
>>> results_all = experiment.analyze(data_three_variants, all_variants=True)
>>> results_all
variants             metric control treatment rel_effect_size rel_effect_size_ci pvalue
  (0, 1)  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
  (0, 1) orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
  (0, 1)    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  (0, 1)   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
  (0, 2)  sessions_per_user    2.00      2.02           0.98%      [-2.1%, 4.1%]  0.532
  (0, 2) orders_per_session   0.266     0.273            2.8%       [-6.6%, 13%]  0.575
  (0, 2)    orders_per_user   0.530     0.550            3.8%       [-6.0%, 15%]  0.465
  (0, 2)   revenue_per_user    5.24      5.41            3.1%       [-8.1%, 16%]  0.599
  (1, 2)  sessions_per_user    1.98      2.02            1.7%      [-1.4%, 4.8%]  0.294
  (1, 2) orders_per_session   0.289     0.273           -5.5%       [-14%, 3.6%]  0.225
  (1, 2)    orders_per_user   0.573     0.550           -4.0%       [-13%, 5.7%]  0.407
  (1, 2)   revenue_per_user    5.73      5.41           -5.7%       [-16%, 5.8%]  0.319

```

The result of the analysis is a mapping of `ExperimentResult` objects with tuples (control, treatment) as keys. You can view the result for a selected pair of variants:

```pycon
>>> results[0, 1]
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```

By default, tea-tasting does not adjust for multiple hypothesis testing. However, it provides several methods for multiple testing correction. For more details, see the [guide on multiple hypothesis testing](multiple-testing.md).
