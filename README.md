# tea-tasting: statistical analysis of A/B tests

[![CI](https://github.com/e10v/tea-tasting/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/e10v/tea-tasting/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/github/e10v/tea-tasting/coverage.svg?branch=main)](https://codecov.io/gh/e10v/tea-tasting)
[![License](https://img.shields.io/github/license/e10v/tea-tasting)](https://github.com/e10v/tea-tasting/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)
[![Package Status](https://img.shields.io/pypi/status/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)

**tea-tasting** is a Python package for statistical analysis of A/B tests that features:

- Student's t-test and Z-test out of the box.
- Extensible API: Define and use statistical tests of your choice.
- [Delta method](https://alexdeng.github.io/public/files/kdd2018-dm.pdf) for ratio metrics.
- Variance reduction with [CUPED](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)/[CUPAC](https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/) (also in combination with delta method for ratio metrics).
- Confidence interval for both absolute and percent change.

**tea-tasting** calculates statistics within data backends such as BigQuery, ClickHouse, PostgreSQL, Snowflake, Spark, and other of 20+ backends supported by [Ibis](https://ibis-project.org/). This approach eliminates the need to import granular data into a Python environment, though Pandas DataFrames are also supported.

**tea-tasting** is still in alpha, but already includes all the features listed above. The following features are coming soon:

- Sample ratio mismatch check.
- More statistical tests:
  - Asymptotic and exact tests for frequency data.
  - Bootstrap.
  - Quantile test (using Bootstrap).
  - Mann–Whitney U test.
- Power analysis.
- A/A tests and simulations.
- Pretty output for experiment results (round etc.).
- Documentation on how to define metrics with custom statistical tests.
- Documentation with MkDocs and Material for MkDocs.
- More examples.

## Installation

```bash
pip install tea-tasting
```

## Basic usage

Begin with this simple example to understand the basic functionality:

```python
import tea_tasting as tt


users_data = tt.make_users_data(seed=42)

experiment = tt.Experiment(
    sessions_per_user=tt.Mean("sessions"),
    orders_per_session=tt.RatioOfMeans("orders", "sessions"),
    orders_per_user=tt.Mean("orders"),
    revenue_per_user=tt.Mean("revenue"),
)

experiment_results = experiment.analyze(users_data)
print(experiment_results.to_pandas())
```

In the following sections, each step of this process will be explained in detail.

### Input data

The `make_users_data` function creates synthetic data for demonstration purposes. This data mimics what you might encounter in an A/B test for an online store. Each row represents an individual user, with the following columns:

- `user`: The unique identifier for each user.
- `variant`: The specific variant (e.g., 0 or 1) assigned to the user in the A/B test.
- `sessions`: The total number of sessions by the user.
- `orders`: The total number of purchases made by the user.
- `revenue`: The total revenue generated from the user's purchases.

**tea-tasting** accepts data as either a Pandas DataFrame or an Ibis Table. [Ibis](https://ibis-project.org/) is a Python package which serves as a DataFrame API to various data backends. It supports 20+ backends including BigQuery, ClickHouse, DuckDB, Polars, PostgreSQL, Snowflake, Spark etc. You can write an SQL-query, wrap it as an Ibis Table and pass it to **tea-tasting**.

Many statistical tests, like Student's t-test or Z-test, don't need granular data for analysis. For such tests, **tea-tasting** will query aggregated statistics, like mean and variance, instead of downloading all the detailed data.

**tea-tasting** assumes that:

- The data is grouped by randomization units, such as individual users.
- There is a column indicating the variant of the A/B test (typically labeled as A, B, etc.).
- All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the table.

### A/B test definition

The `Experiment` class defines the parameters of an A/B test: metrics and a variant column name. There are two ways to define metrics:

- Using keyword parameters, with metric names as parameter names and metric definitions as parameter values, as in example above.
- Using the first argument `metrics` which accepts metrics if a form of dictionary with metric names as keys and metric definitions as values.

By default, **tea-testing** assumes that A/B test variant is stored in a column named `"variant"`. You can change it using the `variant` parameter of the `Experiment` class.

Example usage:

```python
experiment = tt.Experiment(
    {
        "sessions per user": tt.Mean("sessions"),
        "orders per session": tt.RatioOfMeans("orders", "sessions"),
        "orders per user": tt.Mean("orders"),
        "revenue per user": tt.Mean("revenue"),
    },
    variant="variant",
)
```

### Metrics

Metrics are instances of metric classes which define how metrics are calculated. Those calculations include calculation of effect size, confidence interval, p-value and other statistics.

Use the `Mean` class to compare metric averages between variants of an A/B test. For example, average number of orders per user, where user is a randomization unit of an experiment. Specify the column containing the metric values using the first parameter `value`.

Use the `RatioOfMeans` class to compare ratios of metrics averages between variants of an A/B test. For example, average number of orders per average number of sessions. Specify the columns containing the numerator and denominator values using the parameters `numer` and `denom`.

Use the following parameters of `Mean` and `RatioOfMeans` to customize the analysis:

- `alternative`: Alternative hypothesis. The following options are available:
  - `two-sided` (default): the means are unequal.
  - `greater`: the mean in the treatment variant is greater than the mean in the control variant.
  - `less`: the mean in the treatment variant is less than the mean in the control variant.
- `confidence_level`: Confidence level of the confidence interval. Default is `0.95`.
- `equal_var`: If `False` (default), assume unequal population variances in calculation of the standard deviation and the number of degrees of freedom. Otherwise, assume equal population variance and calculated pooled standard deviation.
- `use_t`: If `True` (default), use Student's t-distribution in p-value and confidence interval calculations. Otherwise use Normal distribution.

Example usage:

```python
experiment = tt.Experiment(
    sessions_per_user=tt.Mean("sessions", alternative="greater"),
    orders_per_session=tt.RatioOfMeans("orders", "sessions", confidence_level=0.9),
    orders_per_user=tt.Mean("orders", equal_var=True),
    revenue_per_user=tt.Mean("revenue", use_t=False),
)
```

You can change the default values of these four parameters using global settings (see details below).

### Analyzing and retrieving experiment results

After defining an experiment and metrics, you can analyze the experiment data using the `analyze` method of the `Experiment` class. This method takes data as an input and returns an `ExperimentResult` object with experiment result.

```python
experiment_results = experiment.analyze(users_data)
```

By default, **tea-tasting** assumes that the variant with the lowest ID is a control. Change the default behavior using the `control` parameter:

```python
experiment_results = experiment.analyze(users_data, control=0)
```

`ExperimentResult` is a mapping. Get a metric's analysis result using metric name as a key.

```python
print(experiment_results["sessions_per_user"])
```

`ExperimentResult` provides two methods to serialize and view the experiment result (and more to come):

- `to_dicts()`: Return a sequence of dictionaries, each corresponding to a metric.
- `to_pandas()`: Return a Pandas DataFrame, each row corresponding to a metric.

```python
print(experiment_results.to_pandas())
```

The fields in the result depend on metrics. For `Mean` and `RatioOfMeans`, the fields include:

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
- `statistic`: Statistic.

## More features

### Variance reduction with CUPED/CUPAC

**tea-tasting** supports variance reduction with CUPED/CUPAC, within both `Mean` and `RatioOfMeans` classes.

Example usage:

```python
users_data = tt.make_users_data(seed=42, covariates=True)

experiment = tt.Experiment(
    sessions_per_user=tt.Mean("sessions", "sessions_covariate"),
    orders_per_session=tt.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
    ),
    orders_per_user=tt.Mean("orders", "orders_covariate"),
    revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
)

experiment_results = experiment.analyze(users_data)
print(experiment_results.to_pandas())
```

Set the `covariates` parameter of the `make_users_data` functions to `True` to add the following columns with pre-experimental data:

- `sessions_covariate`: Number of sessions before the experiment.
- `orders_covariate`: Number of orders before the experiment.
- `revenue_covariate`: Revenue before the experiment.

Define the metrics' covariates:

- In `Mean`, specify the covariate using the `covariate` parameter.
- In `RatioOfMeans`, specify the covariates for the numerator and denominator using the `numer_covariate` and `denom_covariate` parameters, respectively.

### Global settings

In **tea-tasting**, you can change defaults for the following parameters:

- `alternative`: Alternative hypothesis.
- `confidence_level`: Confidence level of the confidence interval.
- `equal_var`: If False, assume unequal population variances in calculation of the standard deviation and the number of degrees of freedom. Otherwise, assume equal population variance and calculated pooled standard deviation.
- `use_t`: If True, use Student's t-distribution in p-value and confidence interval calculations. Otherwise use Normal distribution.

Use `set_config` to set a global option value:

```python
tt.set_config(confidence_level=0.9)
```

Use `config_context` to temporarily set a global option value within a context:

```python
with tt.config_context(confidence_level=0.9):
    experiment = tt.Experiment(
        sessions_per_user=tt.Mean("sessions"),
        orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        orders_per_user=tt.Mean("orders"),
        revenue_per_user=tt.Mean("revenue"),
    )
```

Use `get_config` with the option name as a parameter to get a global option value:

```python
default_pvalue = tt.get_config("confidence_level")
```

Use `get_config` without parameters to get a dictionary of global options:

```python
global_config = tt.get_config()
```

### More than two variants

In **tea-tasting**, it's possible to analyze experiments with more than two variants. However, the variants will be compared in pairs through two-sample statistical tests.

How variant pairs are determined:

- Default control variant: When the `control` parameter of the `analyze` method is set to `None`, **tea-tasting** automatically compares each variant pair. The variant with the lowest ID in each pair is a control.
- Specified control variant: If a specific variant is set as `control`, it is then compared against each of the other variants.

The result of the analysis is a dictionary of `ExperimentResult` objects with tuples (control, treatment) as keys.

Keep in mind that **tea-tasting** does not adjust for multiple comparisons. When dealing with multiple variant pairs, additional steps may be necessary to account for this, depending on your analysis needs.

## Package name

The package name "tea-tasting" is a play of words which refers to two subjects:

- [Lady tasting tea](https://en.wikipedia.org/wiki/Lady_tasting_tea) is a famous experiment which was devised by Ronald Fisher. In this experiment, Fisher developed the null hypothesis significance testing framework to analyze a lady's claim that she could discern whether the tea or the milk was added first to a cup.
- "tea-tasting" phonetically resembles "t-testing" or Student's t-test, a statistical test developed by William Gosset.
