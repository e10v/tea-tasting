# Tea-tasting: statistical analysis of A/B tests

Tea-tasting is a Python package for statistical analysis of A/B tests that features:

- T-test, Z-test, and bootstrap out of the box.
- Extensible API: You can define and use statistical tests of your choice.
- Delta method for ratio metrics.
- Variance reduction with CUPED/CUPAC (also in combination with delta method for ratio metrics).
- Fieller's confidence interval for percent change.
- Sample ratio mismatch check.
- Power analysis.
- A/A tests.

The package is in the planning stage. This means that there is no working code at this time. This readme describes the future API of the package. See more details in Tom Preston-Werner's blog post on [Readme Driven Development](https://tom.preston-werner.com/2010/08/23/readme-driven-development).

## Installation

```bash
pip install tea-tasting
```

## Basic usage

Let's start with a simple example:

```python
import tea_tasting as tt


users_data = tt.sample_users_data(size=1000, seed=42)

experiment = tt.Experiment({
    "Visits per user": tt.SimpleMean("visits"),
    "Orders per visits": tt.RatioOfMeans(numer="orders", denom="visits"),
    "Orders per user": tt.SimpleMean("orders"),
    "Revenue per user": tt.SimpleMean("revenue"),
})

experiment_result = experiment.analyze(users_data)
experiment_result.to_polars()
```

I'll discuss each step below.

### Input data

The `sample_users_data` function generates synthetic data which can be used as an example. Data contains information about an A/B test in an online store. The randomization unit is user. It's a Polars dataframe with rows representing users and the following columns:

- `user_id` -- User ID (`int`).
- `variant` -- Variant of the A/B test (`int`, `0` or `1`).
- `visits` -- Number of user's visits (`int`, `>= 1`).
- `orders` -- Number of user's purchases (`int`, `>= 0`, `<= visits`).
- `revenue` -- Total amount of user's purchases (`float`, `>= 0`, `0` if `orders == 0`).

Tea-tasting accepts dataframes of the following types:

- Polars dataframes.
- Pandas dataframes.
- Object supporting the [Python dataframe interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html).

By default, tea-tasting assumes that:

- Data are grouped by randomization units (e.g. users).
- There is a column that represent a variant (e.g. A, B).
- There is a column for each value needed for metric calculation (e.g. number of orders,
revenue etc.).

### A/B test definition

The `Experiment` class defines the A/B test. The first parameter, `metrics`, is a dictionary of metric names as keys and metric definitions as values.

You can specify a custom variant column name using the `variant` parameter. Default is `"variant"`.

Also you can specify a control variant. Default is `None`, which means that variant with minimal ID is used as control.

```python
experiment = tt.Experiment(
    {
        "Visits per user": tt.SimpleMean("visits"),
        "Orders per visits": tt.RatioOfMeans(numer="orders", denom="visits"),
        "Orders per user": tt.SimpleMean("orders"),
        "Revenue per user": tt.SimpleMean("revenue"),
    },
    variant="variant_id",
    control="A",
)
```

### Simple metrics

The `SimpleMean` class is useful if you want to compare simple metric averages. The first parameter, `value`, is a name of a column that contains the metric values.

It applies the Welch's t-test, Student's t-test, or Z-test, depending on parameters:

- `use_t: bool` -- Indicates to use the Student’s t-distribution (`True`) or the Normal distribution (`False`) when computing p-value and confidence interval. Default is `True`.
- `equal_var: bool` -- Not used if `use_t is False`. If `True`, perform a standard independent Student's t-test that assumes equal population variances. If `False`, perform Welch’s t-test, which does not assume equal population variance. Default is `False`.

The `confidence_level` parameter defines a confidence level for the computed confidence interval.

### Ratio metrics

Ratio metrics are useful when an analysis unit differs from a randomization units. For example, you might want to compare orders per visit (the analysis unit). And there can be several visits per user (the randomization unit). It's not correct to use the `tt.SimpleMean` class in this case.

The `RatioOfMeans` class defines a ratio metric that compares ratios of averages. For example, average number of orders per average number of visits. The `numer` parameter defines a numerator column name. The `denom` parameter defines a denominator column name.

Similar to `SimpleMean`,  `RatioOfMeans` applies the Welch's t-test, Student's t-test, or Z-test, depending on parameters `use_t` and `equal_var`. It applies the delta method to calculate p-value and confidence intervals. The `confidence_level` parameter defines a confidence level for the computed confidence interval.

### Result

Once you've defined an experiment, you can calculate the result by calling `experiment.analyze`. It accepts the experiment data as the first parameter, `data`, and returns an instance of the `ExperimentResult` class.

The `ExperimentResult` object contains the experiment result for each metrics. You can serialize results using one of these methods:

- `to_polars` -- Polars dataframe, with a row for each metric.
- `to_pandas` -- Pandas dataframe, with a row for each metric.
- `to_dicts` -- Sequence of dictionaries, with a dictionary for each metric.
- `to_html` -- HTML table.

The list of columns depends on the metric. For `SimpleMean` and `RatioOfMeans` the columns are:

- `variant_{control_variant_id}` -- Control mean.
- `variant_{treatment_variant_id}` -- Treatment mean.
- `diff` -- Difference of means.
- `diff_conf_int_lower`, `diff_conf_int_upper` -- The lower and the upper bounds of the confidence interval of the difference of means.
- `rel_diff` -- Relative difference of means.
- `rel_diff_conf_int_lower`, `rel_diff_conf_int_upper` -- The lower and the upper bounds of the confidence interval of the relative difference of means.
- `pvalue` -- P-value.

## More features

### Variance reduction with CUPED/CUPAC

Both `SimpleMean` and `RatioOfMeans` classes support variance reduction with CUPED/CUPAC.

```python
users_data = tt.sample_users_data(size=1000, seed=42, pre=True)

experiment = tt.Experiment(
    {
        "Visits per user": tt.SimpleMean("visits", covariate="pre_visits"),
        "Orders per visits": tt.RatioOfMeans(
            numer="orders",
            denom="visits",
            numer_covariate="pre_orders",
            denom_covariate="pre_visits",
        ),
        "Orders per user": tt.SimpleMean("orders", covariate="pre_orders",),
        "Revenue per user": tt.SimpleMean("revenue", covariate="pre_revenue"),
    },
)
```

The parameter `pre` of the function `sample_users_data` indicates whether to generate synthetic pre-experimental data in the example dataset. They will be included as additional columns:

- `pre_visits` -- Number of user's visits in some period before the experiment.
- `pre_orders` -- Number of user's purchases in some period before the experiment.
- `pre_revenue` -- Total amount of user's purchases in some period before the experiment.

You can define the covariates:

- Using the parameter `covariate` of the `SimpleMean` class.
- Using the parameters `numer_covariate` and `denom_covariate` of the `RatioOfMeans` class.

You can use a simple metric as a covariate for ratio metric as well:

```python
experiment = tt.Experiment(
    {
        "Orders per visits": tt.RatioOfMeans(
            numer="orders",
            denom="visits",
            numer_covariate="pre_orders_per_visits",
            # denom_covariate is None by default,
            # which means it's equal to 1 for all users.
        ),
    },
)
```

Actually, under the hood, `SimpleMean` utilizes the `RatioOfMeans` class:

- `SimpleMean("orders")` is similar to `tt.RatioOfMeans(numer="orders")`.
- `SimpleMean("orders", covariate="pre_orders")` is similar to `tt.RatioOfMeans(numer="orders", numer_covariate="pre_orders")`.

### Sample ratio mismatch

### Power analysis

### Simulations and A/A tests

## Other features

### Bootstrap

### Custom metrics

### More than two variants

### Group by units

## Package name

## Appendix. Design choices

### Naming

Test class:

- `Experiment`
- `ABTest`
- `Test`

Test calculation method:

- `analyze`
- `analyse`
- `compute`
- `fit`
- `calc`
- `calculate`

Out of the box metrics:

- `SimpleMean`
- `RatioOfMeans`
- `Bootstrap`
- `SampleRatio`

Confidence interval:

- `conf_int`
- `ci`

### Metric definition

- `tt.Experiment({"metric_name": tt.MetricType(**metric_kwargs)}, **test_kwargs)`
- `tt.Experiment(metric_name=tt.MetricType(**metric_kwargs), **test_kwargs)`
- `tt.Experiment(tt.MetricType(metric_name, **metric_kwargs), **test_kwargs)`

### Separate results class

### Dataframes
