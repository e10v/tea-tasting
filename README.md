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
    "visits_per_user": tt.SimpleMean("visits"),
    "cr_visits_to_orders": tt.RatioOfMeans(numer="orders", denom="visits"),
    "orders_per_user": tt.SimpleMean("orders"),
    "revenue_per_user": tt.SimpleMean("revenue"),
})

experiment_results = experiment.analyze(users_data)
experiment_results.to_polars()
```

I'll discuss each step below.

### Input data

The `tt.sample_users_data` function samples data which can be used as an example. Data contains information about an A/B test in an online store. The randomization unit is user. It's a Polars dataframe with rows representing users and the following columns:

- `user_id` -- user ID (`int`),
- `variant` -- variant of the A/B test (`int`, `0` or `1`),
- `visits` -- number of users's visits (`int`, `>= 1`),
- `orders` -- number of users's purchases (`int`, `>= 0`, `<= visits`),
- `revenue` -- total amount of user's purchases (`float`, `>= 0`, `0` if `orders == 0`).

Tea-tasting accepts dataframes of the following types:

- Polars dataframes,
- Pandas dataframes,
- Object supporting the [Python dataframe interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html).

By default, tea-tasting assumes that:

- Data are grouped by randomization units (e.g. users).
- There is a column that represent a variant (e.g. A, B).
- There is a column for each value needed for metric calculation (e.g. number of orders,
revenue etc.).

### A/B test definition

The `tt.Experiment` class defines the A/B test. The first argument, `metrics`, is a dictionary of metric names as keys and metric definitions as values.

Also you can specify a custom variant column name using the `variant` parameter (the default value is `"variant"`):

```python
experiment = tt.Experiment(
    {
        "Visits per user": tt.SimpleMean("visits"),
        "CR visits to orders": tt.RatioOfMeans(numer="orders", denom="visits"),
        "Orders per user": tt.SimpleMean("orders"),
        "Revenue per user": tt.SimpleMean("revenue"),
    },
    variant="variant_id",
)
```

### Simple metrics

The `tt.SimpleMean` class defines average metric values per randomization units. The first argument, `value`, is a name of a column that contains the metric values.

It applies the Welch's t-test, Student's t-test, or Z-test, depending on parameters:

- `use_t: bool` -- Indicates to use the Student’s t-distribution (`True`) or the Normal distribution (`False`) when computing p-value and confidence interval. Default is `True`.
- `equal_var: bool` -- Not used if `use_t is False`. If `True`, perform a standard independent Student's t-test that assumes equal population variances. If `False`, perform Welch’s t-test, which does not assume equal population variance. Default is `False`.

### Ratio metrics

### Results

## More features

### Variance reduction with CUPED/CUPAC

### Sample ratio mismatch

### Power analysis

### Simulations and A/A tests

## Other features

### Bootstrap

### Custom metrics

### Analysis from stats

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

### Metric definition

- `tt.Experiment({"metric_name": tt.MetricType(**metric_kwargs)}, **test_kwargs)`
- `tt.Experiment(metric_name=tt.MetricType(**metric_kwargs), **test_kwargs)`
- `tt.Experiment(tt.MetricType(metric_name, **metric_kwargs), **test_kwargs)`

### Separate results class

### Dataframes
