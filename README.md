# Tea-tasting: statistical analysis of A/B tests

Tea-tasting is a Python package for statistical analysis of A/B tests that features:

- T-test, Z-test, and bootstrap out of the box
- Extensible API: You can define and use statistical tests of your choice
- Delta method for ratio metrics
- Variance reduction with CUPED/CUPAC (also in combination with delta method for ratio metrics)
- Fieller's confidence interval for percent change
- Sample ratio mismatch check
- Power analysis
- A/A tests

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

Function `tt.sample_users_data` samples data which can be used as an example. Data contains information about an A/B test in an online store. The randomization unit is user. It's a Polars dataframe with rows representing users and the following columns:

- `user_id` -- user ID (`int`)
- `variant` -- variant of the A/B test (`int`, `0` or `1`)
- `visits` -- number of users's visits (`int`, `>= 1`)
- `orders` -- number of users's purchases (`int`, `>= 0`, `<= visits`)
- `revenue` -- total amount of user's purchases (`float`, `>= 0`, `0` if `orders == 0`)

Tea-tasting accepts dataframes of the following types:

- Polars dataframes
- Pandas dataframes
- Object supporting the [Python dataframe interchange protocol](https://data-apis.org/dataframe-protocol/latest/index.html)

By default, tea-tasting assumes that:

- Data are grouped by randomizaion units (e.g. users)
- There is a column that represnet a variant (e.g. A, B)
- There is a column for each value needed for metric calculation (e.g. number of orders,
revenue etc.)

### A/B test definition

Class `tt.Experiment` defines the A/B test. The first argument, `metrics`, is a dictionary of metric names as keys and metric definitions as values.

Also you can specify a custom variant column name using `variant` parameter (the default value is `"variant"`):

```python
experiment = tt.Experiment(
    {
        "visits_per_user": tt.SimpleMean("visits"),
        "cr_visits_to_orders": tt.RatioOfMeans(numer="orders", denom="visits"),
        "orders_per_user": tt.SimpleMean("orders"),
        "revenue_per_user": tt.SimpleMean("revenue"),
    },
    variant="variant_id",
)
```

### Simple metrics

### Ratio metrics

### Results

## More features

### Variance reduction with CUPED / CUPAC

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
