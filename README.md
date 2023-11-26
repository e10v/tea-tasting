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

### Input data

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
