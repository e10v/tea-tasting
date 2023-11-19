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

## Test and metrics definition

First, define the test and its metrics:

```python
import tea_tasting as tt

test = tt.Test()
```

## Input data

## Package name

## Decisions

### Naming

Test class:

- `Test`
- `ABTest`
- `Experiment`

Test calculation method:

- `analyse`
- `analyze`
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

- `tt.Test({"metric_name": tt.MetricType(**metric_kwargs)}, **test_kwargs)`
- `tt.Test(metric_name=tt.MetricType(**metric_kwargs), **test_kwargs)`
- `tt.Test(tt.MetricType(metric_name, **metric_kwargs), **test_kwargs)`

### Immutable objects
