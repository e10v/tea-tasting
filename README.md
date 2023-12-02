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

The package is in the planning stage. This means that there is no working code at the time. This readme describes the future API of the package. See more details in Tom Preston-Werner's blog post on [Readme Driven Development](https://tom.preston-werner.com/2010/08/23/readme-driven-development).

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

- `user_id`: User ID (`int`).
- `variant`: Variant of the A/B test (`int`, `0` or `1`).
- `visits`: Number of user's visits (`int`, `>= 1`).
- `orders`: Number of user's purchases (`int`, `>= 0`, `<= visits`).
- `revenue`: Total amount of user's purchases (`float`, `>= 0`, `0` if `orders == 0`).

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

- `use_t`: Indicates to use the Student’s t-distribution (`True`) or the Normal distribution (`False`) when computing p-value and confidence interval. Default is `True`.
- `equal_var`: Not used if `use_t` is `False`. If `True`, perform a standard independent Student's t-test that assumes equal population variances. If `False`, perform Welch’s t-test, which does not assume equal population variance. Default is `False`.

The `alternative` parameter defines the alternative hypothesis. The following options are available:

- `"two-sided"` (default): The means of the distributions underlying the samples are unequal.
- `"less"`: The mean of the distribution underlying the first sample is less than the mean of the distribution underlying the second sample.
- `"greater"`: The mean of the distribution underlying the first sample is greater than the mean of the distribution underlying the second sample.

The `confidence_level` parameter defines a confidence level for the computed confidence interval.

### Ratio metrics

Ratio metrics are useful when an analysis unit differs from a randomization units. For example, you might want to compare orders per visit (the analysis unit). And there can be several visits per user (the randomization unit). It's not correct to use the `tt.SimpleMean` class in this case.

The `RatioOfMeans` class defines a ratio metric that compares ratios of averages. For example, average number of orders per average number of visits. The `numer` parameter defines a numerator column name. The `denom` parameter defines a denominator column name.

Similar to `SimpleMean`,  `RatioOfMeans` applies the Welch's t-test, Student's t-test, or Z-test, depending on parameters `use_t` and `equal_var`. It applies the delta method to calculate p-value and confidence intervals.

The `alternative` parameter defines the alternative hypothesis. The `confidence_level` parameter defines a confidence level for the computed confidence interval.

### Result

Once you've defined an experiment, you can calculate the result by calling `experiment.analyze`. It accepts the experiment data as the first parameter, `data`, and returns an instance of the `ExperimentResult` class.

The `ExperimentResult` object contains the experiment result for each metrics. You can serialize results using one of these methods:

- `to_polars`: Polars dataframe, with a row for each metric.
- `to_pandas`: Pandas dataframe, with a row for each metric.
- `to_dicts`: Sequence of dictionaries, with a dictionary for each metric.
- `to_html`: HTML table.

The list of fields depends on the metric. For `SimpleMean` and `RatioOfMeans` the fields are:

- `variant_{control_variant_id}`: Control mean.
- `variant_{treatment_variant_id}`: Treatment mean.
- `diff`: Difference of means.
- `diff_conf_int_lower`, `diff_conf_int_upper`: The lower and the upper bounds of the confidence interval of the difference of means.
- `rel_diff`: Relative difference of means.
- `rel_diff_conf_int_lower`, `rel_diff_conf_int_upper`: The lower and the upper bounds of the confidence interval of the relative difference of means.
- `pvalue`: P-value.

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

- `pre_visits`: Number of user's visits in some period before the experiment.
- `pre_orders`: Number of user's purchases in some period before the experiment.
- `pre_revenue`: Total amount of user's purchases in some period before the experiment.

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

To perform a sample ratio mismatch, use the `SampleRatio` class:

```python
experiment = tt.Experiment({
    "Visits per user": tt.SimpleMean("visits"),
    "Orders per visits": tt.RatioOfMeans(numer="orders", denom="visits"),
    "Orders per user": tt.SimpleMean("orders"),
    "Revenue per user": tt.SimpleMean("revenue"),
    "Sample ratio": tt.SampleRatio(),
})
```

By default, it expects the equal number of observations per variant. To set a different ratio use the `ratio` parameter. It accept the values of two types:

- A ratio of number of treatment observations per number of control observations, as a number. For, example `SampleRatio(0.5)` -- ratio of treatment observations per number of control observations is 1:2.
- A dictionary with variants as keys and expected ratios. For example, `SampleRatio({"A": 2, "B": 1})`.

The statistical criteria depends on the `test` parameter:

- `"auto"` (default): Perform the binomial test if the number of observations is less than 1000. Otherwise perform the G-test.
- `"binomial"`: Binomial test.
- `"g-test"`: G-test.
- `"chi-squared"`: Pearson’s chi-squared test.

The results contains the following fields:

- `variant_{control_variant_id}`: Number of observations in control.
- `variant_{treatment_variant_id}`: Number of observations in treatment.
- `ratio`: Ratio of the number of observations in treatment relative to control.
- `ratio_conf_int_lower`, `ratio_conf_int_upper`: The lower and the upper bounds of the confidence interval of the ratio. Only for binomial test.
- `pvalue`: P-value. The nul hypothesis is that the actual ratio of the number of observations is equal to the expected.

The `confidence_level` parameter defines a confidence level for the computed confidence interval. The default is `0.95`.

### Power analysis

Both classes, `SimpleMean` and `RatioOfMeans`, provide two methods for power analysis:

- `power`: Calculate the power of a test.
- `solve_power`: Solve for any one parameter of the power.

Example usage:

```python
orders_power = SimpleMean("orders").power(users_data, rel_diff=0.05)
```

The `power` accepts the following parameters:

- `data`: A sample of data in the same format as the data required for the analysis of A/B test, with an exception that a column with variant of test is not required.
- `rel_diff`: Relative difference of means.
- `nobs`: Number of observations in control and a treatment in total. If `None` (default) then it will be computed from the sample.
- `alpha`: Significance level. Default is `0.05`.
- `ratio`: Ratio of the number of observations in treatment relative to control. Default is `1`.
- `alternative`: Alternative hypothesis. Default is `"two-sided"`.
- `use_t`: Indicates to use the Student’s t-distribution (`True`) or the Normal distribution (`False`) when computing power. Default is `True`.
- `equal_var`: Not used if `use_t` is `False`. If `True`, calculate the power of a standard independent Student's t-test that assumes equal population variances. If `False`, calculate the power of a Welch’s t-test, which does not assume equal population variance. Default is `False`.

The `solve_power` accepts the same parameters as the `power`. Also it accepts an additional parameter `power`, the power of a test. One parameters of `rel_diff`, `nobs`, `alpha`, `power`, `ratio` should be `None`. This is the parameter to be solved.

### Simulations and A/A tests

Tea-tasting provide the method `simulate` which:

- Randomly splits the provided dataset on treatment and control multiple times.
- Optionally, updates the treatment data in each split.
- Calculates results in each split.
- Summarizes statistics of the simulations.

This can be useful for A/A tests and for power analysis.

Example usage:

```python
aa_test = experiment.simulate(users_data, n_iter=10000)
aa_test.describe()
```

The method `simulate` accepts the following parameters:

- `data`: A sample of data in the same format as the data required for the analysis of A/B test, with an exception that a column with variant of test is not required.
- `n_iter`: Number of simulations to run. Default is `10_000`.
- `ratio`: Ratio of the number of observations in treatment relative to control. Default is `1`.
- `random_seed`: Random seed. Default is `None`.
- `treatment`: An optional function which updates a treatment data on each iteration. It should accept a Polars dataframe and return a Polars dataframe of the same length and the same set of columns. Default is `None`, which means that treatment data are not updated (A/A test).

It returns an instance of the class `SimulationsResult` which provide the following methods:

- `to_polars`: Create a Polars dataframe with detailed results, with a row for each pair (simulation, metric).
- `to_pandas`: Create a Pandas dataframe with detailed results, with a row for each pair (simulation, metric).
- `describe`: Summarize statistics of the simulations.

Methods `to_polars` and `to_pandas` return the same columns as similar methods of the experiment results. In addition, there is a column with a number of simulation.

Method `describe` returns a Polars dataframe with the following columns:

- `metric`: Metric name.
- `null_rejected`: Proportion of iterations in which the null hypothesis has been rejected. By default, it's calculated based on p-value. But if a metric doesn't provide a p-value, then confidence interval is used.
- `null_rejected_conf_int_lower`, `null_rejected_conf_int_upper`: The lower and the upper bounds of the confidence interval of the proportion iterations in which the null hypothesis has been rejected.

It accepts an optional parameter `alpha` which is used in the calculation of the proportion. But it only used in calculations based on p-values. Default is `0.05`.

## Other features

### Bootstrap

### Custom metrics

### More than two variants

### Group by units

### Global settings

alpha, confidence_level, ratio, alternative, use_t, equal_var

## Package name
