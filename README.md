# tea-tasting: statistical analysis of A/B tests

**tea-tasting** is a Python package for statistical analysis of A/B tests that features:

- Student's t-test, Z-test, and Bootstrap out of the box.
- Extensible API: Define and use statistical tests of your choice.
- [Delta method](https://alexdeng.github.io/public/files/kdd2018-dm.pdf) for ratio metrics.
- Variance reduction with [CUPED](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)/[CUPAC](https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/) (also in combination with delta method for ratio metrics).
- [Fieller's confidence interval](https://en.wikipedia.org/wiki/Fieller%27s_theorem) for percent change.
- Sample ratio mismatch check.
- Power analysis.
- A/A tests.

Currently, **tea-tasting** is in the planning stage, and I'm starting with a README that outlines the proposed API — an approach known as Readme Driven Development (RDD).

Check out my [blog post](https://e10v.me/tea-tasting-rdd) where I explain the motivation for creating this package and the benefits of the RDD approach.

## Installation

```bash
pip install tea-tasting
```

## Basic usage

Begin with this simple example to understand the basic functionality:

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

In the following sections, each step of this process will be explained in detail.

### Input data

The `sample_users_data` function in **tea-tasting** creates synthetic data for demonstration purposes. This data mimics what you might encounter in an A/B test for an online store, structured as a Polars DataFrame. Each row in this DataFrame represents an individual user, with the following columns:

- `user_id`: The unique identifier for each user.
- `variant`: The specific variant (e.g., A or B) assigned to the user in the A/B test.
- `visits`: The total number of visits by the user.
- `orders`: The total number of purchases made by the user.
- `revenue`: The total revenue generated from the user's purchases.

**tea-tasting** is designed to work with DataFrames in various formats, including:

- Polars DataFrames.
- Pandas DataFrames.
- Any object that adheres to the [Python DataFrame Interchange Protocol](https://data-apis.org/dataframe-protocol/latest/index.html).

By default, **tea-tasting** assumes that:

- The data is grouped by randomization units, such as individual users.
- There is at least one column indicating the variant of the A/B test (typically labeled as A, B, etc.).
- All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the DataFrame.

### A/B test definition

The `Experiment` class in **tea-tasting** is used to define the parameters of an A/B test. The key aspects of this definition include:

- Metrics: Specified using the `metrics` parameter, which is a dictionary. Here, metric names are the keys, and their corresponding definitions are the values. These definitions determine how each metric is calculated and analyzed.
- Variant column: If your data uses a different column name to denote the variant (other than the default `"variant")`, specify this using the `variant` parameter. The default is `"variant"`.
- Control variant: To define a specific control group variant, use the `control` parameter. By default, this is set to `None`, meaning the variant with the minimal ID is automatically considered the control group.

Example usage:

```python
data = users_data.with_columns(
    pl.col("variant").replace({0: "A", 1: "B"}).alias("variant_id"),
)

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

experiment_result = experiment.analyze(data)
```

### Simple metrics

The `SimpleMean` class in **tea-tasting** facilitates the comparison of simple metric averages. Utilize this class to perform statistical tests on the average values of different metrics. Key aspects include:

- Metric column: Specify the column containing metric values using the `value` parameter.
- Statistical tests: Based on the parameters, `SimpleMean` can perform different types of tests:
  - `use_t` (default `True`): Set to `True` to use the Student's t-distribution or `False` for the Normal distribution in p-value and confidence interval calculations.
  - `equal_var` (default `False`): When `True`, a standard independent Student's t-test assuming equal population variances is used. If `False`, Welch’s t-test is used, which does not assume equal population variance. This parameter is ignored if `use_t` is set to `False`.
- Alternative hypothesis: The `alternative` parameter specifies the nature of the hypothesis test:
  - `"two-sided"` (default): Tests if the means of the two distributions are unequal.
  - `"less"`: Tests if the mean of the treatment distribution is less than that of the control.
  - `"greater"`: Tests if the mean of the treatment distribution is greater than that of the control.
- Confidence level: The `confidence_level` parameter, defaulting to `0.95`, sets the confidence level for the confidence interval of the test.

You can customize the default values for `use_t`, `equal_var`, `alternative`, and `confidence_level` in the global settings for consistent application across your analyses.

### Ratio metrics

The `RatioOfMeans` class in **tea-tasting** is specifically designed for situations where the analysis unit differs from the randomization unit. This is common in cases where you need to compare ratios, such as orders per visit, where visits per user vary.

- Defining ratio metrics: `RatioOfMeans` calculates the ratio of averages, such as the average number of orders per average number of visits. It requires two parameters:
  - `numer`: The column name for the numerator of the ratio.
  - `denom`: The column name for the denominator of the ratio.
- Statistical tests: Like `SimpleMean`, `RatioOfMeans` supports various statistical tests, including Welch's t-test, Student's t-test, and Z-test. The parameters are:
  - `use_t` (default `True`): Choose between Student's t-distribution (`True`) and Normal distribution (`False`).
  - `equal_var` (default `False`): Used to determine the type of t-test (standard or Welch’s). Irrelevant if `use_t` is `False`.
  - `alternative`: Specifies the nature of the hypothesis test (`"two-sided"`, `"less"`, or `"greater"`).
- `confidence_level` (default `0.95`): Sets the confidence level for the test.

### Analyzing and retrieving experiment results

After defining an experiment with the `Experiment` class, you can analyze the data and obtain results using the `analyze` method. This method takes your experiment data as input and returns an `ExperimentResult` object containing detailed outcomes for each defined metric.

The `ExperimentResult` object offers several methods to serialize and view the experiment results in different formats:

- `to_polars()`: Returns a Polars DataFrame with each row representing a metric.
- `to_pandas()`: Converts the results into a Pandas DataFrame, again with each row for a metric.
- `to_dicts()`: Provides a sequence of dictionaries, each corresponding to a metric.
- `to_html()`: Generates an HTML table for an easily readable web format.

The fields in the result vary based on the metric. For metrics defined using `SimpleMean` and `RatioOfMeans`, the fields include:

- `metric`: The name of the metric.
- `variant_{control_variant_id}`: The mean value for the control variant.
- `variant_{treatment_variant_id}`: The mean value for the treatment variant.
- `diff`: The difference in means between the treatment and control.
- `diff_conf_int_lower` and `diff_conf_int_upper`: The lower and upper bounds of the confidence interval for the difference in means.
- `rel_diff`: The relative difference between means.
- `rel_diff_conf_int_lower` and `rel_diff_conf_int_upper`: The confidence interval bounds for the relative difference.
- `pvalue`: The p-value from the statistical test.

## More features

### Variance reduction with CUPED/CUPAC

**tea-tasting** supports variance reduction with CUPED/CUPAC, within both `SimpleMean` and `RatioOfMeans` classes.

Example usage:

```python
import tea_tasting as tt


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

experiment_result = experiment.analyze(users_data)
```

The `sample_users_data` function's `pre` parameter controls the inclusion of pre-experimental data, useful for variance reduction. These data appear as additional columns:

- `pre_visits`: User visits before the experiment.
- `pre_orders`: User purchases before the experiment.
- `pre_revenue`: User-generated revenue before the experiment.

Defining covariates:

- In `SimpleMean`, use the `covariate` parameter to specify the pre-experimental metric.
- In `RatioOfMeans`, `numer_covariate` and `denom_covariate` define covariates for the numerator and denominator, respectively.

### Checking for sample ratio mismatch

The `SampleRatio` class in **tea-tasting** is designed to detect mismatches in the sample ratios of different variants in your A/B test.

Example usage:

```python
experiment = tt.Experiment({
    "Visits per user": tt.SimpleMean("visits"),
    "Orders per visits": tt.RatioOfMeans(numer="orders", denom="visits"),
    "Orders per user": tt.SimpleMean("orders"),
    "Revenue per user": tt.SimpleMean("revenue"),
    "Sample ratio": tt.SampleRatio(),
})
```

By default, `SampleRatio` assumes an equal number of observations across all variants. To specify a different expected ratio, use the `ratio` parameter. It accepts two types of values:

- A numerical ratio of treatment observations to control observations (e.g., 1:2), like `SampleRatio(0.5)`.
- A dictionary with variants as keys and expected ratios as values, like `SampleRatio({"A": 2, "B": 1})`.

`test` parameter determines the statistical test to apply:

- `"auto"` (default): Uses the binomial test for observations under 1000, and the G-test for larger datasets.
- `"binomial"`: Specifically uses the binomial test.
- `"g"`: Applies the G-test.
- `"pearson"`: Utilizes Pearson’s chi-squared test.

The output from `SampleRatio` includes:

- `metric`: The name of the metric.
- `variant_{control_variant_id}` and `variant_{treatment_variant_id}`: Observation counts for control and treatment.
- `ratio`: The observed ratio of treatment to control observations.
- `ratio_conf_int_lower` and `ratio_conf_int_upper`: Confidence interval bounds for the observed ratio, applicable for the binomial test.
- `pvalue`: The p-value, testing if the actual observation ratio matches the expected.

The `confidence_level` parameter, which defaults to `0.95`, sets the confidence level for the interval calculations.

### Bootstrap

The `Bootstrap` class in **tea-tasting** is designed for comparing statistics between variants using bootstrap resampling methods.

Example usage:

```python
import numpy as np


experiment = tt.Experiment({
    "Median revenue per user": tt.Bootstrap("revenue", statistic=np.median),
})
experiment_result = experiment.analyze(users_data)
```

Configuring the `Bootstrap` class:

- Statistical function: The `statistic` parameter is a callable, such as a NumPy function, that computes the statistic. It should accept a NumPy array and an `axis` parameter for computation direction.
- Target columns: The first argument of Bootstrap is either a single column name or a list of column names from which the statistic will be computed.

Additional parameters:

- `n_resamples` (default `10_000`): Sets the number of bootstrap resamples for distribution estimation.
- `confidence_level` (default `0.95`): Determines the confidence level for interval calculations.
- `method` (default `"bca"`): Whether to return the "percentile" bootstrap confidence interval (`"percentile"`), the "basic" (aka "reverse") bootstrap confidence interval (`"basic"`), or the bias-corrected and accelerated bootstrap confidence interval (`"bca"`).
- `random_seed` (default `None`): Random seed.

The `Bootstrap` class provides a detailed result output, including:

- `metric`: The name of the computed metric.
- `variant_{control_variant_id}` and `variant_{treatment_variant_id}`: Statistic values for control and treatment groups.
- `diff`: The difference between the statistic values of the treatment and control.
- `diff_conf_int_lower` and `diff_conf_int_upper`: Confidence interval bounds for this difference.
- `rel_diff`: The relative difference between the statistic values.
- `rel_diff_conf_int_lower` and `rel_diff_conf_int_upper`: Confidence interval bounds for the relative difference.

### Power analysis

Both the `SimpleMean` and `RatioOfMeans` classes in **tea-tasting** offer two methods for conducting power analysis:

- `power()`: Calculates the statistical power of the test.
- `solve_power()`: Solves for a specific power analysis parameter, such as the minimum detectable effect.

Example usage:

```python
orders_per_user = SimpleMean("orders")
orders_power = orders_per_user.power(users_data, rel_diff=0.05)
orders_mde = orders_per_user.solve_power(users_data, parameter="rel_diff")
```

Parameters for power analysis:

- `data`: The dataset, formatted similarly to A/B test analysis data, excluding the variant column.
- `rel_diff` (default `None`): Relative difference of means.
- `nobs` (default `"auto"`): Total number of observations. If `"auto"`, automatically computed from the sample.
- `alpha` (default `0.05`): Significance level.
- `ratio` (default `1`): The ratio of observations in the treatment group to the control group.
- `alternative` (default `"two-sided"`): The alternative hypothesis.
- `use_t` (default `True`): Determines whether to use the Student's t-distribution (`True`) or Normal distribution (`False`).
- `equal_var` (default `False`): Relevant only when `use_t` is `True`. If `True`, assumes equal population variances for a standard Student's t-test. If `False`, uses Welch’s t-test.

Additional parameters for `solve_power`:

- `power` (default `0.8`): The desired power of the test.
- `parameter` (default `"rel_diff"`): The specific parameter to solve for.

You can customize the default values for the parameters `ratio`, `use_t`, `equal_var`, `alternative` and `alpha` globally in **tea-tasting**'s settings.

### Simulations and A/A tests

**tea-tasting**'s `simulate` method is useful in A/A testing and power analysis. How it works:

- Data splitting: Randomly divides the dataset into treatment and control groups multiple times.
- Data updating (optional): Allows for the modification of treatment data in each iteration.
- Result calculation: Computes the results for each data split.
- Statistical summary: Aggregates and summarizes the statistical outcomes of all simulations.

Example usage:

```python
aa_test = experiment.simulate(users_data)
aa_test.describe()
```

Parameters of `simulate`:

- `data`: The dataset for simulation, similar in format to A/B testing data but without the variant column.
- `n_iter` (default `10_000`): Determines the number of simulation runs.
- `ratio` (default `1`): The expected ratio of treatment to control observations.
- `random_seed` (default `None`): Sets a seed for reproducibility.
- `treatment` (default `None`): A function to modify treatment data per iteration. If `None`, the treatment data is not updated, suitable for A/A testing.

The simulate method returns a `SimulationsResult` object, offering several ways to access and analyze the simulation data:

- `to_polars()`: Converts results into a Polars DataFrame, detailing each simulation and metric.
- `to_pandas()`: Provides a similar output in a Pandas DataFrame format.
- `describe()`: Offers a summary of the simulations, including metrics such as:
  - `metric`: The name of each analyzed metric.
  - `null_rejected`: The proportion of simulations where the null hypothesis was rejected, calculated using either p-value or confidence intervals.
  - `null_rejected_conf_int_lower` and `null_rejected_conf_int_upper`: Bounds of the confidence interval for the proportion of null hypothesis rejections.

Additional parameters for `describe`:

- `alpha`: Threshold for p-value based calculations of null hypothesis rejection.
- `confidence_level`: Sets the confidence level for interval estimations.

## Other features

### Global settings

In **tea-tasting**, global settings allow you to manage default values for various parameters across all metrics.

You can globally set defaults for the following parameters:

- `alpha`: Significance level for statistical tests.
- `alternative`: Specifies the alternative hypothesis in testing.
- `confidence_level`: Sets the confidence level for intervals.
- `equal_var`: Determines the assumption of equal variances in tests.
- `n_resamples`: Number of resamples for bootstrap methods.
- `ratio`: Default ratio of treatment to control observations.
- `use_t`: Chooses between the Student's t-distribution and the Normal distribution.

Additionally, custom parameter defaults can also be defined.

Use `set_config` to set a global option value:

```python
tt.set_config(confidence_level=0.98, some_custom_parameter=1)
```

Use `config_context` to temporarily set a global option value within a context:

```python
with tt.config_context(confidence_level=0.98, some_custom_parameter=1):
    # Define the experiment and the metrics here.
```

Use `get_config` with the option name as a parameter to get a global option value:

```python
default_pvalue = tt.get_config("confidence_level")
```

Use `get_config` without parameters to get a dictionary of global options:

```python
global_config = tt.get_config()
```

### Custom metrics

In **tea-tasting**, you can create custom metrics. This is done by defining a new class that inherits from `MetricBase` and implements at least two key methods: `__init__` and `analyze`:

- `__init__`: Initializes the new metric, setting up any necessary parameters.
- `analyze`: Performs the actual metric analysis. This method should accept:
  - `contr_data`: A Polars DataFrame containing control group data.
  - `treat_data`: A Polars DataFrame with treatment group data.
  - `contr_variant`: The identifier for the control variant.
  - `treat_variant`: The identifier for the treatment variant.

The `analyze` method should return a `NamedTuple` containing the analysis results. It's important to use consistent field names across different metrics for uniformity (e.g., use `pvalue` instead of `p_value`). Note that field names starting with `_` are not included in the final experiment results.

Here's an example demonstrating how to create a custom metric using the Mann-Whitney U test:

```python
from typing import Any, NamedTuple

import polars as pl
import scipy.stats
import tea_tasting as tt


tt.set_config(use_continuity=True)  # Set default value for a custom parameter.

class MannWhitneyUResult(NamedTuple):
    pvalue: float
    _statistic: float  # Not used in experiment results.

class MannWhitneyU(tt.MetricBase):
    def __init__(
        self,
        value: str,
        use_continuity: bool | None = None,
        alternative: str | None = None,
        method: str = "auto",
        nan_policy: str = "propagate",
    ):
        self.value = value

        # Get default value for a custom parameter.
        self.use_continuity = use_continuity or tt.get_config("use_continuity")

        # Get default value for a standard parameter.
        self.alternative = alternative or tt.get_config("alternative")

        self.method = method
        self.nan_policy = nan_policy

    def analyze(
        self,
        contr_data: pl.DataFrame,
        treat_data: pl.DataFrame,
        contr_variant: Any,  # Not used.
        treat_variant: Any,  # Not used.
    ) -> MannWhitneyUResult:
        res = scipy.stats.mannwhitneyu(
            treat_data.get_column(self.value).to_numpy(),
            contr_data.get_column(self.value).to_numpy(),
            use_continuity=self.use_continuity,
            alternative=self.alternative,
            method=self.method,
            nan_policy=self.nan_policy,
        )

        return MannWhitneyUResult(pvalue=res.pvalue. _statistic=res.statistic)


users_data = tt.sample_users_data(size=1000, seed=42)
experiment = tt.Experiment({"Revenue rank test": MannWhitneyU("revenue")})
experiment_result = experiment.analyze(users_data)
experiment_result.to_polars()
```

### More than two variants

In **tea-tasting**, it's possible to analyze experiments with more than two variants. However, the variants will be compared in pairs through two-sample statistical tests.

How variant pairs are determined:

- Default control variant: When the `control` parameter is set to `None`, **tea-tasting** automatically compares each variant pair. The variant with the lowest ID in each pair acts as the control.
- Specified control variant: If a specific variant is set as `control`, it is then compared against each of the other variants.

To access the results, use one of the following methods and specify both control and treatment variant IDs:

- `to_pandas()`,
- `to_polars()`,
- `to_dicts()`,
- `to_html()`.

It's important to note that **tea-tasting** does not adjust for multiple comparisons. When dealing with multiple variant pairs, additional steps may be necessary to account for this, depending on your analysis needs.

### Group by units

By default, **tea-tasting** assumes data are grouped by randomization units, such as users. But sometimes one might want to perform clustered error analysis, multilevel modelling, or other methods that rely on more detailed data, such as visits.

Steps to analyze data grouped by different units:

- Define a custom metric for detailed data:
  - When you need to work with detailed data, create a custom metric (for example, a class-based metric like `CRSE`).
  - In your custom metric, set the attribute `use_raw_data` to `True`. This tells **tea-tasting** to work with the data as is, without pre-aggregating it.
- Set up your experiment with specific randomization units:
  - When defining your `Experiment`, specify the randomization unit using the `randomization_unit` parameter.
  - This parameter should be either a single column name or a sequence of column names that define your randomization unit (e.g., `randomization_unit="user_id"` for user-level analysis).
- Analyze with detailed data:
  - Perform the analysis by calling the `analyze` method on your experiment, passing in the detailed dataset.

How **tea-tasting** manages this data:

- For the custom metric that uses detailed data, **tea-tasting** directly applies the analysis on this raw, detailed dataset.
- For other metrics in the experiment, **tea-tasting** will first group the data by the specified randomization units before conducting the analysis.

### Analyze from stats

**tea-tasting** facilitates analysis using aggregated statistics, particularly beneficial for statistical tests like Z-test or Student's t-test, where full datasets are not necessary. This method is often more computationally efficient, especially when these aggregated statistics can be calculated directly within a database.

The `experiment.analyze_from_stats` method is specifically designed for this type of analysis. It works with metrics like `SimpleMean`, `RatioOfMeans`, and `SampleRatio` that support analysis based on aggregated statistics.

This method requires a dictionary where each key is a variant ID, and the corresponding value is an instance of the `Stats` class, representing aggregated statistics for that variant.

To prepare your data for this method, initialize the `Stats` class with these parameters:

- `mean`: A dictionary with column names as keys and their mean values as values.
- `var`: A dictionary mapping column names to their variances.
- `cov`: A dictionary with tuples of column names `(column_name_1, column_name_2)` as keys and the covariance between these columns as values. Ensure these column names are in alphabetical order.
- `nobs`: The total number of observations per variant.

## Package name

The package name "tea-tasting" is a play of words which refers to two subjects:

- [Lady tasting tea](https://en.wikipedia.org/wiki/Lady_tasting_tea) is a famous experiment which was devised by Ronald Fisher. In this experiment, Fisher developed the null hypothesis significance testing framework to analyze a lady's claim that she could discern whether the tea or the milk was added first to a cup.
- "tea-tasting" phonetically resembles "t-testing" or Student's t-test, a statistical test developed by William Gosset.
