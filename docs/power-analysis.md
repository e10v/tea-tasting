# Power analysis

In **tea-tasting**, you can analyze statistical power for `Mean` and `RatioOfMeans` metrics. There are three possible options:

- Calculate the effect size, given statistical power and the total number of observations.
- Calculate the total number of observations, given statistical power and the effect size.
- Calculate statistical power, given the effect size and the total number of observations.

In the following example, **tea-tasting** calculates statistical power given the relative effect size and the number of observations:

```python
import tea_tasting as tt


data = tt.make_users_data(
    seed=42,
    sessions_uplift=0,
    orders_uplift=0,
    revenue_uplift=0,
    covariates=True,
)

orders_per_session = tt.RatioOfMeans("orders", "sessions", rel_effect_size=0.1)
print(orders_per_session.solve_power(data, "power"))
#> power effect_size rel_effect_size n_obs
#>   52%      0.0261             10%  4000
```

Besides `alternative`, `equal_var`, `use_t`, and covariates (CUPED), the following metric parameters impact the result:

- `alpha`: Significance level.
- `ratio`: Ratio of the number of observations in the treatment relative to the control.
- `power`: Statistical power.
- `effect_size` and `rel_effect_size`: Absolute and relative effect size. Only one of them can be defined.
- `n_obs`: Number of observations in the control and in the treatment together. If the number of observations is not set explicitly, it's inferred from the dataset.

You can change default values of `alpha`, `ratio`, `power`, and `n_obs` using the [global settings](user-guide.md#global-settings).

**tea-tasting** can analyze power for several values of parameters `effect_size`, `rel_effect_size`, or `n_obs`. Example:

```python
orders_per_user = tt.Mean("orders", alpha=0.1, power=0.7, n_obs=(10_000, 20_000))
print(orders_per_user.solve_power(data, "rel_effect_size"))
#> power effect_size rel_effect_size n_obs
#>   70%      0.0367            7.1% 10000
#>   70%      0.0260            5.0% 20000
```

You can analyze power for all metrics in the experiment. Example:

```python
with tt.config_context(n_obs=(10_000, 20_000)):
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

power_result = experiment.solve_power(data)
print(power_result)
#>             metric power effect_size rel_effect_size n_obs
#>  sessions_per_user   80%      0.0458            2.3% 10000
#>  sessions_per_user   80%      0.0324            1.6% 20000
#> orders_per_session   80%      0.0177            6.8% 10000
#> orders_per_session   80%      0.0125            4.8% 20000
#>    orders_per_user   80%      0.0374            7.2% 10000
#>    orders_per_user   80%      0.0264            5.1% 20000
#>   revenue_per_user   80%       0.488            9.2% 10000
#>   revenue_per_user   80%       0.345            6.5% 20000
```

In the example above, **tea-tasting** calculates the relative and absolute effect size for all metrics for two possible sample size values, `10_000` and `20_000`.

The `solve_power` methods of a [metric](api/metrics/mean.md#tea_tasting.metrics.mean.Mean.solve_power) and of an [experiment](api/experiment.md#tea_tasting.experiment.Experiment.solve_power) return the instances of [`MetricPowerResults`](api/metrics/base.md#tea_tasting.metrics.base.MetricPowerResults) and [`ExperimentPowerResult`](api/experiment.md#tea_tasting.experiment.ExperimentPowerResult) respectively. These result classes provide the serialization methods similar to the experiment result: `to_dicts`, `to_pandas`, `to_pretty`, `to_string`, `to_html`.
