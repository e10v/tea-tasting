# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "narwhals>=1.25",
#     "tea-tasting",
# ]
# ///

import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Power analysis

        In tea-tasting, you can analyze the statistical power for `Mean` and `RatioOfMeans` metrics. There are three possible options:

        - Calculate the effect size, given statistical power and the total number of observations.
        - Calculate the total number of observations, given statistical power and the effect size.
        - Calculate statistical power, given the effect size and the total number of observations.

        In this example, tea-tasting calculates statistical power given the relative effect size and the number of observations:
        """
    )
    return


@app.cell
def _():
    import tea_tasting as tt

    data = tt.make_users_data(
        seed=42,
        sessions_uplift=0,
        orders_uplift=0,
        revenue_uplift=0,
        covariates=True,
    )
    orders_per_session = tt.RatioOfMeans("orders", "sessions", rel_effect_size=0.1)
    orders_per_session.solve_power(data, "power")
    return data, tt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Besides `alternative`, `equal_var`, `use_t`, and covariates (CUPED), the following metric parameters affect the result:

        - `alpha`: Significance level.
        - `ratio`: Ratio of the number of observations in the treatment relative to the control.
        - `power`: Statistical power.
        - `effect_size` and `rel_effect_size`: Absolute and relative effect size. Only one of them can be defined.
        - `n_obs`: Number of observations in the control and in the treatment together. If the number of observations is not set explicitly, it's inferred from the dataset.

        You can change the default values of `alpha`, `ratio`, `power`, and `n_obs` using the [global settings](https://tea-tasting.e10v.me/user-guide/#global-settings).

        tea-tasting can analyze power for several values of parameters `effect_size`, `rel_effect_size`, or `n_obs`. Example:
        """
    )
    return


@app.cell
def _(data, tt):
    orders_per_user = tt.Mean("orders", alpha=0.1, power=0.7, n_obs=(10_000, 20_000))
    orders_per_user.solve_power(data, "rel_effect_size")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can analyze power for all metrics in the experiment. Example:
        """
    )
    return


@app.cell
def _(data, tt):
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
    power_result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the example above, tea-tasting calculates both the relative and absolute effect size for all metrics for two possible sample size values, `10_000` and `20_000`.

        The `solve_power` methods of a [metric](https://tea-tasting.e10v.me/api/metrics/mean/#tea_tasting.metrics.mean.Mean.solve_power) and of an [experiment](https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.Experiment.solve_power) return the instances of [`MetricPowerResults`](https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricPowerResults) and [`ExperimentPowerResult`](https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.ExperimentPowerResult) respectively. These result classes provide the serialization methods similar to the experiment result: `to_dicts`, `to_arrow`, `to_pandas`, `to_polars`, `to_pretty_dicts`, `to_string`, `to_html`. They are also rendered as HTML tables in IPython, Jupyter, or Marimo.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
