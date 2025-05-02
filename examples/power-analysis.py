# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "tea-tasting",
# ]
# ///

import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1 id="power-analysis">Power analysis</h1>
        <p>In tea-tasting, you can analyze the statistical power for <code>Mean</code> and <code>RatioOfMeans</code> metrics. There are three possible options:</p>
        <ul>
        <li>Calculate the effect size, given statistical power and the total number of observations.</li>
        <li>Calculate the total number of observations, given statistical power and the effect size.</li>
        <li>Calculate statistical power, given the effect size and the total number of observations.</li>
        </ul>
        <p>In this example, tea-tasting calculates statistical power given the relative effect size and the number of observations:</p>
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
        <p>Besides <code>alternative</code>, <code>equal_var</code>, <code>use_t</code>, and covariates (CUPED), the following metric parameters affect the result:</p>
        <ul>
        <li><code>alpha</code>: Significance level.</li>
        <li><code>ratio</code>: Ratio of the number of observations in the treatment relative to the control.</li>
        <li><code>power</code>: Statistical power.</li>
        <li><code>effect_size</code> and <code>rel_effect_size</code>: Absolute and relative effect size. Only one of them can be defined.</li>
        <li><code>n_obs</code>: Number of observations in the control and in the treatment together. If the number of observations is not set explicitly, it's inferred from the dataset.</li>
        </ul>
        <p>You can change the default values of <code>alpha</code>, <code>ratio</code>, <code>power</code>, and <code>n_obs</code> using the <a href="https://tea-tasting.e10v.me/user-guide/#global-settings" target="_blank">global settings</a>.</p>
        <p>tea-tasting can analyze power for several values of parameters <code>effect_size</code>, <code>rel_effect_size</code>, or <code>n_obs</code>. Example:</p>
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
        <p>You can analyze power for all metrics in the experiment. Example:</p>
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
        <p>In the example above, tea-tasting calculates both the relative and absolute effect size for all metrics for two possible sample size values, <code>10_000</code> and <code>20_000</code>.</p>
        <p>The <code>solve_power</code> methods of a <a href="https://tea-tasting.e10v.me/api/metrics/mean/#tea_tasting.metrics.mean.Mean.solve_power" target="_blank">metric</a> and of an <a href="https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.Experiment.solve_power" target="_blank">experiment</a> return the instances of <a href="https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricPowerResults" target="_blank"><code>MetricPowerResults</code></a> and <a href="https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.ExperimentPowerResult" target="_blank"><code>ExperimentPowerResult</code></a> respectively. These result classes provide the serialization methods similar to the experiment result: <code>to_dicts</code>, <code>to_arrow</code>, <code>to_pandas</code>, <code>to_polars</code>, <code>to_pretty_dicts</code>, <code>to_string</code>, <code>to_html</code>. They are also rendered as HTML tables in IPython, Jupyter, or Marimo.</p>
        """
    )
    return


if __name__ == "__main__":
    app.run()
