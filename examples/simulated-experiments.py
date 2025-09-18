# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
#     "tea-tasting",
# ]
# [tool.marimo.display]
# cell_output = "below"
# ///

import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Simulated experiments

        ## Intro

        In tea-tasting, you can run multiple simulated A/A or A/B tests. In each simulation, tea-tasting splits the data into control and treatment groups and can optionally modify the treatment data. A simulation without changing the treatment data is called an A/A test.

        A/A tests are useful for identifying potential issues before conducting the actual A/B test. Treatment simulations are great for power analysis—especially when you need a specific uplift distribution or when an analytical formula doesn’t exist.

        /// admonition | Note

        This guide uses [Polars](https://github.com/pola-rs/polars) and [marimo](https://github.com/marimo-team/marimo). Install these packages in addition to tea-tasting to reproduce the examples:

        ```bash
        uv pip install polars marimo
        ```

        ///

        ## Running A/A tests

        First, let's prepare the data without any uplift and drop the `"variant"` column.
        """
    )
    return


@app.cell
def _():
    import polars as pl
    import tea_tasting as tt

    data = (
        tt.make_users_data(seed=42, orders_uplift=0, revenue_uplift=0)
        .drop_columns("variant")
    )
    data
    return data, pl, tt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To run A/A tests, first define the metrics for the experiment, then call the [`simulate`](https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.Experiment.simulate) method, providing the data and the number of simulations as arguments.
        """
    )
    return


@app.cell
def _(data, tt):
    experiment = tt.Experiment(
        sessions_per_user=tt.Mean("sessions"),
        orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        orders_per_user=tt.Mean("orders"),
        revenue_per_user=tt.Mean("revenue"),
        n_users=tt.SampleRatio(),
    )
    results = experiment.simulate(data, 100, seed=42)
    results_data = results.to_polars()
    results_data.select(
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "rel_effect_size_ci_lower",
        "rel_effect_size_ci_upper",
        "pvalue",
    )
    return experiment, results_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `simulate` method accepts data in the same formats as the `analyze` method. Internally, however, it converts the data to a PyArrow Table before running the simulations.

        The method returns an instance of the [`SimulationResults`](https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.SimulationResults) class, which contains the results of all simulations for all metrics. The resulting object provides serialization methods to those of the experiment result, including `to_dicts`, `to_arrow`, `to_pandas`, `to_polars`, `to_pretty_dicts`, `to_string`, `to_html`.

        For instance, we can now calculate the proportion of rejected null hypotheses, using various significance levels (`alpha`). In A/A tests, it estimates the type I error rate.
        """
    )
    return


@app.cell
def _(pl, results_data):
    def null_rejected(
        results_data: pl.DataFrame,
        alphas: tuple[float, ...] = (0.01, 0.02, 0.05),
    ) -> pl.DataFrame:
        return results_data.group_by("metric", maintain_order=True).agg(
            pl.col("pvalue").le(alpha).mean().alias(f"null_rejected_{alpha}")
            for alpha in alphas
        )

    null_rejected(results_data)
    return (null_rejected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        100 simulations, as in the example above, produce a very rough estimate. In practice, a larger number of simulations, such as the default `10_000`, is recommended.

        ## Simulating experiments with treatment

        To simulate experiments with treatment, define a treatment function that takes data in the form of a PyArrow Table and returns a PyArrow Table with the modified data:
        """
    )
    return


@app.cell
def _(data, experiment, null_rejected):
    import pyarrow as pa
    import pyarrow.compute as pc

    def treat(data: pa.Table) -> pa.Table:
        return (
            data.drop_columns(["orders", "revenue"])
            .append_column("orders", pc.multiply(data["orders"], pa.scalar(1.1)))
            .append_column("revenue", pc.multiply(data["revenue"], pa.scalar(1.1)))
        )

    results_treat = experiment.simulate(data, 100, seed=42, treat=treat)
    null_rejected(results_treat.to_polars())
    return (treat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the example above, we've defined a function that increases the number of orders and the revenue by 10%. For these metrics, the proportion of rejected null hypotheses is an estimate of statistical power.

        ## Using a function instead of static data

        You can use a function instead of static data to generate input dynamically. The function should take an instance of `numpy.random.Generator` as a parameter named `seed` and return experimental data in any format supported by tea-tasting.

        As an example, let's use the `make_users_data` function.
        """
    )
    return


@app.cell
def _(experiment, null_rejected, tt):
    results_data_gen = experiment.simulate(tt.make_users_data, 100, seed=42)
    null_rejected(results_data_gen.to_polars())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        On each iteration, tea-tasting calls `make_users_data` with a new `seed` and uses the returned data for the analysis of the experiment. The data returned by `make_users_data` already contains the `"variant"` column, so tea-tasting reuses that split. By default, `make_users_data` also adds the treatment uplift, and you can see it in the proportion of rejected null hypotheses.

        ## Tracking progress

        To track the progress of simulations with [`tqdm`](https://github.com/tqdm/tqdm) or [`marimo.status.progress_bar`](https://docs.marimo.io/api/status/#progress-bar), use the `progress` parameter.
        """
    )
    return


@app.cell
def _(data, experiment, mo):
    results_progress = experiment.simulate(
        data,
        100,
        seed=42,
        progress=mo.status.progress_bar,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel execution

        /// admonition | Note

        The code below won't work in the [marimo online playground](https://docs.marimo.io/guides/publishing/playground/) as it relies on the `multiprocessing` module which is currently [not supported](https://docs.marimo.io/guides/wasm/#limitations) by WASM notebooks. [WASM notebooks](https://docs.marimo.io/guides/wasm/) are the marimo notebooks that run entirely in the browser.

        ///

        To speed up simulations and run them in parallel, use the `map_` parameter with an alternative mapping function.
        """
    )
    return


@app.cell
def _(data, experiment, mo, treat):
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_parallel = experiment.simulate(
            data,
            100,
            seed=42,
            treat=treat,
            map_=executor.map,
            progress=mo.status.progress_bar,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As an alternative to [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor), you can use the `map`, `imap`, or `imap_unordered` methods of [`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool).

        It's also possible to run simulations on a distributed [Dask](https://distributed.dask.org/en/stable/api.html#distributed.Client.map) or [Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.map.html#ray.util.ActorPool.map) cluster.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
