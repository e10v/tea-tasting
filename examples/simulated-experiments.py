# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
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
        <h1 id="simulated-experiments">Simulated experiments</h1>
        <h2 id="intro">Intro</h2>
        <p>In tea-tasting, you can run multiple simulated A/A or A/B tests. In each simulation, tea-tasting splits the data into control and treatment groups and can optionally modify the treatment data. A simulation without changing the treatment data is called an A/A test. A/A tests are useful for identifying potential issues before conducting the actual A/B test. Simulations where the treatment data is modified are useful for power analysis, especially when you need a specific uplift distribution or when an analytical solution isn't feasible.</p>
        <details class="note" open="open">
        <summary>Note</summary>
        <p>This guide uses <a href="https://github.com/pola-rs/polars" target="_blank">Polars</a> and <a href="https://github.com/tqdm/tqdm" target="_blank">tqdm</a>. Install these packages in addition to tea-tasting to reproduce the examples:</p>
        <div class="highlight"><pre><span></span><code>pip<span class="w"> </span>install<span class="w"> </span>polars<span class="w"> </span>tqdm
        </code></pre></div>
        </details>
        <h2 id="running-aa-tests">Running A/A tests</h2>
        <p>First, let's prepare the data without any uplift and drop the <code>"variant"</code> column.</p>
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
        <p>To run A/A tests, first define the metrics for the experiment, then call the <a href="https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.Experiment.simulate" target="_blank"><code>simulate</code></a> method, providing the data and the number of simulations as arguments.</p>
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
        <p>The <code>simulate</code> method accepts data in the same formats as the <code>analyze</code> method. Internally, however, it converts the data to a PyArrow Table before running the simulations.</p>
        <p>The method returns an instance of the <a href="https://tea-tasting.e10v.me/api/experiment/#tea_tasting.experiment.SimulationResults" target="_blank"><code>SimulationResults</code></a> class, which contains the results of all simulations for all metrics. The resulting object provides serialization methods to those of the experiment result, including <code>to_dicts</code>, <code>to_arrow</code>, <code>to_pandas</code>, <code>to_polars</code>, <code>to_pretty_dicts</code>, <code>to_string</code>, <code>to_html</code>.</p>
        <p>For instance, we can now calculate the proportion of rejected null hypotheses, using various significance levels (<code>alpha</code>). In A/A tests, it estimates the type I error rate.</p>
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
        <p>100 simulations, as in the example above, produce a very rough estimate. In practice, a larger number of simulations, such as the default <code>10_000</code>, is recommended.</p>
        <h2 id="simulating-experiments-with-treatment">Simulating experiments with treatment</h2>
        <p>To simulate experiments with treatment, define a treatment function that takes data in the form of a PyArrow Table and returns a PyArrow Table with the modified data:</p>
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
        <p>In the example above, we've defined a function that increases the number of orders and the revenue by 10%. For these metrics, the proportion of rejected null hypotheses is an estimate of statistical power.</p>
        <h2 id="using-a-function-instead-of-static-data">Using a function instead of static data</h2>
        <p>You can use a function instead of static data to generate input dynamically. The function should take an instance of <code>numpy.random.Generator</code> as a parameter named <code>seed</code> and return experimental data in any format supported by tea-tasting.</p>
        <p>As an example, let's use the <code>make_users_data</code> function.</p>
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
        <p>On each iteration, tea-tasting calls <code>make_users_data</code> with a new <code>seed</code> and uses the returned data for the analysis of the experiment. The data returned by <code>make_users_data</code> already contains the <code>"variant"</code> column, so tea-tasting reuses that split. By default, <code>make_users_data</code> also adds the treatment uplift, and you can see it in the proportion of rejected null hypotheses.</p>
        <h2 id="tracking-progress">Tracking progress</h2>
        <p>To track the progress of simulations with <a href="https://tqdm.github.io/" target="_blank"><code>tqdm.tqdm</code></a>, use the <code>progress</code> parameter.</p>
        """
    )
    return


@app.cell
def _(data, experiment):
    import tqdm

    results_progress = experiment.simulate(data, 100, seed=42, progress=tqdm.tqdm)
    return (tqdm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h2 id="parallel-execution">Parallel execution</h2>
        <p>To speed up simulations and run them in parallel, use the <code>map_</code> parameter with an alternative mapping function.</p>
        """
    )
    return


@app.cell
def _(data, experiment, tqdm, treat):
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_parallel = experiment.simulate(
            data,
            100,
            seed=42,
            treat=treat,
            map_=executor.map,
            progress=tqdm.tqdm,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>As an alternative to <a href="https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor" target="_blank"><code>concurrent.futures.ProcessPoolExecutor</code></a>, you can use the <code>map</code>, <code>imap</code>, or <code>imap_unordered</code> methods of <a href="https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool" target="_blank"><code>multiprocessing.pool.Pool</code></a>.</p>
        <p>It's also possible to run simulations on a distributed <a href="https://distributed.dask.org/en/stable/api.html#distributed.Client.map" target="_blank">Dask</a> or <a href="https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.map.html#ray.util.ActorPool.map" target="_blank">Ray</a> cluster.</p>
        """
    )
    return


if __name__ == "__main__":
    app.run()
