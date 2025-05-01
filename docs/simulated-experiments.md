# Simulated experiments

## Intro

In **tea-tasting**, you can run multiple simulated A/A or A/B tests. In each simulation, **tea-tasting** splits the data into control and treatment groups and can optionally modify the treatment data. A simulation without changing the treatment data is called an A/A test. A/A tests are useful for identifying potential issues before conducting the actual A/B test. Simulations where the treatment data is modified are useful for power analysis, especially when you need a specific uplift distribution or when an analytical solution isn't feasible.

???+ note

    This guide uses [Polars](https://github.com/pola-rs/polars) and [tqdm](https://github.com/tqdm/tqdm). Install these packages in addition to **tea-tasting** to reproduce the examples:

    ```bash
    pip install polars tqdm
    ```

## Running A/A tests

First, let's prepare the data without any uplift and drop the `"variant"` column.

```pycon
>>> import polars as pl
>>> import tea_tasting as tt

>>> data = (
...     tt.make_users_data(seed=42, orders_uplift=0, revenue_uplift=0)
...     .drop_columns("variant")
... )
>>> data
pyarrow.Table
user: int64
sessions: int64
orders: int64
revenue: double
----
user: [[0,1,2,3,4,...,3995,3996,3997,3998,3999]]
sessions: [[2,2,2,2,1,...,2,2,3,1,5]]
orders: [[1,1,1,0,1,...,0,1,1,0,4]]
revenue: [[19.06,12.09,8.84,0,9.9,...,0,4.8,9.63,0,12.7]]

```

To run A/A tests, first define the metrics for the experiment, then call the [`simulate`](api/experiment.md#tea_tasting.experiment.Experiment.simulate) method, providing the data and the number of simulations as arguments.

```pycon
>>> experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
...     n_users=tt.SampleRatio(),
... )
>>> results = experiment.simulate(data, 100, seed=42)
>>> results_data = results.to_polars()
>>> results_data.select(
...     "metric",
...     "control",
...     "treatment",
...     "rel_effect_size",
...     "rel_effect_size_ci_lower",
...     "rel_effect_size_ci_upper",
...     "pvalue",
... )  # doctest: +SKIP
shape: (500, 7)
┌────────────────────┬──────────┬───────────┬─────────────────┬────────────────────┬────────────────────┬──────────┐
│ metric             ┆ control  ┆ treatment ┆ rel_effect_size ┆ rel_effect_size_ci ┆ rel_effect_size_ci ┆ pvalue   │
│ ---                ┆ ---      ┆ ---       ┆ ---             ┆ _lower             ┆ _upper             ┆ ---      │
│ str                ┆ f64      ┆ f64       ┆ f64             ┆ ---                ┆ ---                ┆ f64      │
│                    ┆          ┆           ┆                 ┆ f64                ┆ f64                ┆          │
╞════════════════════╪══════════╪═══════════╪═════════════════╪════════════════════╪════════════════════╪══════════╡
│ sessions_per_user  ┆ 1.98004  ┆ 1.998998  ┆ 0.009575        ┆ -0.021272          ┆ 0.041393           ┆ 0.547091 │
│ orders_per_session ┆ 0.263105 ┆ 0.258647  ┆ -0.016945       ┆ -0.108177          ┆ 0.083621           ┆ 0.730827 │
│ orders_per_user    ┆ 0.520958 ┆ 0.517034  ┆ -0.007532       ┆ -0.102993          ┆ 0.098087           ┆ 0.883462 │
│ revenue_per_user   ┆ 5.446662 ┆ 5.14521   ┆ -0.055346       ┆ -0.162811          ┆ 0.065914           ┆ 0.356327 │
│ n_users            ┆ 2004.0   ┆ 1996.0    ┆ null            ┆ null               ┆ null               ┆ 0.91187  │
│ …                  ┆ …        ┆ …         ┆ …               ┆ …                  ┆ …                  ┆ …        │
│ sessions_per_user  ┆ 1.993624 ┆ 1.985212  ┆ -0.00422        ┆ -0.034685          ┆ 0.027207           ┆ 0.78959  │
│ orders_per_session ┆ 0.269373 ┆ 0.251991  ┆ -0.064527       ┆ -0.151401          ┆ 0.03124            ┆ 0.179445 │
│ orders_per_user    ┆ 0.537028 ┆ 0.500255  ┆ -0.068475       ┆ -0.158141          ┆ 0.030742           ┆ 0.169217 │
│ revenue_per_user   ┆ 5.511967 ┆ 5.071928  ┆ -0.079833       ┆ -0.184806          ┆ 0.038656           ┆ 0.177868 │
│ n_users            ┆ 2039.0   ┆ 1961.0    ┆ null            ┆ null               ┆ null               ┆ 0.223423 │
└────────────────────┴──────────┴───────────┴─────────────────┴────────────────────┴────────────────────┴──────────┘

```

The `simulate` method accepts data in the same formats as the `analyze` method. Internally, however, it converts the data to a PyArrow Table before running the simulations.

The method returns an instance of the [`SimulationResults`](api/experiment.md#tea_tasting.experiment.SimulationResults) class, which contains the results of all simulations for all metrics. The resulting object provides serialization methods to those of the experiment result, including `to_dicts`, `to_arrow`, `to_pandas`, `to_polars`, `to_pretty_dicts`, `to_string`, `to_html`.

For instance, we can now calculate the proportion of rejected null hypotheses, using various significance levels (`alpha`). In A/A tests, it estimates the type I error rate.

```pycon
>>> def null_rejected(
...     results_data: pl.DataFrame,
...     alphas: tuple[float, ...] = (0.01, 0.02, 0.05),
... ) -> pl.DataFrame:
...     return results_data.group_by("metric", maintain_order=True).agg(
...         pl.col("pvalue").le(alpha).mean().alias(f"null_rejected_{alpha}")
...         for alpha in alphas
...     )
>>> null_rejected(results_data)
shape: (5, 4)
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ metric             ┆ null_rejected_0.01 ┆ null_rejected_0.02 ┆ null_rejected_0.05 │
│ ---                ┆ ---                ┆ ---                ┆ ---                │
│ str                ┆ f64                ┆ f64                ┆ f64                │
╞════════════════════╪════════════════════╪════════════════════╪════════════════════╡
│ sessions_per_user  ┆ 0.01               ┆ 0.02               ┆ 0.05               │
│ orders_per_session ┆ 0.02               ┆ 0.02               ┆ 0.06               │
│ orders_per_user    ┆ 0.01               ┆ 0.02               ┆ 0.05               │
│ revenue_per_user   ┆ 0.02               ┆ 0.03               ┆ 0.06               │
│ n_users            ┆ 0.01               ┆ 0.01               ┆ 0.04               │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘

```

100 simulations, as in the example above, produce a very rough estimate. In practice, a larger number of simulations, such as the default `10_000`, is recommended.

## Simulating experiments with treatment

To simulate experiments with treatment, define a treatment function that takes data in the form of a PyArrow Table and returns a PyArrow Table with the modified data:

```pycon
>>> import pyarrow as pa
>>> import pyarrow.compute as pc

>>> def treat(data: pa.Table) -> pa.Table:
...     return (
...         data.drop_columns(["orders", "revenue"])
...         .append_column("orders", pc.multiply(data["orders"], pa.scalar(1.1)))
...         .append_column("revenue", pc.multiply(data["revenue"], pa.scalar(1.1)))
...     )
>>> results = experiment.simulate(data, 100, seed=42, treat=treat)
>>> null_rejected(results.to_polars())
shape: (5, 4)
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ metric             ┆ null_rejected_0.01 ┆ null_rejected_0.02 ┆ null_rejected_0.05 │
│ ---                ┆ ---                ┆ ---                ┆ ---                │
│ str                ┆ f64                ┆ f64                ┆ f64                │
╞════════════════════╪════════════════════╪════════════════════╪════════════════════╡
│ sessions_per_user  ┆ 0.01               ┆ 0.02               ┆ 0.05               │
│ orders_per_session ┆ 0.23               ┆ 0.31               ┆ 0.42               │
│ orders_per_user    ┆ 0.21               ┆ 0.29               ┆ 0.4                │
│ revenue_per_user   ┆ 0.11               ┆ 0.16               ┆ 0.31               │
│ n_users            ┆ 0.01               ┆ 0.01               ┆ 0.04               │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘

```

In the example above, we've defined a function that increases the number of orders and the revenue by 10%. For these metrics, the proportion of rejected null hypotheses is an estimate of statistical power.

## Using a function instead of static data

You can use a function instead of static data to generate input dynamically. The function should take an instance of `numpy.random.Generator` as a parameter named `seed` and return experimental data in any format supported by **tea-tasting**.

As an example, let's use the `make_users_data` function.

```pycon
>>> results = experiment.simulate(tt.make_users_data, 100, seed=42)
>>> null_rejected(results.to_polars())
shape: (5, 4)
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ metric             ┆ null_rejected_0.01 ┆ null_rejected_0.02 ┆ null_rejected_0.05 │
│ ---                ┆ ---                ┆ ---                ┆ ---                │
│ str                ┆ f64                ┆ f64                ┆ f64                │
╞════════════════════╪════════════════════╪════════════════════╪════════════════════╡
│ sessions_per_user  ┆ 0.01               ┆ 0.01               ┆ 0.06               │
│ orders_per_session ┆ 0.27               ┆ 0.36               ┆ 0.54               │
│ orders_per_user    ┆ 0.24               ┆ 0.32               ┆ 0.49               │
│ revenue_per_user   ┆ 0.17               ┆ 0.26               ┆ 0.39               │
│ n_users            ┆ 0.01               ┆ 0.01               ┆ 0.04               │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘

```

On each iteration, **tea-tasting** calls `make_users_data` with a new `seed` and uses the returned data for the analysis of the experiment. The data returned by `make_users_data` already contains the `"variant"` column, so **tea-tasting** reuses that split. By default, `make_users_data` also adds the treatment uplift, and you can see it in the proportion of rejected null hypotheses.

## Tracking progress

To track the progress of simulations with [`tqdm.tqdm`](https://tqdm.github.io/) or [`marimo.status.progress_bar`](https://docs.marimo.io/api/status/#progress-bar), use the `progress` parameter.

```pycon
>>> import tqdm

>>> results = experiment.simulate(data, 100, seed=42, progress=tqdm.tqdm)  # doctest: +SKIP
100it [00:01, 73.19it/s]

```

## Parallel execution

To speed up simulations and run them in parallel, use the `map_` parameter with an alternative mapping function.

```pycon
>>> import concurrent.futures

>>> with concurrent.futures.ProcessPoolExecutor() as executor:
...     results = experiment.simulate(
...         data,
...         100,
...         seed=42,
...         treat=treat,
...         map_=executor.map,
...         progress=tqdm.tqdm,
...     )  # doctest: +SKIP
100it [00:00, 254.90it/s]

```

As an alternative to [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor), you can use the `map`, `imap`, or `imap_unordered` methods of [`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool).

It's also possible to run simulations on a distributed [Dask](https://distributed.dask.org/en/stable/api.html#distributed.Client.map) or [Ray](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.map.html#ray.util.ActorPool.map) cluster.
