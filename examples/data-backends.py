# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ibis-framework[duckdb]",
#     "marimo",
#     "narwhals>=1.25",
#     "polars",
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
        # Data backends

        ## Intro

        tea-tasting supports a wide range of data backends such as BigQuery, ClickHouse, DuckDB, PostgreSQL, Snowflake, Spark, and many other backends supported by [Ibis](https://github.com/ibis-project/ibis). Ibis is a DataFrame API to various data backends.

        Many statistical tests, such as the Student's t-test or the Z-test, require only aggregated data for analysis. For these tests, tea-tasting retrieves only aggregated statistics like mean and variance instead of downloading all detailed data.

        For example, if the raw experimental data are stored in ClickHouse, it's faster and more efficient to calculate counts, averages, variances, and covariances directly in ClickHouse rather than fetching granular data and performing aggregations in a Python environment.

        tea-tasting also accepts dataframes supported by [Narwhals](https://github.com/narwhals-dev/narwhals): cuDF, Dask, Modin, pandas, Polars, PyArrow. Narwhals is a compatibility layer between dataframe libraries.

        This guide:

        - Shows how to use tea-tasting with a data backend of your choice for the analysis of an experiment.
        - Explains some internals of how tea-tasting uses Ibis to work with data backends.

        ## Demo database

        /// admonition | Note

        This guide uses [DuckDB](https://github.com/duckdb/duckdb), an in-process analytical database, and [Polars](https://github.com/pola-rs/polars) as example data backends. Install these packages in addition to tea-tasting to reproduce the examples:

        ```bash
        pip install ibis-framework[duckdb] polars
        ```

        ///

        First, let's prepare a demo database:
        """
    )
    return


@app.cell
def _():
    import ibis
    import polars as pl
    import tea_tasting as tt

    users_data = tt.make_users_data(seed=42)
    con = ibis.duckdb.connect()
    con.create_table("users_data", users_data)
    return con, ibis, pl, tt, users_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the example above:

        - Function `tt.make_users_data` returns a PyArrow Table with example experimental data.
        - Function `ibis.duckdb.connect` creates a DuckDB in-process database using Ibis API.
        - Method `con.create_table` creates and populates a table in the database based on the PyArrow Table.

        See the [Ibis documentation on how to create connections](https://ibis-project.org/reference/connection) to other data backends.

        ## Querying experimental data

        Method `con.create_table` in the example above returns an Ibis Table which already can be used in the analysis of the experiment. But let's see how to use an SQL query to create an Ibis Table:
        """
    )
    return


@app.cell
def _(con):
    data = con.sql("select * from users_data")
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It's a very simple query. In the real world, you might need to use joins, aggregations, and CTEs to get the data. You can define any SQL query supported by your data backend and use it to create Ibis Table.

        Keep in mind that tea-tasting assumes that:

        - Data is grouped by randomization units, such as individual users.
        - There is a column indicating the variant of the A/B test (typically labeled as A, B, etc.).
        - All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the table.

        Ibis Table is a lazy object. It doesn't fetch the data when created. You can use Ibis DataFrame API to query the table and fetch the result:
        """
    )
    return


@app.cell
def _(data, ibis):
    ibis.options.interactive = True
    print(data.head(5))

    ibis.options.interactive = False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ibis example

        To better understand what Ibis does, let's consider the example with grouping and aggregation by variants:
        """
    )
    return


@app.cell
def _(data):
    aggr_data = data.group_by("variant").aggregate(
        sessions_per_user=data.sessions.mean(),
        orders_per_session=data.orders.mean() / data.sessions.mean(),
        orders_per_user=data.orders.mean(),
        revenue_per_user=data.revenue.mean(),
    )
    aggr_data
    return (aggr_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `aggr_data` is another Ibis Table defined as a query over the previously defined `data`. Let's fetch the result:
        """
    )
    return


@app.cell
def _(aggr_data, ibis):
    ibis.options.interactive = True
    print(aggr_data)

    ibis.options.interactive = False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Internally, Ibis compiles a Table to an SQL query supported by the backend:
        """
    )
    return


@app.cell
def _(aggr_data):
    print(aggr_data.compile(pretty=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        See [Ibis documentation](https://ibis-project.org/tutorials/getting_started) for more details.

        ## Experiment analysis

        The example above shows how to query the metric averages. But for statistical inference, it's not enough. For example, Student's t-test and Z-test also require number of rows and variance. Additionally, analysis of ratio metrics and variance reduction with CUPED requires covariances.

        Querying all the required statistics manually can be a daunting and error-prone task. But don't worry—tea-tasting does this work for you. You just need to specify the metrics:
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
    )
    result = experiment.analyze(data)
    result
    return (experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the example above, tea-tasting fetches all the required statistics with a single query and then uses them to analyze the experiment.

        Some statistical methods, like bootstrap, require granular data for analysis. In this case, tea-tasting fetches the detailed data as well.

        ## Example with CUPED

        An example of a slightly more complicated analysis using variance reduction with CUPED:
        """
    )
    return


@app.cell
def _(con, tt):
    users_data_cuped = tt.make_users_data(seed=42, covariates=True)
    con.create_table("users_data_cuped", users_data_cuped)

    data_cuped = con.sql("select * from users_data_cuped")
    experiment_cuped = tt.Experiment(
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
    result_cuped = experiment_cuped.analyze(data_cuped)
    result_cuped
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Polars example

        Here’s an example of how to analyze data using a Polars DataFrame:
        """
    )
    return


@app.cell
def _(experiment, pl, users_data):
    data_polars = pl.from_arrow(users_data)
    experiment.analyze(data_polars)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
