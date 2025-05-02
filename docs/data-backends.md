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

???+ note

    This guide uses [DuckDB](https://github.com/duckdb/duckdb), an in-process analytical database, and [Polars](https://github.com/pola-rs/polars) as example data backends. Install these packages in addition to tea-tasting to reproduce the examples:

    ```bash
    pip install ibis-framework[duckdb] polars
    ```

First, let's prepare a demo database:

```pycon
>>> import ibis
>>> import polars as pl
>>> import tea_tasting as tt

>>> users_data = tt.make_users_data(seed=42)
>>> con = ibis.duckdb.connect()
>>> con.create_table("users_data", users_data)
DatabaseTable: memory.main.users_data
  user     int64
  variant  int64
  sessions int64
  orders   int64
  revenue  float64

```

In the example above:

- Function `tt.make_users_data` returns a PyArrow Table with example experimental data.
- Function `ibis.duckdb.connect` creates a DuckDB in-process database using Ibis API.
- Method `con.create_table` creates and populates a table in the database based on the PyArrow Table.

See the [Ibis documentation on how to create connections](https://ibis-project.org/reference/connection) to other data backends.

## Querying experimental data

Method `con.create_table` in the example above returns an Ibis Table which already can be used in the analysis of the experiment. But let's see how to use an SQL query to create an Ibis Table:

```pycon
>>> data = con.sql("select * from users_data")
>>> data
SQLQueryResult
  query:
    select * from users_data
  schema:
    user     int64
    variant  int64
    sessions int64
    orders   int64
    revenue  float64

```

It's a very simple query. In the real world, you might need to use joins, aggregations, and CTEs to get the data. You can define any SQL query supported by your data backend and use it to create Ibis Table.

Keep in mind that tea-tasting assumes that:

- Data is grouped by randomization units, such as individual users.
- There is a column indicating the variant of the A/B test (typically labeled as A, B, etc.).
- All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the table.

Ibis Table is a lazy object. It doesn't fetch the data when created. You can use Ibis DataFrame API to query the table and fetch the result:

```pycon
>>> ibis.options.interactive = True
>>> print(data.head(5))
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ user  ┃ variant ┃ sessions ┃ orders ┃ revenue ┃
┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ int64 │ int64   │ int64    │ int64  │ float64 │
├───────┼─────────┼──────────┼────────┼─────────┤
│     0 │       1 │        2 │      1 │    9.17 │
│     1 │       0 │        2 │      1 │    6.43 │
│     2 │       1 │        2 │      1 │    7.94 │
│     3 │       1 │        2 │      1 │   15.93 │
│     4 │       0 │        1 │      1 │    7.14 │
└───────┴─────────┴──────────┴────────┴─────────┘

>>> ibis.options.interactive = False

```

## Ibis example

To better understand what Ibis does, let's consider the example with grouping and aggregation by variants:

```pycon
>>> aggr_data = data.group_by("variant").aggregate(
...     sessions_per_user=data.sessions.mean(),
...     orders_per_session=data.orders.mean() / data.sessions.mean(),
...     orders_per_user=data.orders.mean(),
...     revenue_per_user=data.revenue.mean(),
... )
>>> aggr_data
r0 := SQLQueryResult
  query:
    select * from users_data
  schema:
    user     int64
    variant  int64
    sessions int64
    orders   int64
    revenue  float64
<BLANKLINE>
Aggregate[r0]
  groups:
    variant: r0.variant
  metrics:
    sessions_per_user:  Mean(r0.sessions)
    orders_per_session: Mean(r0.orders) / Mean(r0.sessions)
    orders_per_user:    Mean(r0.orders)
    revenue_per_user:   Mean(r0.revenue)

```

`aggr_data` is another Ibis Table defined as a query over the previously defined `data`. Let's fetch the result:

```pycon
>>> ibis.options.interactive = True
>>> print(aggr_data)  # doctest: +SKIP
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ variant ┃ sessions_per_user ┃ orders_per_session ┃ orders_per_user ┃ revenue_per_user ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ int64   │ float64           │ float64            │ float64         │ float64          │
├─────────┼───────────────────┼────────────────────┼─────────────────┼──────────────────┤
│       0 │          1.996045 │           0.265726 │        0.530400 │         5.241028 │
│       1 │          1.982802 │           0.289031 │        0.573091 │         5.730111 │
└─────────┴───────────────────┴────────────────────┴─────────────────┴──────────────────┘

>>> ibis.options.interactive = False

```

Internally, Ibis compiles a Table to an SQL query supported by the backend:

```pycon
>>> print(aggr_data.compile(pretty=True))
SELECT
  "t0"."variant",
  AVG("t0"."sessions") AS "sessions_per_user",
  AVG("t0"."orders") / AVG("t0"."sessions") AS "orders_per_session",
  AVG("t0"."orders") AS "orders_per_user",
  AVG("t0"."revenue") AS "revenue_per_user"
FROM (
  SELECT
    *
  FROM users_data
) AS "t0"
GROUP BY
  1

```

See [Ibis documentation](https://ibis-project.org/tutorials/getting_started) for more details.

## Experiment analysis

The example above shows how to query the metric averages. But for statistical inference, it's not enough. For example, Student's t-test and Z-test also require number of rows and variance. Additionally, analysis of ratio metrics and variance reduction with CUPED requires covariances.

Querying all the required statistics manually can be a daunting and error-prone task. But don't worry—tea-tasting does this work for you. You just need to specify the metrics:

```pycon
>>> experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
... )
>>> result = experiment.analyze(data)
>>> result
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```

In the example above, tea-tasting fetches all the required statistics with a single query and then uses them to analyze the experiment.

Some statistical methods, like bootstrap, require granular data for analysis. In this case, tea-tasting fetches the detailed data as well.

## Example with CUPED

An example of a slightly more complicated analysis using variance reduction with CUPED:

```pycon
>>> users_data_cuped = tt.make_users_data(seed=42, covariates=True)
>>> con.create_table("users_data_cuped", users_data_cuped)
DatabaseTable: memory.main.users_data_cuped
  user               int64
  variant            int64
  sessions           int64
  orders             int64
  revenue            float64
  sessions_covariate int64
  orders_covariate   int64
  revenue_covariate  float64

>>> data_cuped = con.sql("select * from users_data_cuped")
>>> experiment_cuped = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions", "sessions_covariate"),
...     orders_per_session=tt.RatioOfMeans(
...         numer="orders",
...         denom="sessions",
...         numer_covariate="orders_covariate",
...         denom_covariate="sessions_covariate",
...     ),
...     orders_per_user=tt.Mean("orders", "orders_covariate"),
...     revenue_per_user=tt.Mean("revenue", "revenue_covariate"),
... )
>>> result_cuped = experiment_cuped.analyze(data_cuped)
>>> result_cuped
            metric control treatment rel_effect_size rel_effect_size_ci  pvalue
 sessions_per_user    2.00      1.98          -0.68%      [-3.2%, 1.9%]   0.603
orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229
   orders_per_user   0.523     0.581             11%        [2.9%, 20%] 0.00733
  revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00674

```

## Polars example

Here’s an example of how to analyze data using a Polars DataFrame:

```pycon
>>> data_polars = pl.from_arrow(users_data)
>>> experiment.analyze(data_polars)
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```
