# Data backends

## Intro

**tea-tasting** supports a wide range of data backends such as BigQuery, ClickHouse, PostgreSQL/GreenPlum, Snowflake, Spark, and 20+ other backends supported by [Ibis](https://ibis-project.org/). Ibis is a Python package that serves as a DataFrame API to various data backends.

Many statistical tests, such as the Student's t-test or the Z-test, require only aggregated data for analysis. For these tests, **tea-tasting** retrieves only aggregated statistics like mean and variance instead of downloading all detailed data.

For example, if the raw experimental data are stored in ClickHouse, it's faster and more efficient to calculate counts, averages, variances, and covariances directly in ClickHouse rather than fetching granular data and performing aggregations in a Python environment.

This guide:

- Shows how to use **tea-tasting** with a data backend of your choice for the analysis of an experiment.
- Explains some internals of how **tea-tasting** uses Ibis to work with data backends.

## Demo database

This guide uses [DuckDB](https://duckdb.org/), an in-process analytical database, as an example data backend. To be able to reproduce the example code, install both **tea-tasting** and Ibis with DuckDB extra:

```bash
pip install tea-tasting ibis-framework[duckdb]
```

First, let's prepare a demo database:

```python
import ibis
import tea_tasting as tt


users_data = tt.make_users_data(seed=42)
con = ibis.duckdb.connect()
con.create_table("users_data", users_data)
#> DatabaseTable: memory.main.users_data
#>   user     int64
#>   variant  uint8
#>   sessions int64
#>   orders   int64
#>   revenue  float64
```

In the example above:

- Function `tt.make_users_data` returns a Pandas DataFrame with example experimental data.
- Function `ibis.duckdb.connect` creates a DuckDB in-process database using Ibis API.
- Method `con.create_table` creates and populates a table in the database based on the DataFrame.

See the [Ibis documentation on how to create connections](https://ibis-project.org/reference/connection) to other data backends.

## Querying experimental data

Method `con.create_table` in the example above returns an instance of Ibis Table which already can be used in the analysis of the experiment. But let's see how to use an SQL query to create Ibis Table:

```python
data = con.sql("select * from users_data")
print(data)
#> SQLQueryResult
#>   query:
#>     select * from users_data
#>   schema:
#>     user     int64
#>     variant  uint8
#>     sessions int64
#>     orders   int64
#>     revenue  float64
```

It's a very simple query. In real world, you might need to use joins, aggregations, and CTEs to get the data. You can define any SQL query supported by your data backend and use it to create Ibis Table.

Keep in mind that **tea-tasting** assumes that:

- Data is grouped by randomization units, such as individual users.
- There is a column indicating variant of the A/B test (typically labeled as A, B, etc.).
- All necessary columns for metric calculations (like the number of orders, revenue, etc.) are included in the table.

Ibis Table is a lazy object. It doesn't fetch the data when created. You can use Ibis DataFrame API to query the table and fetch the result:

```python
print(data.head(5).to_pandas())
#>    user  variant  sessions  orders    revenue
#> 0     0        1         2       1   9.166147
#> 1     1        0         2       1   6.434079
#> 2     2        1         2       1   7.943873
#> 3     3        1         2       1  15.928675
#> 4     4        0         1       1   7.136917
```

## Ibis example

To better understand what Ibis does, let's consider the following example:

```python
aggr_data = data.group_by("variant").aggregate(
    sessions_per_user=data.sessions.mean(),
    orders_per_session=data.orders.mean() / data.sessions.mean(),
    orders_per_user=data.orders.mean(),
    revenue_per_user=data.revenue.mean(),
)
print(aggr_data)
#> r0 := SQLQueryResult
#>   query:
#>     select * from users_data
#>   schema:
#>     user     int64
#>     variant  uint8
#>     sessions int64
#>     orders   int64
#>     revenue  float64
#>
#> Aggregate[r0]
#>   groups:
#>     variant: r0.variant
#>   metrics:
#>     sessions_per_user:  Mean(r0.sessions)
#>     orders_per_session: Mean(r0.orders) / Mean(r0.sessions)
#>     orders_per_user:    Mean(r0.orders)
#>     revenue_per_user:   Mean(r0.revenue)
```

`aggr_data` is another Ibis Table defined as a query over the previously defined `data`. Let's fetch the result:

```python
print(aggr_data.to_pandas())
#>    variant  sessions_per_user  orders_per_session  orders_per_user  revenue_per_user
#> 0        0           1.996045            0.265726         0.530400          5.241079
#> 1        1           1.982802            0.289031         0.573091          5.730132
```

Internally, Ibis compiles a Table to an SQL query supported by the backend:

```python
print(aggr_data.compile(pretty=True))
#> SELECT
#>   "t0"."variant",
#>   AVG("t0"."sessions") AS "sessions_per_user",
#>   AVG("t0"."orders") / AVG("t0"."sessions") AS "orders_per_session",
#>   AVG("t0"."orders") AS "orders_per_user",
#>   AVG("t0"."revenue") AS "revenue_per_user"
#> FROM (
#>   SELECT
#>     *
#>   FROM users_data
#> ) AS "t0"
#> GROUP BY
#>   1
```

See [Ibis documentation](https://ibis-project.org/tutorials/getting_started) for more details.

## Experiment analysis

The example above shows how to query the metric averages. But for statistical inference it's not enough. For example, Student's t-test and Z-test also require number of rows and variance. And analysis of ratio metrics and variance reduction with CUPED require covariances.

Querying all the required statistics manually can be a daunting and error-prone task. But don't worry—**tea-tasting** does this work for you. You just need to specify the metrics:

```python
experiment = tt.Experiment(
    sessions_per_user=tt.Mean("sessions"),
    orders_per_session=tt.RatioOfMeans("orders", "sessions"),
    orders_per_user=tt.Mean("orders"),
    revenue_per_user=tt.Mean("revenue"),
)
result = experiment.analyze(data)
print(result)
#>             metric control treatment rel_effect_size rel_effect_size_ci pvalue
#>  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
#> orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
#>    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
#>   revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123
```

In the example above, **tea-tasting** fetches all the required statistics with a single query and then uses them to analyse the experiment.

Some statistical methods, like Bootstrap, require granular data for the analysis. In this case, **tea-tasting** fetches the detailed data as well.

## Example with CUPED

An example of a slightly more complicated analysis using variance reduction with CUPED:

```python
users_data_with_cov = tt.make_users_data(seed=42, covariates=True)
con.create_table("users_data_with_cov", users_data_with_cov)
#> DatabaseTable: memory.main.users_data_with_cov
#>   user               int64
#>   variant            uint8
#>   sessions           int64
#>   orders             int64
#>   revenue            float64
#>   sessions_covariate int64
#>   orders_covariate   int64
#>   revenue_covariate  float64

data_with_cov = con.sql("select * from users_data_with_cov")
experiment_with_cov = tt.Experiment(
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
result_with_cov = experiment_with_cov.analyze(data_with_cov)
print(result_with_cov)
#>             metric control treatment rel_effect_size rel_effect_size_ci  pvalue
#>  sessions_per_user    2.00      1.98          -0.68%      [-3.2%, 1.9%]   0.603
#> orders_per_session   0.262     0.293             12%        [4.2%, 21%] 0.00229
#>    orders_per_user   0.523     0.581             11%        [2.9%, 20%] 0.00733
#>   revenue_per_user    5.12      5.85             14%        [3.8%, 26%] 0.00675
```
