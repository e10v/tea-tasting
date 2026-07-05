from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import duckdb
import ibis
import polars as pl
import pytest
import sqlframe.duckdb

import tea_tasting.backends.sql
import tea_tasting.data
import tea_tasting.datasets


if TYPE_CHECKING:
    import narwhals.typing
    import pandas as pd
    import pyarrow as pa


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, rng=42)


@pytest.fixture
def data_pandas(data_arrow: pa.Table) -> pd.DataFrame:
    return data_arrow.to_pandas()


@pytest.fixture
def data_duckdb(data_arrow: pa.Table) -> duckdb.DuckDBPyRelation:
    return duckdb.from_arrow(data_arrow)


@pytest.fixture
def data_polars(data_arrow: pa.Table) -> pl.DataFrame:
    return pl.from_arrow(data_arrow)  # ty:ignore[invalid-return-type]


@pytest.fixture
def data_polars_lazy(data_polars: pl.DataFrame) -> pl.LazyFrame:
    return data_polars.lazy()


@pytest.fixture
def data_sqlframe_duckdb(
    data_pandas: pd.DataFrame,
) -> sqlframe.duckdb.DuckDBDataFrame:
    return sqlframe.duckdb.DuckDBSession().createDataFrame(data_pandas)


@pytest.fixture
def data_ibis_duckdb(data_arrow: pa.Table) -> ibis.Table:
    return ibis.connect("duckdb://").create_table("data", data_arrow)


@pytest.fixture
def data_ibis_sqlite(data_arrow: pa.Table) -> ibis.Table:
    return ibis.connect("sqlite://").create_table("data", data_arrow)


@pytest.fixture
def data_sql_duckdb(data_arrow: pa.Table) -> tea_tasting.backends.sql.SQLQuery:
    conn = duckdb.connect()
    conn.register("data_arrow", data_arrow)
    conn.execute("CREATE TABLE data AS SELECT * FROM data_arrow")
    return tea_tasting.backends.sql.SQLQuery("SELECT * FROM data ORDER BY user", conn)


@pytest.fixture
def data_sql_sqlite(data_arrow: pa.Table) -> tea_tasting.backends.sql.SQLQuery:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE data ("
        "user INTEGER, variant INTEGER, sessions INTEGER, "
        "orders INTEGER, revenue REAL"
        ")",
    )
    conn.executemany(
        "INSERT INTO data VALUES (?, ?, ?, ?, ?)",
        (
            (
                row["user"],
                row["variant"],
                row["sessions"],
                row["orders"],
                row["revenue"],
            )
            for row in data_arrow.to_pylist()
        ),
    )
    return tea_tasting.backends.sql.SQLQuery("SELECT * FROM data ORDER BY user", conn)


@pytest.fixture(params=["data_ibis_duckdb", "data_ibis_sqlite"])
def data_ibis(request: pytest.FixtureRequest) -> ibis.Table:
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["data_sql_duckdb", "data_sql_sqlite"])
def data_sql(request: pytest.FixtureRequest) -> tea_tasting.backends.sql.SQLQuery:
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "data_arrow", "data_pandas", "data_duckdb",
    "data_polars", "data_polars_lazy",
    "data_sqlframe_duckdb",
])
def data_narwhals(request: pytest.FixtureRequest) -> narwhals.typing.IntoFrame:
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "data_arrow", "data_pandas", "data_duckdb",
    "data_polars", "data_polars_lazy",
    "data_sqlframe_duckdb",
    "data_ibis_duckdb", "data_ibis_sqlite",
    "data_sql_duckdb", "data_sql_sqlite",
])
def data(request: pytest.FixtureRequest) -> tea_tasting.data.Table:
    return request.getfixturevalue(request.param)
