from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting
import tea_tasting.aggr
import tea_tasting.backends
import tea_tasting.backends._executor
import tea_tasting.backends.sql


if TYPE_CHECKING:
    from collections.abc import Hashable


pytest_plugins = ("tests.fixtures",)


def _expected_aggr(data: pa.Table) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=data.num_rows,
        mean_={
            "sessions": pc.mean(data["sessions"]).as_py(),
            "orders": pc.mean(data["orders"]).as_py(),
        },
        var_={
            "sessions": pc.variance(data["sessions"], ddof=1).as_py(),
            "orders": pc.variance(data["orders"], ddof=1).as_py(),
        },
        cov_={
            ("orders", "sessions"): np.cov(
                data["sessions"].combine_chunks().to_numpy(zero_copy_only=False),
                data["orders"].combine_chunks().to_numpy(zero_copy_only=False),
                ddof=1,
            )[0, 1],
        },
    )


def _expected_aggrs(data: pa.Table) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    variant_col = data["variant"]
    return {
        variant: _expected_aggr(
            data.filter(pc.equal(variant_col, pa.scalar(variant))),
        )
        for variant in variant_col.unique().to_pylist()
    }


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == pytest.approx(right.count_)
    assert left.mean_ == pytest.approx(right.mean_, nan_ok=True)
    assert left.var_ == pytest.approx(right.var_, nan_ok=True)
    assert left.cov_ == pytest.approx(right.cov_, nan_ok=True)


def _duckdb_query(
    data: pa.Table,
    **kwargs: Any,
) -> tea_tasting.backends.sql.SQLQuery:
    conn = duckdb.connect()
    conn.register("data_arrow", data)
    conn.execute("CREATE TABLE data AS SELECT * FROM data_arrow")
    return tea_tasting.backends.sql.SQLQuery("SELECT * FROM data", conn, **kwargs)


def _null_data_query(
    *,
    var: bool | str | None,
    cov: bool | str | None,
) -> tea_tasting.backends.sql.SQLQuery:
    data = pa.table({
        "variant": pa.array([0, 0, 0, 1, 1, 1], type=pa.int64()),
        "x": pa.array([1, None, 3, None, 5, None], type=pa.float64()),
        "y": pa.array([2, 4, None, 8, None, None], type=pa.float64()),
        "z": pa.array([10, 20, 30, None, 50, 60], type=pa.float64()),
        "empty": pa.array(
            [None, None, None, None, None, None],
            type=pa.float64(),
        ),
    })
    return _duckdb_query(data, var=var, cov=cov)


def _expected_null_aggr() -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=6,
        mean_={
            "x": 3.0,
            "y": 14 / 3,
            "z": 34.0,
            "empty": np.nan,
        },
        var_={
            "x": 4.0,
            "y": 28 / 3,
            "z": 430.0,
            "empty": np.nan,
        },
        cov_={
            ("x", "y"): np.nan,
            ("x", "z"): 40.0,
            ("empty", "x"): np.nan,
        },
    )


def _expected_null_aggrs() -> dict[int, tea_tasting.aggr.Aggregates]:
    return {
        0: tea_tasting.aggr.Aggregates(
            count_=3,
            mean_={
                "x": 2.0,
                "y": 3.0,
                "z": 20.0,
                "empty": np.nan,
            },
            var_={
                "x": 2.0,
                "y": 2.0,
                "z": 100.0,
                "empty": np.nan,
            },
            cov_={
                ("x", "y"): np.nan,
                ("x", "z"): 20.0,
                ("empty", "x"): np.nan,
            },
        ),
        1: tea_tasting.aggr.Aggregates(
            count_=3,
            mean_={
                "x": 5.0,
                "y": 8.0,
                "z": 55.0,
                "empty": np.nan,
            },
            var_={
                "x": np.nan,
                "y": np.nan,
                "z": 50.0,
                "empty": np.nan,
            },
            cov_={
                ("x", "y"): np.nan,
                ("x", "z"): np.nan,
                ("empty", "x"): np.nan,
            },
        ),
    }


def test_sql_query_init(data_sql_duckdb: tea_tasting.backends.sql.SQLQuery) -> None:
    assert data_sql_duckdb.query.sql(dialect="duckdb") == (
        'SELECT * FROM data ORDER BY "user"'
    )
    assert data_sql_duckdb.dialect == "duckdb"
    assert data_sql_duckdb.var is True
    assert data_sql_duckdb.cov is True
    assert data_sql_duckdb.chunk_size == 100_000


def test_sql_query_init_sqlglot_query(data_arrow: pa.Table) -> None:
    import sqlglot  # noqa: PLC0415

    conn = duckdb.connect()
    conn.register("data_arrow", data_arrow)
    conn.execute("CREATE TABLE data AS SELECT * FROM data_arrow")
    query = sqlglot.parse_one("SELECT * FROM data")

    adapter = tea_tasting.backends.sql.SQLQuery(query, conn, dialect="duckdb")

    assert adapter.query is not query
    assert isinstance(adapter.query, sqlglot.exp.Query)
    assert adapter.query.sql(dialect="duckdb") == "SELECT * FROM data"
    assert adapter.select("sessions").column_names == ["sessions"]


def test_sql_query_init_overrides(data_arrow: pa.Table) -> None:
    query = _duckdb_query(
        data_arrow,
        dialect="duckdb",
        var="VAR_SAMP",
        cov=False,
        chunk_size=None,
    )
    assert query.dialect == "duckdb"
    assert query.var == "VAR_SAMP"
    assert query.cov is False
    assert query.chunk_size is None


def test_sql_query_infers_dialect() -> None:
    class DuckDBConnection:
        pass

    class ChDBConnection:
        pass

    class MssqlConnection:
        pass

    class UnknownConnection:
        pass

    DuckDBConnection.__module__ = "duckdb"
    ChDBConnection.__module__ = "chdb"
    MssqlConnection.__module__ = "pymssql"
    UnknownConnection.__module__ = "package.driver"

    def dialect(connection: Any) -> str:
        return tea_tasting.backends.sql.SQLQuery("SELECT 1", connection).dialect

    assert dialect(DuckDBConnection()) == "duckdb"
    assert dialect(ChDBConnection()) == "clickhouse"
    assert dialect(MssqlConnection()) == "tsql"
    assert dialect(UnknownConnection()) == "postgres"


def test_sql_query_select(data_sql_duckdb: tea_tasting.backends.sql.SQLQuery) -> None:
    selected = data_sql_duckdb.select("sessions", "orders")
    assert selected.column_names == ["sessions", "orders"]


def test_sql_query_select_all(
    data_sql_duckdb: tea_tasting.backends.sql.SQLQuery,
    data_arrow: pa.Table,
) -> None:
    assert data_sql_duckdb.select().equals(data_arrow)


def test_sql_query_select_all_without_chunks(data_arrow: pa.Table) -> None:
    assert _duckdb_query(data_arrow, chunk_size=None).select().equals(data_arrow)


def test_sql_query_select_col_unique(
    data_sql: tea_tasting.backends.sql.SQLQuery,
) -> None:
    assert set(data_sql.select_col_unique("variant")) == {0, 1}


def test_sql_query_group_by(data_sql_duckdb: tea_tasting.backends.sql.SQLQuery) -> None:
    grouped = data_sql_duckdb.group_by("variant")
    assert isinstance(grouped, tea_tasting.backends.sql.SQLQueryGroupBy)
    assert grouped.sql_query is data_sql_duckdb
    assert grouped.by == "variant"


def test_sql_query_aggregate(
    data_sql: tea_tasting.backends.sql.SQLQuery,
    data_arrow: pa.Table,
) -> None:
    aggr = data_sql.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions", "orders"),
            cov_cols=(("orders", "sessions"),),
        ),
    )
    _compare_aggrs(aggr, _expected_aggr(data_arrow))


def test_sql_query_aggregate_no_count(
    data_sql_duckdb: tea_tasting.backends.sql.SQLQuery,
) -> None:
    aggr = data_sql_duckdb.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=("sessions",),
            var_cols=(),
            cov_cols=(),
        ),
    )
    assert aggr.count_ is None
    assert set(aggr.mean_) == {"sessions"}
    assert aggr.var_ == {}
    assert aggr.cov_ == {}


def test_sql_query_aggregate_function_overrides(data_arrow: pa.Table) -> None:
    aggr = _duckdb_query(
        data_arrow,
        var="VAR_SAMP",
        cov="COVAR_SAMP",
    ).aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions", "orders"),
            cov_cols=(("orders", "sessions"),),
        ),
    )
    _compare_aggrs(aggr, _expected_aggr(data_arrow))


def test_sql_query_group_by_init(
    data_sql_duckdb: tea_tasting.backends.sql.SQLQuery,
) -> None:
    grouped = tea_tasting.backends.sql.SQLQueryGroupBy(
        data_sql_duckdb,
        "variant",
    )
    assert grouped.sql_query is data_sql_duckdb
    assert grouped.by == "variant"


def test_sql_query_group_by_aggregate(
    data_sql: tea_tasting.backends.sql.SQLQuery,
    data_arrow: pa.Table,
) -> None:
    aggrs = data_sql.group_by("variant").aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions", "orders"),
            cov_cols=(("orders", "sessions"),),
        ),
    )
    expected = _expected_aggrs(data_arrow)
    assert set(aggrs) == {0, 1}
    for variant, expected_aggr in expected.items():
        _compare_aggrs(aggrs[variant], expected_aggr)


@pytest.mark.parametrize(
    ("var", "cov"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_sql_query_aggregate_nulls(
    var: bool,  # noqa: FBT001
    cov: bool,  # noqa: FBT001
) -> None:
    aggr = _null_data_query(var=var, cov=cov).aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("x", "y", "z", "empty"),
            var_cols=("x", "y", "z", "empty"),
            cov_cols=(("x", "y"), ("x", "z"), ("empty", "x")),
        ),
    )

    _compare_aggrs(aggr, _expected_null_aggr())


@pytest.mark.parametrize(
    ("var", "cov"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_sql_query_group_by_aggregate_nulls(
    var: bool,  # noqa: FBT001
    cov: bool,  # noqa: FBT001
) -> None:
    aggrs = _null_data_query(var=var, cov=cov).group_by("variant").aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("x", "y", "z", "empty"),
            var_cols=("x", "y", "z", "empty"),
            cov_cols=(("x", "y"), ("x", "z"), ("empty", "x")),
        ),
    )

    expected = _expected_null_aggrs()
    assert set(aggrs) == {0, 1}
    for variant, expected_aggr in expected.items():
        _compare_aggrs(aggrs[variant], expected_aggr)


def test_sql_query_sqlite_uses_fallback() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE data (x INTEGER, y INTEGER)")
    conn.executemany("INSERT INTO data VALUES (?, ?)", [(1, 2), (2, 4), (3, 6)])
    query = tea_tasting.backends.sql.SQLQuery("SELECT * FROM data", conn)

    assert query.dialect == "sqlite"
    assert query.var is False
    assert query.cov is False
    aggr = query.aggregate(
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=(),
            var_cols=("x",),
            cov_cols=(("x", "y"),),
        ),
    )
    assert aggr.var_ == pytest.approx({"x": 1.0})
    assert aggr.cov_ == pytest.approx({("x", "y"): 2.0})
