"""SQL data backend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import tea_tasting.backends._executor
from tea_tasting.backends.base import (
    _COUNT,
    _COV,
    _MEAN,
    _VAR,
    BaseTable,
    BaseTableGroupBy,
    _get_aggregates,
)
import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Hashable

    import pyarrow as pa
    import sqlglot

    import tea_tasting.aggr  # noqa: TC004


    type Dialect = Literal[
        "athena",
        "bigquery",
        "clickhouse",
        "databricks",
        "doris",
        "dremio",
        "drill",
        "druid",
        "duckdb",
        "dune",
        "exasol",
        "fabric",
        "hive",
        "materialize",
        "mysql",
        "oracle",
        "postgres",
        "presto",
        "prql",
        "redshift",
        "risingwave",
        "singlestore",
        "snowflake",
        "solr",
        "spark",
        "spark2",
        "sqlite",
        "starrocks",
        "tableau",
        "teradata",
        "trino",
        "tsql",
    ]


DIALECT_VAR: dict[Dialect, bool | str] = {
    "athena": True,
    "bigquery": True,
    "clickhouse": True,
    "databricks": True,
    "doris": "VAR_SAMP",
    "dremio": "VAR_SAMP",
    "drill": True,
    "druid": True,
    "duckdb": True,
    "dune": True,
    "exasol": True,
    "fabric": "VAR",
    "hive": "VAR_SAMP",
    "materialize": True,
    "mysql": "VAR_SAMP",
    "oracle": True,
    "postgres": True,
    "presto": True,
    "prql": False,
    "redshift": True,
    "risingwave": True,
    "singlestore": True,
    "snowflake": True,
    "solr": False,
    "spark": True,
    "spark2": True,
    "sqlite": False,
    "starrocks": "VAR_SAMP",
    "tableau": "VAR",
    "teradata": "VAR_SAMP",
    "trino": True,
    "tsql": "VAR",
}
DIALECT_COV: dict[Dialect, bool | str] = {
    "athena": True,
    "bigquery": True,
    "clickhouse": "covarSamp",
    "databricks": True,
    "doris": True,
    "dremio": True,
    "drill": True,
    "druid": False,
    "duckdb": True,
    "dune": True,
    "exasol": True,
    "fabric": False,
    "hive": True,
    "materialize": False,
    "mysql": False,
    "oracle": True,
    "postgres": True,
    "presto": True,
    "prql": False,
    "redshift": False,
    "risingwave": False,
    "singlestore": False,
    "snowflake": True,
    "solr": False,
    "spark": True,
    "spark2": True,
    "sqlite": False,
    "starrocks": True,
    "tableau": "COVAR",
    "teradata": True,
    "trino": True,
    "tsql": False,
}

_SUBQUERY = "__subquery__"
_CENTERED = "__centered__{}__"
_CENTERED_LEFT = "__centered__left__{}__{}__"
_CENTERED_RIGHT = "__centered_right__{}__{}__"


class SQLQuery(BaseTable):  # noqa: D101
    def __init__(
        self,
        query: str | sqlglot.exp.Query,
        connection: (
            tea_tasting.backends._executor.Connection |
            tea_tasting.backends._executor.Cursor
        ),
        dialect: Dialect | None = None,
        *,
        var: bool | str | None = None,
        cov: bool | str | None = None,
        chunk_size: int | None = 100_000,
    ) -> None:
        """SQL query adapter.

        Args:
            query: SQL query string or SQLGlot query expression.
            connection: PEP 249-compatible DB-API connection or cursor.
            dialect: SQLGlot dialect string. If `None`, infer it from
                `connection`.
            var: Sample variance support. If `True`, use SQLGlot variance.
                If `False`, use a fallback expression. If a string, call a
                function with that name. If `None`, infer from `dialect`.
            cov: Sample covariance support. If `True`, use SQLGlot covariance.
                If `False`, use a fallback expression. If a string, call a
                function with that name. If `None`, infer from `dialect`.
            chunk_size: Chunk size for fetching data. Used only in the `select`
                method when the cursor does not fetch Arrow directly. If `None`,
                fetch all rows.
        """
        import sqlglot  # noqa: PLC0415

        tea_tasting.utils.check_scalar(query, "query", typ=str | sqlglot.exp.Query)
        tea_tasting.utils.check_scalar(var, "var", typ=bool | str | None)
        tea_tasting.utils.check_scalar(cov, "cov", typ=bool | str | None)
        if chunk_size is not None:
            tea_tasting.utils.check_scalar(chunk_size, "chunk_size", typ=int, gt=0)

        self.connection = connection
        self.dialect = (
            _infer_dialect(connection) if dialect is None else
            tea_tasting.utils.check_scalar(dialect, "dialect", typ=str, in_=DIALECT_VAR)
        )
        self.var = DIALECT_VAR.get(self.dialect, True) if var is None else var
        self.cov = DIALECT_COV.get(self.dialect, True) if cov is None else cov
        self.chunk_size = chunk_size
        self.query = (
            sqlglot.parse_one(query, dialect=self.dialect)
            if isinstance(query, str) else query.copy()
        )

    def select(self, *cols: str) -> pa.Table:
        """Select columns and return data as a PyArrow Table.

        Args:
            *cols: Column names. If empty, select all columns.

        Returns:
            Selected data.
        """
        import sqlglot  # noqa: PLC0415

        exprs = [sqlglot.exp.Star()] if len(cols) == 0 else [_col(col) for col in cols]
        query = (
            sqlglot.exp.select(*exprs)
            .from_(self.query.copy().subquery(_SUBQUERY))
            .sql(dialect=self.dialect)
        )
        with tea_tasting.backends._executor.Executor(self.connection) as executor:
            return executor.execute(query).to_arrow(self.chunk_size)

    def select_col_unique(self, col: str) -> list[Hashable]:
        """Select unique column values.

        Args:
            col: Column name.

        Returns:
            Unique column values.
        """
        import sqlglot  # noqa: PLC0415

        query = (
            sqlglot.exp.select(sqlglot.exp.Distinct(expressions=[_col(col)]))
            .from_(self.query.copy().subquery(_SUBQUERY))
            .sql(dialect=self.dialect)
        )
        with tea_tasting.backends._executor.Executor(self.connection) as executor:
            return executor.execute(query).to_list()

    def group_by(self, by: str) -> SQLQueryGroupBy:
        """Group table by a column.

        Args:
            by: Column name to group by.

        Returns:
            Grouped SQL query adapter.
        """
        return SQLQueryGroupBy(self, by)

    def aggregate(
        self,
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> tea_tasting.aggr.Aggregates:
        """Aggregate table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics.
        """
        return _get_aggregates(_aggregate(self, aggr_cols, None)[0], aggr_cols)


class SQLQueryGroupBy(BaseTableGroupBy):  # noqa: D101
    def __init__(
        self,
        sql_query: SQLQuery,
        by: str,
    ) -> None:
        """Grouped SQL query adapter.

        Args:
            sql_query: SQL query adapter.
            by: Column name to group by.
        """
        self.sql_query = sql_query
        self.by = by

    def aggregate(
        self,
        aggr_cols: tea_tasting.aggr.AggrCols,
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        """Aggregate grouped table data.

        Args:
            aggr_cols: Columns to be aggregated.

        Returns:
            Aggregated statistics by group value.
        """
        return {
            group_data[self.by]: _get_aggregates(group_data, aggr_cols)
            for group_data in _aggregate(self.sql_query, aggr_cols, self.by)
        }


def _infer_dialect(connection: object) -> Dialect:
    driver_name = connection.__class__.__module__.split(".", maxsplit=1)[0].lower()
    for dialect in sorted(DIALECT_VAR):
        if dialect in driver_name:
            return dialect
    if driver_name == "chdb":
        return "clickhouse"
    if "mssql" in driver_name:
        return "tsql"
    return "postgres"


def _aggregate(
    sql_query: SQLQuery,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
) -> list[dict[str, int | float]]:
    import sqlglot  # noqa: PLC0415

    query = sql_query.query.copy().subquery(_SUBQUERY)
    fallback_var_cols = () if sql_query.var else aggr_cols.var_cols
    fallback_cov_cols = () if sql_query.cov else aggr_cols.cov_cols
    if len(fallback_var_cols) > 0 or len(fallback_cov_cols) > 0:
        query = _add_centered_cols(
            query=query,
            aggr_cols=aggr_cols,
            group_col=group_col,
            var_cols=fallback_var_cols,
            cov_cols=fallback_cov_cols,
        )

    exprs = _aggr_exprs(sql_query, aggr_cols, group_col)
    query = sqlglot.exp.select(*exprs).from_(query)
    if group_col is not None:
        query = query.group_by(_col(group_col))

    with tea_tasting.backends._executor.Executor(sql_query.connection) as executor:
        return executor.execute(query.sql(dialect=sql_query.dialect)).to_dicts()


def _add_centered_cols(
    query: sqlglot.exp.Subquery,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
    *,
    var_cols: tuple[str, ...],
    cov_cols: tuple[tuple[str, str], ...],
) -> sqlglot.exp.Subquery:
    import sqlglot  # noqa: PLC0415

    keep_cols = set(aggr_cols.mean_cols)
    if len(var_cols) == 0:
        keep_cols.update(aggr_cols.var_cols)
    if len(cov_cols) == 0:
        keep_cols.update(col for cols in aggr_cols.cov_cols for col in cols)
    if group_col is not None:
        keep_cols.add(group_col)

    exprs: list[sqlglot.exp.Expression] = [_col(col) for col in keep_cols]
    exprs.extend(
        sqlglot.exp.alias_(
            _float(col) - _mean_over(_float(col), group_col),
            _CENTERED.format(col),
        )
        for col in var_cols
    )
    for left, right in cov_cols:
        valid = (
            _col(left).is_(sqlglot.exp.Null()).not_() &
            _col(right).is_(sqlglot.exp.Null()).not_()
        )
        left_float = _float(left)
        right_float = _float(right)
        left_valid = _if_valid(valid, left_float.copy())
        right_valid = _if_valid(valid, right_float.copy())
        exprs.extend((
            sqlglot.exp.alias_(
                left_float - _mean_over(left_valid, group_col),
                _CENTERED_LEFT.format(left, right),
            ),
            sqlglot.exp.alias_(
                right_float - _mean_over(right_valid, group_col),
                _CENTERED_RIGHT.format(left, right),
            ),
        ))
    return sqlglot.exp.select(*exprs).from_(query).subquery(_SUBQUERY)


def _aggr_exprs(
    sql_query: SQLQuery,
    aggr_cols: tea_tasting.aggr.AggrCols,
    group_col: str | None,
) -> list[sqlglot.exp.Expression]:
    import sqlglot  # noqa: PLC0415

    exprs = [_col(group_col)] if group_col is not None else []
    if aggr_cols.has_count:
        exprs.append(
            sqlglot.exp.alias_(sqlglot.exp.Count(this=sqlglot.exp.Star()), _COUNT),
        )
    exprs.extend(
        sqlglot.exp.alias_(sqlglot.exp.Avg(this=_float(col)), _MEAN.format(col))
        for col in aggr_cols.mean_cols
    )
    exprs.extend(
        sqlglot.exp.alias_(_sample_var(col, var=sql_query.var), _VAR.format(col))
        for col in aggr_cols.var_cols
    )
    exprs.extend(
        sqlglot.exp.alias_(
            _sample_cov(left, right, cov=sql_query.cov),
            _COV.format(left, right),
        )
        for left, right in aggr_cols.cov_cols
    )
    return exprs


def _sample_var(col: str, *, var: bool | str) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    if var is True:
        return sqlglot.exp.Variance(this=_float(col))
    if isinstance(var, str):
        return sqlglot.exp.Anonymous(this=var, expressions=[_float(col)])
    centered = _col(_CENTERED.format(col))
    return _fallback_sample_aggr(centered * centered)


def _sample_cov(
    left: str,
    right: str,
    *,
    cov: bool | str,
) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    if cov is True:
        return sqlglot.exp.CovarSamp(this=_float(left), expression=_float(right))
    if isinstance(cov, str):
        return sqlglot.exp.Anonymous(
            this=cov,
            expressions=[_float(left), _float(right)],
        )
    return _fallback_sample_aggr(
        _col(_CENTERED_LEFT.format(left, right)) *
        _col(_CENTERED_RIGHT.format(left, right)),
    )


def _fallback_sample_aggr(
    centered_expr: sqlglot.exp.Expression,
) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    count = sqlglot.exp.Count(this=centered_expr.copy())
    one = sqlglot.exp.Literal.number(1)
    return sqlglot.exp.Case().when(
        count > one,
        sqlglot.exp.Sum(this=centered_expr) / (count.copy() - one.copy()),
    ).else_(sqlglot.exp.Null())


def _if_valid(
    cond: sqlglot.exp.Expression,
    then: sqlglot.exp.Expression,
) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    return sqlglot.exp.Case().when(cond, then).else_(sqlglot.exp.Null())


def _mean_over(
    expr: sqlglot.exp.Expression,
    group_col: str | None,
) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    return sqlglot.exp.Window(
        this=sqlglot.exp.Avg(this=expr),
        partition_by=[_col(group_col)] if group_col is not None else None,
    )


def _float(col: str) -> sqlglot.exp.Expression:
    import sqlglot  # noqa: PLC0415

    return sqlglot.exp.cast(_col(col), "DOUBLE")


def _col(col: str) -> sqlglot.exp.Column:
    import sqlglot  # noqa: PLC0415

    return sqlglot.exp.column(col)
