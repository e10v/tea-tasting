"""Private DB-API query executor."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Protocol


    class Connection(Protocol):
        """PEP 249-compatible connection protocol."""

        def cursor(self) -> Any:
            """Return a cursor."""


    class Cursor(Protocol):
        """PEP 249-compatible cursor protocol."""

        @property
        def description(self) -> Sequence[Sequence[Any]]:
            """Column descriptions."""

        def execute(self, query: str) -> Any:
            """Execute a query."""

        def fetchall(self) -> Any:
            """Fetch all rows."""

        def fetchmany(self, size: int) -> Any:
            """Fetch multiple rows."""


_FETCH_ARROW_NAMES = ("to_arrow_table", "fetch_arrow_table")

_PYTHON_TYPES: dict[type[Any], pa.DataType] = {
    bool: pa.bool_(),
    bytes: pa.binary(),
    datetime.date: pa.date32(),
    datetime.datetime: pa.timestamp("us"),
    datetime.time: pa.time64("us"),
    datetime.timedelta: pa.duration("us"),
    float: pa.float64(),
    int: pa.int64(),
    str: pa.string(),
    type(None): pa.null(),
}

_STRING_TYPES = {
    "bigint unsigned": pa.uint64(),
    "bigint": pa.int64(),
    "bigserial": pa.int64(),
    "binary varying": pa.binary(),
    "binary_double": pa.float64(),
    "binary_float": pa.float32(),
    "binary": pa.binary(),
    "bit": pa.string(),
    "bitstring": pa.string(),
    "blob": pa.binary(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    "bpchar": pa.string(),
    "byte": pa.int8(),
    "bytea": pa.binary(),
    "byteint": pa.int8(),
    "bytes": pa.binary(),
    "char varying": pa.string(),
    "char": pa.string(),
    "character varying": pa.string(),
    "character": pa.string(),
    "clob": pa.string(),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "date": pa.date32(),
    "datetime2": pa.timestamp("us"),
    "datetime64": pa.timestamp("ns"),
    "datetimeoffset": pa.timestamp("us"),
    "datetime": pa.timestamp("us"),
    "double precision": pa.float64(),
    "double": pa.float64(),
    "enum16": pa.string(),
    "enum8": pa.string(),
    "enum": pa.string(),
    "fixedstring": pa.binary(),
    "float32": pa.float32(),
    "float4": pa.float32(),
    "float64": pa.float64(),
    "float8": pa.float64(),
    "float": pa.float32(),
    "image": pa.binary(),
    "int unsigned": pa.uint32(),
    "int": pa.int32(),
    "int1": pa.int8(),
    "int16": pa.int16(),
    "int2": pa.int16(),
    "int32": pa.int32(),
    "int4": pa.int32(),
    "int64": pa.int64(),
    "int8": pa.int8(),
    "integer unsigned": pa.uint32(),
    "integer": pa.int32(),
    "interval": pa.duration("us"),
    "ipv4": pa.string(),
    "ipv6": pa.string(),
    "json": pa.string(),
    "jsonb": pa.string(),
    "logical": pa.bool_(),
    "long": pa.int64(),
    "long varchar": pa.string(),
    "longblob": pa.binary(),
    "longtext": pa.string(),
    "longvarchar": pa.string(),
    "mediumblob": pa.binary(),
    "mediumint unsigned": pa.uint32(),
    "mediumint": pa.int32(),
    "mediumtext": pa.string(),
    "name": pa.string(),
    "nchar varying": pa.string(),
    "nchar": pa.string(),
    "ntext": pa.string(),
    "nvarchar": pa.string(),
    "nvarchar2": pa.string(),
    "real": pa.float32(),
    "rowversion": pa.binary(),
    "serial": pa.int32(),
    "set": pa.string(),
    "short": pa.int16(),
    "signed integer": pa.int32(),
    "signed": pa.int32(),
    "smallint": pa.int16(),
    "smallserial": pa.int16(),
    "smalldatetime": pa.timestamp("us"),
    "sql_double": pa.float64(),
    "sql_varchar": pa.string(),
    "str": pa.string(),
    "string": pa.string(),
    "text": pa.string(),
    "time_ms": pa.time32("ms"),
    "time_ns": pa.time64("ns"),
    "time_s": pa.time32("s"),
    "time": pa.time64("us"),
    "timetz": pa.time64("us"),
    "timestamp_ltz": pa.timestamp("us"),
    "timestamp_ms": pa.timestamp("ms"),
    "timestamp_ns": pa.timestamp("ns"),
    "timestamp_ntz": pa.timestamp("us"),
    "timestamp_s": pa.timestamp("s"),
    "timestamp_tz": pa.timestamp("us"),
    "timestamp_us": pa.timestamp("us"),
    "timestamp": pa.timestamp("us"),
    "timestampltz": pa.timestamp("us"),
    "timestampntz": pa.timestamp("us"),
    "timestamptz": pa.timestamp("us"),
    "tinyblob": pa.binary(),
    "tinyint unsigned": pa.uint8(),
    "tinyint": pa.int8(),
    "tinytext": pa.string(),
    "ubigint": pa.uint64(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "uint8": pa.uint8(),
    "uint": pa.uint32(),
    "uinteger": pa.uint32(),
    "umediumint": pa.uint32(),
    "unsigned bigint": pa.uint64(),
    "unsigned int": pa.uint32(),
    "unsigned integer": pa.uint32(),
    "unsigned": pa.uint64(),
    "usmallint": pa.uint16(),
    "utinyint": pa.uint8(),
    "uniqueidentifier": pa.string(),
    "uuid": pa.string(),
    "varbinary": pa.binary(),
    "varbyte": pa.binary(),
    "varchar": pa.string(),
    "varchar2": pa.string(),
    "xml": pa.string(),
    "year": pa.int16(),
}

_STRING_TYPE_PREFIXES = {
    "character varying": pa.string(),
    "interval ": pa.duration("us"),
    "time with": pa.time64("us"),
    "time without": pa.time64("us"),
    "time ": pa.time64("us"),
    "timestamp with": pa.timestamp("us"),
    "timestamp without": pa.timestamp("us"),
    "timestamp ": pa.timestamp("us"),
}


class Executor:
    def __init__(self, connection: Connection | Cursor) -> None:
        self.cursor: Cursor
        cursor = getattr(connection, "cursor", None)
        if callable(cursor):
            self.cursor = cursor()
            self.can_close_cursor = True
        else:
            self.cursor = connection  # ty:ignore[invalid-assignment]
            self.can_close_cursor = False

    def __enter__(self) -> Executor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        close = getattr(self.cursor, "close", None)
        if self.can_close_cursor and callable(close):
            close()

    def execute(self, query: str) -> Executor:
        self.cursor.execute(query)
        return self

    def to_dicts(self) -> list[dict[str, Any]]:
        names = [str(col[0]) for col in self.cursor.description]
        return [dict(zip(names, row, strict=True)) for row in self.cursor.fetchall()]

    def to_list(self) -> list[Any]:
        return [row[0] for row in self.cursor.fetchall()]

    def to_arrow(self, chunk_size: int | None) -> pa.Table:
        for fetch_arrow_name in _FETCH_ARROW_NAMES:
            fetch_arrow = getattr(self.cursor, fetch_arrow_name, None)
            if callable(fetch_arrow):
                arrow = fetch_arrow()
                return arrow.read_all() if hasattr(arrow, "read_all") else arrow

        col_names = [str(descr[0]) for descr in self.cursor.description]
        col_types = [_to_arrow_type(descr) for descr in self.cursor.description]
        if chunk_size is None:
            return pa.table(_arrays(self.cursor.fetchall(), col_types), names=col_names)

        batches = []
        while rows := self.cursor.fetchmany(chunk_size):
            batches.append(pa.record_batch(_arrays(rows, col_types), names=col_names))
        if len(batches) == 0:
            return pa.table(
                [
                    pa.array([], type=typ if typ is not None else pa.null())
                    for typ in col_types
                ],
                names=col_names,
            )
        if any(typ is None for typ in col_types):
            schema = pa.unify_schemas(
                [batch.schema for batch in batches],
                promote_options="permissive",
            )
            batches = [batch.cast(schema) for batch in batches]
        return pa.Table.from_batches(batches)


def _arrays(rows: Any, col_types: list[pa.DataType | None]) -> list[pa.Array]:
    cols = list(zip(*rows, strict=True)) if len(rows) > 0 else [[] for _ in col_types]
    return [
        pa.array(col, type=typ) if typ is not None else pa.array(col)
        for col, typ in zip(cols, col_types, strict=True)
    ]


def _to_arrow_type(descr: Sequence[Any]) -> pa.DataType | None:
    type_code = descr[1] if len(descr) > 1 else None
    if type_code is None:
        return None
    if isinstance(type_code, type):
        return _PYTHON_TYPES.get(type_code)

    type_name = str(type_code).lower()
    if "(" in type_name:
        type_name = type_name.split("(", maxsplit=1)[0]
    typ = _STRING_TYPES.get(type_name)
    if typ is not None:
        return typ
    for prefix, typ in _STRING_TYPE_PREFIXES.items():
        if type_name.startswith(prefix):
            return typ
    return None
