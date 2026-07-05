from __future__ import annotations

import datetime
from typing import Any

import pyarrow as pa

from tea_tasting.backends._executor import Executor


class FakeCursor:
    def __init__(
        self,
        rows: list[tuple[Any, ...]],
        description: list[tuple[Any, ...]],
    ) -> None:
        self.rows = rows
        self.description = description
        self.closed = False
        self.executed = ""
        self.position = 0

    def execute(self, query: str) -> None:
        self.executed = query

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        rows = self.rows[self.position:self.position + size]
        self.position += size
        return rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_ = cursor

    def cursor(self) -> FakeCursor:
        return self.cursor_


class FakeArrowCursor(FakeCursor):
    def to_arrow_table(self) -> pa.Table:
        return pa.table({"x": [1]})


class FakeArrowReaderCursor(FakeCursor):
    def fetch_arrow_table(self) -> pa.RecordBatchReader:
        table = pa.table({"x": [2]})
        return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def test_executor_connection_closes_cursor() -> None:
    cursor = FakeCursor([(1,)], [("x", int)])
    with Executor(FakeConnection(cursor)) as executor:
        assert executor.execute("SELECT 1") is executor
        assert executor.to_list() == [1]
    assert cursor.executed == "SELECT 1"
    assert cursor.closed is True


def test_executor_cursor_does_not_close_cursor() -> None:
    cursor = FakeCursor([(1, "a")], [("x", int), ("y", str)])
    with Executor(cursor) as executor:
        assert executor.execute("SELECT 1").to_dicts() == [{"x": 1, "y": "a"}]
    assert cursor.closed is False


def test_executor_to_arrow_from_to_arrow_table() -> None:
    cursor = FakeArrowCursor([], [])
    assert Executor(cursor).to_arrow(chunk_size=1).to_pylist() == [{"x": 1}]


def test_executor_to_arrow_from_fetch_arrow_table() -> None:
    cursor = FakeArrowReaderCursor([], [])
    assert Executor(cursor).to_arrow(chunk_size=1).to_pylist() == [{"x": 2}]


def test_executor_to_arrow_fetchall() -> None:
    cursor = FakeCursor(
        [(
            1,
            2.0,
            "a",
            b"b",
            True,
            datetime.date(2026, 7, 4),
            datetime.datetime(2026, 7, 4, 1, 2, 3, 456),
            datetime.time(1, 2, 3, 456),
            datetime.timedelta(days=1, seconds=2, microseconds=3),
            None,
            "unknown",
        )],
        [
            ("i", "Int64"),
            ("f", "DOUBLE"),
            ("s", "VARCHAR(32)"),
            ("b", bytes),
            ("ok", bool),
            ("date", datetime.date),
            ("datetime", datetime.datetime),
            ("time", datetime.time),
            ("timedelta", datetime.timedelta),
            ("none_type", type(None)),
            ("unknown", None),
        ],
    )
    table = Executor(cursor).to_arrow(chunk_size=None)
    assert table.schema == pa.schema({
        "i": pa.int64(),
        "f": pa.float64(),
        "s": pa.string(),
        "b": pa.binary(),
        "ok": pa.bool_(),
        "date": pa.date32(),
        "datetime": pa.timestamp("us"),
        "time": pa.time64("us"),
        "timedelta": pa.duration("us"),
        "none_type": pa.null(),
        "unknown": pa.string(),
    })
    assert table.to_pylist() == [{
        "i": 1,
        "f": 2.0,
        "s": "a",
        "b": b"b",
        "ok": True,
        "date": datetime.date(2026, 7, 4),
        "datetime": datetime.datetime(2026, 7, 4, 1, 2, 3, 456),
        "time": datetime.time(1, 2, 3, 456),
        "timedelta": datetime.timedelta(days=1, seconds=2, microseconds=3),
        "none_type": None,
        "unknown": "unknown",
    }]


def test_executor_to_arrow_sql_type_aliases() -> None:
    aliases = {
        "bigint": ("bigint", pa.int64()),
        "bigint_unsigned": ("bigint unsigned", pa.uint64()),
        "binary_float": ("binary_float", pa.float32()),
        "bit": ("bit", pa.string()),
        "bitstring": ("bitstring", pa.string()),
        "blob": ("blob", pa.binary()),
        "bytea": ("bytea", pa.binary()),
        "byteint": ("byteint", pa.int8()),
        "character": ("character", pa.string()),
        "character_varying": ("character varying(255)", pa.string()),
        "date": ("date", pa.date32()),
        "date64": ("date64", pa.date64()),
        "datetime64": ("datetime64", pa.timestamp("ns")),
        "double_precision": ("double precision", pa.float64()),
        "float": ("float", pa.float32()),
        "int": ("int", pa.int32()),
        "int_unsigned": ("int unsigned", pa.uint32()),
        "interval": ("interval day to second", pa.duration("us")),
        "jsonb": ("jsonb", pa.string()),
        "logical": ("logical", pa.bool_()),
        "long": ("long", pa.int64()),
        "mediumint": ("mediumint", pa.int32()),
        "nvarchar2": ("nvarchar2", pa.string()),
        "real": ("real", pa.float32()),
        "signed": ("signed", pa.int32()),
        "smallint": ("smallint", pa.int16()),
        "time": ("time without time zone", pa.time64("us")),
        "time_ms": ("time_ms", pa.time32("ms")),
        "timestamp": ("timestamp with time zone", pa.timestamp("us")),
        "timestamp_ns": ("timestamp_ns", pa.timestamp("ns")),
        "tinyint_unsigned": ("tinyint unsigned", pa.uint8()),
        "ubigint": ("ubigint", pa.uint64()),
        "uinteger": ("uinteger", pa.uint32()),
        "unsigned_int": ("unsigned int", pa.uint32()),
        "uuid": ("uuid", pa.string()),
        "varbinary": ("varbinary", pa.binary()),
        "varchar2": ("varchar2", pa.string()),
        "year": ("year", pa.int16()),
    }
    cursor = FakeCursor(
        [],
        [(name, type_code) for name, (type_code, _) in aliases.items()],
    )

    table = Executor(cursor).to_arrow(chunk_size=None)

    assert table.schema == pa.schema({
        name: typ for name, (_, typ) in aliases.items()
    })


def test_executor_to_arrow_fetchmany() -> None:
    cursor = FakeCursor([(1,), (2,), (3,)], [("x", "INTEGER")])
    table = Executor(cursor).to_arrow(chunk_size=2)
    assert table.to_pylist() == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_executor_to_arrow_fetchmany_unknown_type_infers_once() -> None:
    cursor = FakeCursor([(None,), (1,)], [("x", None)])
    table = Executor(cursor).to_arrow(chunk_size=1)
    assert table.schema == pa.schema({"x": pa.int64()})
    assert table.to_pylist() == [{"x": None}, {"x": 1}]


def test_executor_to_arrow_fetchmany_unknown_type_promotes_schema() -> None:
    cursor = FakeCursor([(1,), (1.5,)], [("x", None)])
    table = Executor(cursor).to_arrow(chunk_size=1)
    assert table.schema == pa.schema({"x": pa.float64()})
    assert table.to_pylist() == [{"x": 1.0}, {"x": 1.5}]


def test_executor_to_arrow_unknown_string_type() -> None:
    cursor = FakeCursor([(1,)], [("x", "GEOMETRY")])
    table = Executor(cursor).to_arrow(chunk_size=None)
    assert table.schema == pa.schema({"x": pa.int64()})
    assert table.to_pylist() == [{"x": 1}]


def test_executor_to_arrow_fetchmany_empty_unknown() -> None:
    cursor = FakeCursor([], [("x", object)])
    table = Executor(cursor).to_arrow(chunk_size=2)
    assert table.schema == pa.schema({"x": pa.null()})
    assert table.to_pylist() == []


def test_executor_to_arrow_fetchmany_empty_known() -> None:
    cursor = FakeCursor([], [("x", "INTEGER")])
    table = Executor(cursor).to_arrow(chunk_size=2)
    assert table.schema == pa.schema({"x": pa.int32()})
    assert table.to_pylist() == []
