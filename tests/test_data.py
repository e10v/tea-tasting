from __future__ import annotations

import builtins
from types import MappingProxyType
from typing import TYPE_CHECKING

import narwhals as nw
import narwhals.typing
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.backends
import tea_tasting.data


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from types import ModuleType


pytest_plugins = ("tests.fixtures",)


@pytest.fixture
def aggr_cols() -> tea_tasting.data.AggrCols:
    return tea_tasting.data.AggrCols(
        has_count=True,
        mean_cols=("sessions", "orders"),
        var_cols=("sessions", "orders"),
        cov_cols=(("sessions", "orders"),),
    )


@pytest.fixture
def correct_aggr(data_arrow: pa.Table) -> tea_tasting.aggr.Aggregates:
    return tea_tasting.aggr.Aggregates(
        count_=data_arrow.num_rows,
        mean_={
            "sessions": pc.mean(data_arrow["sessions"]).as_py(),
            "orders": pc.mean(data_arrow["orders"]).as_py(),
        },
        var_={
            "sessions": pc.variance(data_arrow["sessions"], ddof=1).as_py(),
            "orders": pc.variance(data_arrow["orders"], ddof=1).as_py(),
        },
        cov_={
            ("orders", "sessions"): np.cov(
                data_arrow["sessions"].combine_chunks().to_numpy(zero_copy_only=False),
                data_arrow["orders"].combine_chunks().to_numpy(zero_copy_only=False),
                ddof=1,
            )[0, 1],
        },
    )


@pytest.fixture
def correct_aggrs(data_arrow: pa.Table) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
    variant_col = data_arrow["variant"]
    aggrs = {}
    for variant in variant_col.unique().to_pylist():
        variant_data = data_arrow.filter(pc.equal(variant_col, pa.scalar(variant)))
        aggrs |= {variant: tea_tasting.aggr.Aggregates(
            count_=variant_data.num_rows,
            mean_={
                "sessions": pc.mean(variant_data["sessions"]).as_py(),
                "orders": pc.mean(variant_data["orders"]).as_py(),
            },
            var_={
                "sessions": pc.variance(variant_data["sessions"], ddof=1).as_py(),
                "orders": pc.variance(variant_data["orders"], ddof=1).as_py(),
            },
            cov_={
                ("orders", "sessions"): np.cov(
                    variant_data["sessions"]
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False),
                    variant_data["orders"]
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False),
                    ddof=1,
                )[0, 1],
            },
        )}
    return aggrs


@pytest.fixture
def cols() -> tuple[str, ...]:
    return ("sessions", "orders", "revenue")


@pytest.fixture
def correct_gran(
    data_arrow: pa.Table,
    cols: tuple[str, ...],
) -> dict[Hashable, pa.Table]:
    variant_col = data_arrow["variant"]
    table = data_arrow.select(cols)
    return {
        variant: table.filter(pc.equal(variant_col, pa.scalar(variant)))
        for variant in variant_col.unique().to_pylist()
    }


class FakeTable(tea_tasting.backends.BaseTable):
    def __init__(self, data: pa.Table) -> None:
        self.data = data

    def select(self, *cols: str) -> pa.Table:
        return self.data.select(cols) if len(cols) > 0 else self.data

    def select_col_unique(self, col: str) -> list[Hashable]:
        return self.data[col].unique().to_pylist()

    def group_by(self, by: str) -> FakeTableGroupBy:
        return FakeTableGroupBy(self.data, by)

    def aggregate(
        self,
        *,
        has_count: bool,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> tea_tasting.aggr.Aggregates:
        return tea_tasting.backends.NarwhalsFrame(self.data).aggregate(
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )


class FakeTableGroupBy(tea_tasting.backends.BaseTableGroupBy):
    def __init__(self, data: pa.Table, group_col: str) -> None:
        self.data = data
        self.group_col = group_col

    def aggregate(
        self,
        *,
        has_count: bool,
        mean_cols: Sequence[str],
        var_cols: Sequence[str],
        cov_cols: Sequence[tuple[str, str]],
    ) -> dict[Hashable, tea_tasting.aggr.Aggregates]:
        return tea_tasting.backends.NarwhalsFrame(self.data).group_by(
            self.group_col,
        ).aggregate(
            has_count=has_count,
            mean_cols=mean_cols,
            var_cols=var_cols,
            cov_cols=cov_cols,
        )


def test_aggr_cols() -> None:
    aggr_cols = tea_tasting.data.AggrCols(
        has_count=True,
        mean_cols=("b", "a", "a"),
        var_cols=("c", "b", "b"),
        cov_cols=(("b", "a"), ("a", "b"), ("d", "c")),
    )

    assert aggr_cols.has_count is True
    assert set(aggr_cols.mean_cols) == {"a", "b"}
    assert len(aggr_cols.mean_cols) == 2
    assert set(aggr_cols.var_cols) == {"b", "c"}
    assert len(aggr_cols.var_cols) == 2
    assert set(aggr_cols.cov_cols) == {("a", "b"), ("c", "d")}
    assert len(aggr_cols.cov_cols) == 2
    assert len(aggr_cols) == 7


def test_aggr_cols_or() -> None:
    aggr_cols0 = tea_tasting.data.AggrCols(
        has_count=False,
        mean_cols=("a", "b"),
        var_cols=("b", "c"),
        cov_cols=(("a", "b"), ("c", "b")),
    )

    aggr_cols1 = tea_tasting.data.AggrCols(
        has_count=True,
        mean_cols=("b", "c"),
        var_cols=("c", "d"),
        cov_cols=(("b", "c"), ("d", "c")),
    )

    aggr_cols = aggr_cols0 | aggr_cols1

    assert isinstance(aggr_cols, tea_tasting.data.AggrCols)
    assert aggr_cols.has_count is True
    assert set(aggr_cols.mean_cols) == {"a", "b", "c"}
    assert len(aggr_cols.mean_cols) == 3
    assert set(aggr_cols.var_cols) == {"b", "c", "d"}
    assert len(aggr_cols.var_cols) == 3
    assert set(aggr_cols.cov_cols) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert len(aggr_cols.cov_cols) == 3


def test_aggr_cols_len() -> None:
    assert len(tea_tasting.data.AggrCols(
        has_count=False,
        mean_cols=("a", "b"),
        var_cols=("b", "c"),
        cov_cols=(("a", "b"), ("c", "b")),
    )) == 6
    assert len(tea_tasting.data.AggrCols(
        has_count=True,
        mean_cols=("b", "c"),
        var_cols=("c", "d"),
        cov_cols=(("b", "c"), ("d", "c")),
    )) == 7


def test_read_aggregates_no_groups(
    data: narwhals.typing.IntoFrame,
    aggr_cols: tea_tasting.data.AggrCols,
    correct_aggr: tea_tasting.aggr.Aggregates,
) -> None:
    aggr = tea_tasting.data.read_aggregates(data, aggr_cols=aggr_cols)
    _compare_aggrs(aggr, correct_aggr)

def test_read_aggregates_groups(
    data: narwhals.typing.IntoFrame,
    aggr_cols: tea_tasting.data.AggrCols,
    correct_aggrs: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    aggrs = tea_tasting.data.read_aggregates(
        data,
        aggr_cols=aggr_cols,
        variant="variant",
    )
    _compare_aggrs(aggrs[0], correct_aggrs[0])
    _compare_aggrs(aggrs[1], correct_aggrs[1])

def test_read_aggregates_no_count(data_arrow: pa.Table) -> None:
    aggr = tea_tasting.data.read_aggregates(
        data_arrow,
        aggr_cols=tea_tasting.data.AggrCols(
            has_count=False,
            mean_cols=("sessions", "orders"),
        ),
    )
    assert aggr.count_ is None
    assert aggr.var_ == {}
    assert aggr.cov_ == {}

def test_read_aggregates_accepts_precomputed() -> None:
    aggr = tea_tasting.aggr.Aggregates(count_=1)
    assert tea_tasting.data.read_aggregates(
        aggr,
        tea_tasting.data.AggrCols(),
    ) is aggr
    with pytest.raises(ValueError, match="variant"):
        tea_tasting.data.read_aggregates(
            aggr,
            tea_tasting.data.AggrCols(),
            variant="variant",
        )  # ty: ignore[no-matching-overload]

def test_read_aggregates_accepts_precomputed_mapping(
    aggr_cols: tea_tasting.data.AggrCols,
    correct_aggrs: dict[Hashable, tea_tasting.aggr.Aggregates],
) -> None:
    aggr_mapping = MappingProxyType(correct_aggrs)
    aggrs = tea_tasting.data.read_aggregates(
        aggr_mapping,
        aggr_cols=aggr_cols,
        variant="variant",
    )
    assert aggrs is aggr_mapping

def test_read_aggregates_custom_backend(data_arrow: pa.Table) -> None:
    aggr = tea_tasting.data.read_aggregates(
        FakeTable(data_arrow),
        tea_tasting.data.AggrCols(
            has_count=True,
            mean_cols=("sessions", "orders"),
            var_cols=("sessions",),
        ),
    )
    assert aggr.count_ == data_arrow.num_rows
    assert aggr.mean_["sessions"] == pytest.approx(
        pc.mean(data_arrow["sessions"]).as_py(),
    )
    assert aggr.mean_["orders"] == pytest.approx(pc.mean(data_arrow["orders"]).as_py())
    assert aggr.var_["sessions"] == pytest.approx(
        pc.variance(data_arrow["sessions"], ddof=1).as_py(),
    )


def _compare_aggrs(
    left: tea_tasting.aggr.Aggregates,
    right: tea_tasting.aggr.Aggregates,
) -> None:
    assert left.count_ == pytest.approx(right.count_)
    assert left.mean_ == pytest.approx(right.mean_)
    assert left.var_ == pytest.approx(right.var_)
    assert left.cov_ == pytest.approx(right.cov_)


def test_read_granular_no_groups(
    data: narwhals.typing.IntoFrame,
    cols: tuple[str, ...],
    data_arrow: pa.Table,
) -> None:
    gran = tea_tasting.data.read_granular(data, cols=cols)
    assert gran.equals(data_arrow.select(cols))

def test_read_granular_groups(
    data: narwhals.typing.IntoFrame,
    cols: tuple[str, ...],
    correct_gran: dict[Hashable, pa.Table],
) -> None:
    gran = tea_tasting.data.read_granular(
        data,
        cols=cols,
        variant="variant",
    )
    assert gran[0].equals(correct_gran[0])
    assert gran[1].equals(correct_gran[1])

def test_read_granular_mapping(
    cols: tuple[str, ...],
    correct_gran: dict[Hashable, pa.Table],
) -> None:
    gran_mapping = MappingProxyType(correct_gran)
    gran = tea_tasting.data.read_granular(
        gran_mapping,
        cols=cols,
        variant="variant",
    )
    assert gran is gran_mapping

def test_read_granular_narwhals_frame(data_arrow: pa.Table) -> None:
    assert tea_tasting.data.read_granular(
        nw.from_native(data_arrow),
        ("sessions",),
    ).equals(data_arrow.select(("sessions",)))

def test_read_granular_without_ibis(
    monkeypatch: pytest.MonkeyPatch,
    data_arrow: pa.Table,
) -> None:
    def import_without_ibis(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "ibis" or name.startswith("ibis."):
            raise ModuleNotFoundError("No module named 'ibis'", name="ibis")
        return real_import(name, globals_, locals_, fromlist, level)
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", import_without_ibis)
    assert tea_tasting.data.read_granular(data_arrow, ("sessions",)).equals(
        data_arrow.select(("sessions",)),
    )

def test_read_granular_missing_ibis_dependency(
    monkeypatch: pytest.MonkeyPatch,
    data_arrow: pa.Table,
) -> None:
    def import_with_missing_dependency(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "ibis" or name.startswith("ibis."):
            raise ModuleNotFoundError("No module named 'missing'", name="missing")
        return real_import(name, globals_, locals_, fromlist, level)
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError):
        tea_tasting.data.read_granular(data_arrow)

def test_read_granular_custom_backend(data_arrow: pa.Table) -> None:
    gran = tea_tasting.data.read_granular(
        FakeTable(data_arrow),
        cols=("sessions",),
        variant="variant",
    )
    assert set(gran) == {0, 1}
    assert gran[0].column_names == ["sessions"]
    assert gran[1].column_names == ["sessions"]

def test_read_variants_custom_backend(data_arrow: pa.Table) -> None:
    assert set(tea_tasting.data.read_variants(FakeTable(data_arrow), "variant")) == {
        0,
        1,
    }
