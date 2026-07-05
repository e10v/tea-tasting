from __future__ import annotations

import pytest

import tea_tasting.aggr
import tea_tasting.backends.base


def test_get_aggregates_handles_folded_aliases() -> None:
    aggr = tea_tasting.backends.base._get_aggregates(
        [{
            "__COUNT__": 2,
            "__MEAN__X__": 1.5,
            "__VAR__X__": 0.5,
            "__COV__X__Y__": 0.25,
        }],
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=("x",),
            var_cols=("x",),
            cov_cols=(("x", "y"),),
        ),
    )

    assert aggr.count_ == 2
    assert aggr.mean_ == {"x": 1.5}
    assert aggr.var_ == {"x": 0.5}
    assert aggr.cov_ == {("x", "y"): 0.25}


def test_get_aggregates_handles_folded_group_column() -> None:
    aggrs = tea_tasting.backends.base._get_aggregates(
        [
            {"VARIANT": 0, "__COUNT__": 2},
            {"VARIANT": 1, "__COUNT__": 3},
        ],
        tea_tasting.aggr.AggrCols(
            has_count=True,
            mean_cols=(),
            var_cols=(),
            cov_cols=(),
        ),
        "variant",
    )

    assert set(aggrs) == {0, 1}
    assert aggrs[0].count_ == 2
    assert aggrs[1].count_ == 3


def test_get_aggregates_fails_on_ambiguous_folded_alias() -> None:
    with pytest.raises(KeyError, match="Ambiguous result column name"):
        tea_tasting.backends.base._get_aggregates(
            [{
                "__MEAN__X__": 1,
                "__Mean__x__": 2,
            }],
            tea_tasting.aggr.AggrCols(
                has_count=False,
                mean_cols=("x",),
                var_cols=(),
                cov_cols=(),
            ),
        )


def test_get_aggregates_converts_null_to_nan() -> None:
    aggr = tea_tasting.backends.base._get_aggregates(
        [{"__MEAN__X__": None}],
        tea_tasting.aggr.AggrCols(
            has_count=False,
            mean_cols=("x",),
            var_cols=(),
            cov_cols=(),
        ),
    )

    assert aggr.mean_["x"] != aggr.mean_["x"]
