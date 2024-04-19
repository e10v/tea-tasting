from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import unittest.mock

import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.proportion


if TYPE_CHECKING:
    from typing import Any

    import ibis.expr.types
    import pandas as pd


@pytest.fixture
def table() -> ibis.expr.types.Table:
    return tea_tasting.datasets.make_users_data(
        n_users=100, ratio=0.9, seed=42, to_ibis=True)

@pytest.fixture
def dataframe(table: ibis.expr.types.Table) -> pd.DataFrame:
    return table.to_pandas()

@pytest.fixture
def data(table: ibis.expr.types.Table) -> dict[Any, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        table,
        group_col="variant",
        has_count=True,
        mean_cols=(),
        var_cols=(),
        cov_cols=(),
    )


def test_sample_ratio_init_default():
    metric = tea_tasting.metrics.proportion.SampleRatio()
    assert metric.ratio == 1
    assert metric.method == "auto"
    assert metric.correction is True

def test_sample_ratio_init_custom():
    metric = tea_tasting.metrics.proportion.SampleRatio(
        {0: 0.5, 1: 0.5},
        method="norm",
        correction=False,
    )
    assert metric.ratio == {0: 0.5, 1: 0.5}
    assert metric.method == "norm"
    assert metric.correction is False


def test_sample_ratio_aggr_cols():
    metric = tea_tasting.metrics.proportion.SampleRatio()
    assert metric.aggr_cols == tea_tasting.metrics.base.AggrCols(has_count=True)


def test_sample_ratio_analyze_table(table: ibis.expr.types.Table):
    metric = tea_tasting.metrics.proportion.SampleRatio()
    result = metric.analyze(table, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.proportion.SampleRatioResult)

def test_sample_ratio_analyze_df(dataframe: pd.DataFrame):
    metric = tea_tasting.metrics.proportion.SampleRatio()
    result = metric.analyze(dataframe, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.proportion.SampleRatioResult)

def test_sample_ratio_analyze_auto():
    metric = tea_tasting.metrics.proportion.SampleRatio()
    with unittest.mock.patch("scipy.stats.binomtest") as mock:
        mock.return_value = NamedTuple("Result", (("pvalue", float),))(pvalue=0.1)
        data = tea_tasting.datasets.make_users_data(
            seed=42,
            n_users=tea_tasting.metrics.proportion._MAX_EXACT_THRESHOLD - 1,
        )
        metric.analyze(data, 0, 1, variant="variant")
        mock.assert_called_once()
    with unittest.mock.patch("scipy.stats.norm.sf") as mock:
        mock.return_value = 0.1
        data = tea_tasting.datasets.make_users_data(
            seed=42,
            n_users=tea_tasting.metrics.proportion._MAX_EXACT_THRESHOLD,
        )
        metric.analyze(data, 0, 1, variant="variant")
        mock.assert_called_once()

def test_sample_ratio_analyze_binom(data: dict[str, tea_tasting.aggr.Aggregates]):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="binom")
    result = metric.analyze(data, 0, 1, variant="variant")
    assert result.control == 54
    assert result.treatment == 46
    assert result.pvalue == pytest.approx(0.48411841360729146)

def test_sample_ratio_analyze_norm_corr(data: dict[str, tea_tasting.aggr.Aggregates]):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=True)
    result = metric.analyze(data, 0, 1, variant="variant")
    assert result.control == 54
    assert result.treatment == 46
    assert result.pvalue == pytest.approx(0.48392730444614607)

def test_sample_ratio_analyze_norm_no_corr(
    data: dict[str, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=False)
    result = metric.analyze(data, 0, 1, variant="variant")
    assert result.control == 54
    assert result.treatment == 46
    assert result.pvalue == pytest.approx(0.4237107971667934)

def test_sample_ratio_analyze_aggregates(data: dict[Any, tea_tasting.aggr.Aggregates]):
    metric = tea_tasting.metrics.proportion.SampleRatio()
    with pytest.raises(NotImplementedError):
        metric.analyze_aggregates(data[0], data[1])
