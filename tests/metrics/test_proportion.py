from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import unittest.mock

import pytest

import tea_tasting.aggr
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.proportion


if TYPE_CHECKING:
    import pyarrow as pa


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42)

@pytest.fixture
def data_aggr(data_arrow: pa.Table) -> dict[object, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
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


def test_sample_ratio_analyze_frame(data_arrow: pa.Table):
    metric = tea_tasting.metrics.proportion.SampleRatio()
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
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

def test_sample_ratio_analyze_binom(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="binom")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.6172994135892521)

def test_sample_ratio_analyze_norm_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=True)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.6170750774519738)

def test_sample_ratio_analyze_norm_no_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=False)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.5485062355001472)

def test_sample_ratio_analyze_aggregates(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.proportion.SampleRatio()
    with pytest.raises(NotImplementedError):
        metric.analyze_aggregates(data_aggr[0], data_aggr[1])
