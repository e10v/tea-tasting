from __future__ import annotations

import math
from typing import NamedTuple
import unittest.mock

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.proportion


def append_has_order(data: pa.Table) -> pa.Table:
    return data.append_column(
        "has_order",
        pc.greater(data["orders"], 0).cast(pa.int64()),  # type: ignore
    )

@pytest.fixture
def data_arrow() -> pa.Table:
    return append_has_order(tea_tasting.datasets.make_users_data(n_users=100, seed=42))

@pytest.fixture
def data_aggr(data_arrow: pa.Table) -> dict[object, tea_tasting.aggr.Aggregates]:
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
        group_col="variant",
        has_count=True,
        mean_cols=("has_order",),
        var_cols=(),
        cov_cols=(),
    )


def test_proportion_init_default() -> None:
    metric = tea_tasting.metrics.proportion.Proportion("a")
    assert metric.column == "a"
    assert metric.method == "auto"
    assert metric.alternative == "two-sided"
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.correction == tea_tasting.config.get_config("correction")
    assert metric.equal_var is True

def test_proportion_init_custom() -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "b",
        method="norm",
        alternative="greater",
        confidence_level=0.8,
        correction=False,
        equal_var=True,
    )
    assert metric.column == "b"
    assert metric.method == "norm"
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.8
    assert metric.correction is False
    assert metric.equal_var is True

def test_proportion_init_config() -> None:
    with tea_tasting.config.config_context(
        alternative="less",
        confidence_level=0.8,
        correction=False,
        equal_var=False,
    ):
        metric = tea_tasting.metrics.proportion.Proportion("a")
    assert metric.column == "a"
    assert metric.method == "auto"
    assert metric.alternative == "less"
    assert metric.confidence_level == 0.8
    assert metric.correction is False
    assert metric.equal_var is True

def test_proportion_init_raises() -> None:
    with pytest.raises(ValueError, match="two-sided"):
        tea_tasting.metrics.proportion.Proportion(
            "a",
            method="log-likelihood",
            alternative="greater",
        )


def test_proportion_aggr_cols() -> None:
    metric = tea_tasting.metrics.proportion.Proportion("a")
    assert metric.aggr_cols == tea_tasting.metrics.base.AggrCols(
        has_count=True,
        mean_cols=("a",),
    )


def test_proportion_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order")
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.proportion.ProportionResult)

def test_proportion_analyze_auto() -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order")
    with unittest.mock.patch("scipy.stats.barnard_exact") as mock:
        mock.return_value = NamedTuple("Result", (("pvalue", float),))(pvalue=0.1)
        data = append_has_order(tea_tasting.datasets.make_users_data(
            seed=42,
            n_users=tea_tasting.metrics.proportion._MAX_EXACT_THRESHOLD - 1,
        ))
        metric.analyze(data, 0, 1, variant="variant")
        mock.assert_called_once()

    with unittest.mock.patch("scipy.stats.barnard_exact") as mock:
        data = append_has_order(tea_tasting.datasets.make_users_data(
            seed=42,
            n_users=tea_tasting.metrics.proportion._MAX_EXACT_THRESHOLD,
        ))
        result = metric.analyze(data, 0, 1, variant="variant")
        mock.assert_not_called()
        assert not math.isnan(result.effect_size_ci_lower)
        assert not math.isnan(result.effect_size_ci_upper)
        assert not math.isnan(result.rel_effect_size_ci_lower)
        assert not math.isnan(result.rel_effect_size_ci_upper)

def test_proportion_analyze_barnard(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="barnard")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == pytest.approx(0.3584905660377358)
    assert result.treatment == pytest.approx(0.2765957446808511)
    assert result.effect_size == pytest.approx(-0.08189482135688475)
    assert math.isnan(result.effect_size_ci_lower)
    assert math.isnan(result.effect_size_ci_upper)
    assert result.rel_effect_size == pytest.approx(-0.22844344904815217)
    assert math.isnan(result.rel_effect_size_ci_lower)
    assert math.isnan(result.rel_effect_size_ci_upper)
    assert result.pvalue == pytest.approx(0.530637102766593)

def test_proportion_analyze_barnard_less_pooled(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="barnard", alternative="less", equal_var=True)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.2894571757676508)

def test_proportion_analyze_boschloo(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="boschloo")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.4000169399363184)

def test_proportion_analyze_boschloo_greater(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="boschloo", alternative="greater")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.8249417819623234)

def test_proportion_analyze_fisher(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="fisher")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.3999948918153569)

def test_proportion_analyze_fisher_less(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="fisher", alternative="less")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.2546303652212273)

def test_proportion_analyze_log_likelihood(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="log-likelihood")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.5076294534617167)

def test_proportion_analyze_log_likelihood_no_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="log-likelihood", correction=False)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.37976268998844287)

def test_proportion_analyze_pearson(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="pearson")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.pvalue == pytest.approx(0.5083165530405195)

def test_proportion_analyze_pearson_zero_margin() -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="pearson")
    data = {
        0: tea_tasting.aggr.Aggregates(count_=10, mean_={"has_order": 0}),
        1: tea_tasting.aggr.Aggregates(count_=10, mean_={"has_order": 0}),
    }
    result = metric.analyze(data, 0, 1, variant="variant")  # type: ignore
    assert math.isnan(result.pvalue)

def test_proportion_analyze_norm(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion("has_order", method="norm")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.effect_size_ci_lower == pytest.approx(-0.2851524037206986)
    assert result.effect_size_ci_upper == pytest.approx(0.12136276100692911)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5884358933441386)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.44643204227351263)
    assert result.pvalue == pytest.approx(0.5083165530405196)

def test_proportion_analyze_norm_unpooled(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", equal_var=False)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.effect_size_ci_lower == pytest.approx(-0.28368859581157896)
    assert result.effect_size_ci_upper == pytest.approx(0.11989895309780946)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5939973223152473)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.46624528367997886)
    assert result.pvalue == pytest.approx(0.5049066155125612)

def test_proportion_analyze_norm_greater(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", alternative="greater")
    result = metric.analyze(data_aggr, 1, 0, variant="variant")
    assert result.effect_size_ci_lower == pytest.approx(-0.0919114073864812)
    assert result.effect_size_ci_upper == 1
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.25445686244444576)
    assert result.rel_effect_size_ci_upper == float("inf")
    assert result.pvalue == pytest.approx(0.2541582765202598)

def test_proportion_analyze_norm_less(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", alternative="less")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.effect_size_ci_lower == -1
    assert result.effect_size_ci_upper == pytest.approx(0.0919114073864812)
    assert result.rel_effect_size_ci_lower == -1
    assert result.rel_effect_size_ci_upper == pytest.approx(0.3192513062736768)
    assert result.pvalue == pytest.approx(0.2541582765202598)

def test_proportion_analyze_norm_no_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", correction=False)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.effect_size_ci_lower == pytest.approx(-0.26508014358420723)
    assert result.effect_size_ci_upper == pytest.approx(0.10129050087043773)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5647346534627081)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.367670355686605)
    assert result.pvalue == pytest.approx(0.3809096823118986)


def test_proportion_analyze_norm_corr_widens_ci(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric_corr = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", correction=True)
    metric_no_corr = tea_tasting.metrics.proportion.Proportion(
        "has_order", method="norm", correction=False)
    result_corr = metric_corr.analyze(data_aggr, 0, 1, variant="variant")
    result_no_corr = metric_no_corr.analyze(data_aggr, 0, 1, variant="variant")
    assert result_corr.effect_size_ci_lower < result_no_corr.effect_size_ci_lower
    assert result_corr.effect_size_ci_upper > result_no_corr.effect_size_ci_upper
    assert (
        result_corr.rel_effect_size_ci_lower <
        result_no_corr.rel_effect_size_ci_lower
    )
    assert (
        result_corr.rel_effect_size_ci_upper >
        result_no_corr.rel_effect_size_ci_upper
    )


def test_proportion_analyze_norm_ci_bounds_limited() -> None:
    metric = tea_tasting.metrics.proportion.Proportion(
        "has_order",
        method="norm",
        correction=False,
    )
    data = {
        0: tea_tasting.aggr.Aggregates(count_=20, mean_={"has_order": 0.05}),
        1: tea_tasting.aggr.Aggregates(count_=20, mean_={"has_order": 1.0}),
    }
    result = metric.analyze(data, 0, 1, variant="variant")  # type: ignore
    assert result.effect_size_ci_lower == pytest.approx(0.640490098153452)
    assert result.effect_size_ci_upper == 1

    result = metric.analyze(data, 1, 0, variant="variant")  # type: ignore
    assert result.effect_size_ci_lower == -1
    assert result.effect_size_ci_upper == pytest.approx(-0.640490098153452)


def test_sample_ratio_init_default() -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio()
    assert metric.ratio == 1
    assert metric.method == "auto"
    assert metric.correction == tea_tasting.config.get_config("correction")

def test_sample_ratio_init_custom() -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio(
        {0: 0.5, 1: 0.5},
        method="norm",
        correction=False,
    )
    assert metric.ratio == {0: 0.5, 1: 0.5}
    assert metric.method == "norm"
    assert metric.correction is False

def test_sample_ratio_init_config() -> None:
    with tea_tasting.config.config_context(correction=False):
        metric = tea_tasting.metrics.proportion.SampleRatio()
    assert metric.correction is False


def test_sample_ratio_aggr_cols() -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio()
    assert metric.aggr_cols == tea_tasting.metrics.base.AggrCols(has_count=True)


def test_sample_ratio_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio()
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.proportion.SampleRatioResult)

def test_sample_ratio_analyze_auto() -> None:
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
) -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio(method="binom")
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.6172994135892521)

def test_sample_ratio_analyze_norm_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=True)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.6170750774519738)

def test_sample_ratio_analyze_norm_no_corr(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio(method="norm", correction=False)
    result = metric.analyze(data_aggr, 0, 1, variant="variant")
    assert result.control == 53
    assert result.treatment == 47
    assert result.pvalue == pytest.approx(0.5485062355001472)

def test_sample_ratio_analyze_aggregates(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
) -> None:
    metric = tea_tasting.metrics.proportion.SampleRatio()
    with pytest.raises(NotImplementedError):
        metric.analyze_aggregates(data_aggr[0], data_aggr[1])
