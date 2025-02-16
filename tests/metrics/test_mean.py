from __future__ import annotations

from typing import TYPE_CHECKING
import unittest.mock

import pytest

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.mean


if TYPE_CHECKING:
    import pyarrow as pa


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, covariates=True, seed=42)

@pytest.fixture
def data_aggr(data_arrow: pa.Table) -> dict[object, tea_tasting.aggr.Aggregates]:
    cols = (
        "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    )
    return tea_tasting.aggr.read_aggregates(
        data_arrow,
        group_col="variant",
        has_count=True,
        mean_cols=cols,
        var_cols=cols,
        cov_cols=tuple(
            (col0, col1)
            for col0 in cols
            for col1 in cols
            if col0 < col1
        ),
    )

@pytest.fixture
def power_data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(
        n_users=100, covariates=True, seed=42,
        sessions_uplift=0, orders_uplift=0, revenue_uplift=0,
    )

@pytest.fixture
def power_data_aggr(power_data_arrow: pa.Table) -> tea_tasting.aggr.Aggregates:
    cols = (
        "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    )
    return tea_tasting.aggr.read_aggregates(
        power_data_arrow,
        group_col=None,
        has_count=True,
        mean_cols=cols,
        var_cols=cols,
        cov_cols=tuple(
            (col0, col1)
            for col0 in cols
            for col1 in cols
            if col0 < col1
        ),
    )


def test_ratio_of_means_init_default():
    metric = tea_tasting.metrics.mean.RatioOfMeans("a")
    assert metric.numer == "a"
    assert metric.denom is None
    assert metric.numer_covariate is None
    assert metric.denom_covariate is None
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.equal_var == tea_tasting.config.get_config("equal_var")
    assert metric.use_t == tea_tasting.config.get_config("use_t")
    assert metric.alpha == tea_tasting.config.get_config("alpha")
    assert metric.ratio == tea_tasting.config.get_config("ratio")
    assert metric.power == tea_tasting.config.get_config("power")
    assert metric.effect_size is None
    assert metric.rel_effect_size is None
    assert metric.n_obs is None

def test_ratio_of_means_init_custom():
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="a",
        denom="b",
        numer_covariate="c",
        denom_covariate="d",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        alpha=0.1,
        ratio=0.5,
        power=0.75,
        rel_effect_size=0.08,
        n_obs=(5_000, 10_000),
    )
    assert metric.numer == "a"
    assert metric.denom == "b"
    assert metric.numer_covariate == "c"
    assert metric.denom_covariate == "d"
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False
    assert metric.alpha == 0.1
    assert metric.ratio == 0.5
    assert metric.power == 0.75
    assert metric.effect_size is None
    assert metric.rel_effect_size == 0.08
    assert metric.n_obs == (5_000, 10_000)
    metric = tea_tasting.metrics.mean.RatioOfMeans("a", effect_size=(1, 0.2))
    assert metric.effect_size == (1, 0.2)

def test_ratio_of_means_init_config():
    with tea_tasting.config.config_context(
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        alpha=0.1,
        ratio=0.5,
        power=0.75,
        n_obs=(5_000, 10_000),
    ):
        metric = tea_tasting.metrics.mean.RatioOfMeans("a")
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False
    assert metric.alpha == 0.1
    assert metric.ratio == 0.5
    assert metric.power == 0.75
    assert metric.n_obs == (5_000, 10_000)


def test_ratio_of_means_aggr_cols():
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="a",
        denom="b",
        numer_covariate="c",
    )
    aggr_cols = metric.aggr_cols
    assert isinstance(aggr_cols, tea_tasting.metrics.base.AggrCols)
    assert aggr_cols.has_count is True
    assert aggr_cols.mean_cols == ("a", "b", "c")
    assert aggr_cols.var_cols == ("a", "b", "c")
    assert len(aggr_cols.cov_cols) == 3
    assert set(aggr_cols.cov_cols) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_ratio_of_means_analyze_frame(data_arrow: pa.Table):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
    )
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)

def test_ratio_of_means_analyze_basic(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.5660377358490566)
    assert result.treatment == pytest.approx(0.3829787234042553)
    assert result.effect_size == pytest.approx(-0.18305901244480127)
    assert result.effect_size_ci_lower == pytest.approx(-0.5004359360326984)
    assert result.effect_size_ci_upper == pytest.approx(0.13431791114309583)
    assert result.rel_effect_size == pytest.approx(-0.32340425531914896)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.6593351111187646)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.34378920945924096)
    assert result.pvalue == pytest.approx(0.2551230546709908)
    assert result.statistic == pytest.approx(-1.1447678040118034)

def test_ratio_of_means_analyze_ratio_greater_equal_var(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        alternative="greater",
        equal_var=True,
    )
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.2857142857142857)
    assert result.treatment == pytest.approx(0.20224719101123595)
    assert result.effect_size == pytest.approx(-0.08346709470304975)
    assert result.effect_size_ci_lower == pytest.approx(-0.21187246704082818)
    assert result.effect_size_ci_upper == float("inf")
    assert result.rel_effect_size == pytest.approx(-0.2921348314606741)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5852384654309937)
    assert result.rel_effect_size_ci_upper == float("inf")
    assert result.pvalue == pytest.approx(0.8584716347525132)
    assert result.statistic == pytest.approx(-1.0794048813446926)

def test_ratio_of_means_analyze_ratio_less_use_norm(
    data_aggr: dict[object, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        alternative="less",
        use_t=False,
    )
    result = metric.analyze(data_aggr, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeanResult)
    assert result.control == pytest.approx(0.25572348175909004)
    assert result.treatment == pytest.approx(0.23549786496156158)
    assert result.effect_size == pytest.approx(-0.020225616797528462)
    assert result.effect_size_ci_lower == float("-inf")
    assert result.effect_size_ci_upper == pytest.approx(0.07287939184944564)
    assert result.rel_effect_size == pytest.approx(-0.07909174651619377)
    assert result.rel_effect_size_ci_lower == float("-inf")
    assert result.rel_effect_size_ci_upper == pytest.approx(0.34578466493619153)
    assert result.pvalue == pytest.approx(0.3604265417728255)
    assert result.statistic == pytest.approx(-0.3573188986307722)


def test_ratio_of_means_solve_power_frame(power_data_arrow: pa.Table):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        rel_effect_size=0.1,
    )
    results = metric.solve_power(power_data_arrow, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power > 0
    assert result.power < 1
    assert result.rel_effect_size == 0.1

def test_ratio_of_means_solve_power_power(power_data_aggr: tea_tasting.aggr.Aggregates):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        alternative="greater",
        rel_effect_size=0.1,
        n_obs=10_000,
    )
    results = metric.solve_power(power_data_aggr, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == pytest.approx(0.895653144822902)
    assert result.effect_size == pytest.approx(0.1 * power_data_aggr.mean("orders"))
    assert result.rel_effect_size == 0.1
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        alternative="less",
        n_obs=10_000,
    )
    results = metric.solve_power(power_data_aggr, "effect_size")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == pytest.approx(-0.04027000268908317)
    assert result.rel_effect_size == pytest.approx(-0.08568085678528335)
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_rel_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", n_obs=10_000)
    results = metric.solve_power(power_data_aggr, "rel_effect_size")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == pytest.approx(0.04537464375458489)
    assert result.rel_effect_size == pytest.approx(0.09654179522252104)
    assert result.n_obs == 10_000

def test_ratio_of_means_solve_power_n_obs(power_data_aggr: tea_tasting.aggr.Aggregates):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", effect_size=0.05)
    results = metric.solve_power(power_data_aggr, "n_obs")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 1
    result = results[0]
    assert result.power == 0.8
    assert result.effect_size == 0.05
    assert result.rel_effect_size == pytest.approx(
        0.05 / power_data_aggr.mean("orders"))
    assert result.n_obs == 8236

def test_ratio_of_means_solve_power_multi(power_data_aggr: tea_tasting.aggr.Aggregates):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        rel_effect_size=(0.05, 0.1),
        n_obs=(5_000, 10_000),
    )
    results = metric.solve_power(power_data_aggr, "power")
    assert isinstance(results, tea_tasting.metrics.base.MetricPowerResults)
    assert len(results) == 4

def test_ratio_of_means_solve_power_raises_effect_size(
    power_data_aggr: tea_tasting.aggr.Aggregates,
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    with pytest.raises(ValueError, match="One of them should be defined"):
        metric.solve_power(power_data_aggr, "power")
    with pytest.raises(ValueError, match="Only one of them should be defined"):
        tea_tasting.metrics.mean.RatioOfMeans(
            numer="orders",
            effect_size=0.05,
            rel_effect_size=0.1,
        )

def test_ratio_of_means_solve_power_raises_max_iter(
    power_data_aggr: tea_tasting.aggr.Aggregates,
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders", rel_effect_size=0.01)
    with (
        unittest.mock.patch("tea_tasting.metrics.mean.MAX_ITER", 1),
        pytest.raises(RuntimeError, match="Maximum number of iterations"),
    ):
        metric.solve_power(power_data_aggr, "n_obs")


def test_mean_analyze(data_aggr: dict[object, tea_tasting.aggr.Aggregates]):
    metric = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
    )
    ratio_metric = tea_tasting.metrics.mean.RatioOfMeans(
        "orders",
        numer_covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
    )
    assert metric.analyze(data_aggr, 0, 1) == ratio_metric.analyze(data_aggr, 0, 1)


def test_mean_solve_power(power_data_aggr: tea_tasting.aggr.Aggregates):
    metric = tea_tasting.metrics.mean.Mean(
        "orders",
        covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        rel_effect_size=0.1,
    )
    ratio_metric = tea_tasting.metrics.mean.RatioOfMeans(
        "orders",
        numer_covariate="orders_covariate",
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
        rel_effect_size=0.1,
    )
    assert metric.solve_power(power_data_aggr, "power") ==  ratio_metric.solve_power(
        power_data_aggr, "power")
