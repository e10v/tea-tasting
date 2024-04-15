from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.mean


if TYPE_CHECKING:
    import ibis.expr.types
    import pandas as pd


@pytest.fixture
def table() -> ibis.expr.types.Table:
    return tea_tasting.datasets.make_users_data(
        n_users=100, covariates=True, seed=42, to_ibis=True)

@pytest.fixture
def dataframe(table: ibis.expr.types.Table) -> pd.DataFrame:
    return table.to_pandas()

@pytest.fixture
def data(table: ibis.expr.types.Table) -> dict[str, tea_tasting.aggr.Aggregates]:
    cols = (
        "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    )
    return tea_tasting.aggr.read_aggregates(
        table,
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
    )
    assert metric.numer == "a"
    assert metric.denom == "b"
    assert metric.numer_covariate == "c"
    assert metric.denom_covariate == "d"
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False

def test_ratio_of_means_init_config():
    with tea_tasting.config.config_context(
        alternative="greater",
        confidence_level=0.9,
        equal_var=True,
        use_t=False,
    ):
        metric = tea_tasting.metrics.mean.RatioOfMeans("a")
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.equal_var is True
    assert metric.use_t is False


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


def test_ratio_of_means_analyze_table(table: ibis.expr.types.Table):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        numer_covariate="orders_covariate",
    )
    result = metric.analyze(table, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.mean.MeansResult)

def test_ratio_of_means_analyze_df(dataframe: pd.DataFrame):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
    )
    result = metric.analyze(dataframe, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.mean.MeansResult)

def test_ratio_of_means_analyze_basic(data: dict[str, tea_tasting.aggr.Aggregates]):
    metric = tea_tasting.metrics.mean.RatioOfMeans(numer="orders")
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeansResult)
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
    data: dict[str, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        alternative="greater",
        equal_var=True,
    )
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeansResult)
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
    data: dict[str, tea_tasting.aggr.Aggregates],
):
    metric = tea_tasting.metrics.mean.RatioOfMeans(
        numer="orders",
        denom="sessions",
        numer_covariate="orders_covariate",
        denom_covariate="sessions_covariate",
        alternative="less",
        use_t=False,
    )
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.mean.MeansResult)
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


def test_mean(data: dict[str, tea_tasting.aggr.Aggregates]):
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
    _compare_results(metric.analyze(data, 0, 1), ratio_metric.analyze(data, 0, 1))


def _compare_results(
    left: tea_tasting.metrics.mean.MeansResult,
    right: tea_tasting.metrics.mean.MeansResult,
) -> None:
    assert isinstance(left, tea_tasting.metrics.mean.MeansResult)
    assert isinstance(right, tea_tasting.metrics.mean.MeansResult)
    assert left.control == pytest.approx(right.control)
    assert left.treatment == pytest.approx(right.treatment)
    assert left.effect_size == pytest.approx(right.effect_size)
    assert left.effect_size_ci_lower == pytest.approx(right.effect_size_ci_lower)
    assert left.effect_size_ci_upper == pytest.approx(right.effect_size_ci_upper)
    assert left.rel_effect_size == pytest.approx(right.rel_effect_size)
    assert left.rel_effect_size_ci_lower == pytest.approx(
        right.rel_effect_size_ci_lower)
    assert left.rel_effect_size_ci_upper == pytest.approx(
        right.rel_effect_size_ci_upper)
    assert left.pvalue == pytest.approx(right.pvalue)
    assert left.statistic == pytest.approx(right.statistic)
