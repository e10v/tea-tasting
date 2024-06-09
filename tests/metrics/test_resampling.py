from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import tea_tasting.aggr
import tea_tasting.config
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.resampling


if TYPE_CHECKING:
    from typing import Any

    import ibis.expr.types
    import numpy.typing as npt
    import pandas as pd


@pytest.fixture
def table() -> ibis.expr.types.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, seed=42, to_ibis=True)

@pytest.fixture
def dataframe(table: ibis.expr.types.Table) -> pd.DataFrame:
    return table.to_pandas()

@pytest.fixture
def data(dataframe: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    return tea_tasting.metrics.base.read_dataframes(
        dataframe,
        ("sessions", "orders", "revenue"),
        variant="variant",
    )


def test_bootstrap_init_default():
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.columns == "a"
    assert metric.statistic == np.mean
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.n_resamples == tea_tasting.config.get_config("n_resamples")
    assert metric.method == "bca"
    assert metric.batch is None
    assert metric.random_state is None

def test_bootstrap_init_custom():
    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        np.mean,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=1000,
        method="basic",
        batch=100,
        random_state=42,
    )
    assert metric.columns == ("a", "b")
    assert metric.statistic == np.mean
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.n_resamples == 1000
    assert metric.method == "basic"
    assert metric.batch == 100
    assert metric.random_state == 42


def test_bootstrap_cols():
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.cols == ("a",)

    metric = tea_tasting.metrics.resampling.Bootstrap(("a", "b"), np.mean)
    assert metric.cols == ("a", "b")


def test_bootstrap_analyze_table(table: ibis.expr.types.Table):
    metric = tea_tasting.metrics.resampling.Bootstrap("orders", np.mean)
    result = metric.analyze(table, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)

def test_bootstrap_analyze_df(dataframe: ibis.expr.types.Table):
    metric = tea_tasting.metrics.resampling.Bootstrap("sessions", np.mean)
    result = metric.analyze(dataframe, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)


def test_bootstrap_analyze_default(data: dict[Any, pd.DataFrame]):
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "revenue",
        np.mean,
        n_resamples=100,
        random_state=42,
    )
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(5.029606016096378)
    assert result.treatment == pytest.approx(5.430045947447926)
    assert result.effect_size == pytest.approx(0.4004399313515483)
    assert result.effect_size_ci_lower == pytest.approx(-3.269115518352006)
    assert result.effect_size_ci_upper == pytest.approx(7.220410053935425)
    assert result.rel_effect_size == pytest.approx(0.07961656043634635)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5658060166766641)
    assert result.rel_effect_size_ci_upper == pytest.approx(1.8185107973505807)

def test_bootstrap_analyze_multiple_columns(data: dict[Any, pd.DataFrame]):
    def ratio_of_means(
        sample: npt.NDArray[np.number[Any]],
        axis: int,
    ) -> npt.NDArray[np.number[Any]]:
        stat = np.mean(sample, axis=axis)  # type: ignore
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("orders", "sessions"),
        ratio_of_means,
        n_resamples=100,
        random_state=42,
    )
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(0.2857142857142857)
    assert result.treatment == pytest.approx(0.20224719101123595)
    assert result.effect_size == pytest.approx(-0.08346709470304975)
    assert result.effect_size_ci_lower == pytest.approx(-0.24780839493679777)
    assert result.effect_size_ci_upper == pytest.approx(0.07730723504025493)
    assert result.rel_effect_size == pytest.approx(-0.2921348314606741)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.6424902672606227)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.4374404130492657)

def test_bootstrap_analyze_division_by_zero(data: dict[Any, pd.DataFrame]):
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "orders",
        np.median,
        n_resamples=100,
        random_state=42,
        method="basic",
    )
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == 0
    assert result.treatment == 0
    assert result.effect_size == 0
    assert result.effect_size_ci_lower == 0
    assert result.effect_size_ci_upper == 0
    assert np.isnan(result.rel_effect_size)
    assert np.isnan(result.rel_effect_size_ci_lower)
    assert np.isnan(result.rel_effect_size_ci_upper)

def test_quantile(data: dict[Any, pd.DataFrame]):
    metric = tea_tasting.metrics.resampling.Quantile(
        "revenue",
        q=0.8,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=100,
        random_state=42,
    )
    assert metric.column == "revenue"
    assert metric.q == 0.8
    result = metric.analyze(data, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(11.97241622964322)
    assert result.treatment == pytest.approx(6.283899054876212)
    assert result.effect_size == pytest.approx(-5.688517174767009)
    assert result.effect_size_ci_lower == pytest.approx(-10.875502551863555)
    assert result.effect_size_ci_upper == float("inf")
    assert result.rel_effect_size == pytest.approx(-0.4751352664036579 )
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.8744367099313992)
    assert result.rel_effect_size_ci_upper == float("inf")
