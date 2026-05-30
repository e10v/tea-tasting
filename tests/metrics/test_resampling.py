from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest

import tea_tasting.config
import tea_tasting.datasets
import tea_tasting.metrics.base
import tea_tasting.metrics.resampling


if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy.typing as npt


@pytest.fixture
def data_arrow() -> pa.Table:
    return tea_tasting.datasets.make_users_data(n_users=100, rng=42)

@pytest.fixture
def data_gran(data_arrow: pa.Table) -> dict[Hashable, pa.Table]:
    return tea_tasting.metrics.base.read_granular(
        data_arrow,
        ("sessions", "orders", "revenue"),
        variant="variant",
    )


def test_bootstrap_init_default() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.columns == "a"
    assert metric.statistic == np.mean
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.confidence_level == tea_tasting.config.get_config("confidence_level")
    assert metric.n_resamples == tea_tasting.config.get_config("n_resamples")
    assert metric.method == "bca"
    assert metric.batch is None
    assert metric.nan_policy == "propagate"
    assert metric.rng is None
    assert metric.random_state is None

def test_bootstrap_init_custom() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        np.mean,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=1000,
        method="basic",
        batch=100,
        nan_policy="omit",
        rng=42,
    )
    assert metric.columns == ("a", "b")
    assert metric.statistic == np.mean
    assert metric.alternative == "greater"
    assert metric.confidence_level == 0.9
    assert metric.n_resamples == 1000
    assert metric.method == "basic"
    assert metric.batch == 100
    assert metric.nan_policy == "omit"
    assert metric.rng == 42
    assert metric.random_state == 42


def test_bootstrap_cols() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("a", np.mean)
    assert metric.cols == ("a",)

    metric = tea_tasting.metrics.resampling.Bootstrap(("a", "b"), np.mean)
    assert metric.cols == ("a", "b")


def test_bootstrap_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap("sessions", np.mean)
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)


def test_bootstrap_analyze_default(data_gran: dict[Hashable, pa.Table]) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "revenue",
        np.mean,
        n_resamples=100,
        rng=42,
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(5.029811320754717)
    assert result.treatment == pytest.approx(5.43)
    assert result.effect_size == pytest.approx(0.4001886792452831)
    assert result.effect_size_ci_lower == pytest.approx(-3.269396309565539)
    assert result.effect_size_ci_upper == pytest.approx(7.219843380442667)
    assert result.rel_effect_size == pytest.approx(0.07956335809137971)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.5658493834599828)
    assert result.rel_effect_size_ci_upper == pytest.approx(1.8185473860534842)

def test_bootstrap_analyze_multiple_columns(
    data_gran: dict[Hashable, pa.Table],
) -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("orders", "sessions"),
        ratio_of_means,
        n_resamples=100,
        rng=42,
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(0.2857142857142857)
    assert result.treatment == pytest.approx(0.20224719101123595)
    assert result.effect_size == pytest.approx(-0.08346709470304975)
    assert result.effect_size_ci_lower == pytest.approx(-0.24780839493679777)
    assert result.effect_size_ci_upper == pytest.approx(0.07730723504025493)
    assert result.rel_effect_size == pytest.approx(-0.2921348314606741)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.6424902672606227)
    assert result.rel_effect_size_ci_upper == pytest.approx(0.4374404130492657)

def test_bootstrap_analyze_division_by_zero(
    data_gran: dict[Hashable, pa.Table],
) -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "orders",
        np.median,
        n_resamples=100,
        rng=42,
        method="basic",
    )
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == 0
    assert result.treatment == 0
    assert result.effect_size == 0
    assert result.effect_size_ci_lower == 0
    assert result.effect_size_ci_upper == 0
    assert np.isnan(result.rel_effect_size)
    assert np.isnan(result.rel_effect_size_ci_lower)
    assert np.isnan(result.rel_effect_size_ci_upper)

def test_bootstrap_analyze_nan_policy_raise() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        nan_policy="raise",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    with pytest.raises(ValueError, match=r"Input contains nan\."):
        metric.analyze(data, 0, 1)

def test_bootstrap_analyze_nan_policy_omit() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, float("nan")]}),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


def test_bootstrap_analyze_nan_policy_omit_multiple_columns() -> None:
    def ratio_of_means(
        sample: npt.NDArray[np.number],
        axis: int,
    ) -> npt.NDArray[np.number]:
        stat = np.mean(sample, axis=axis)
        return stat[0] / stat[1]

    metric = tea_tasting.metrics.resampling.Bootstrap(
        ("a", "b"),
        ratio_of_means,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({
            "a": [2.0, 4.0, float("nan"), 6.0],
            "b": [1.0, 2.0, 4.0, float("nan")],
        }),
        1: pa.table({
            "a": [6.0, 9.0, 8.0, float("nan")],
            "b": [2.0, 3.0, float("nan"), 3.0],
        }),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


def test_bootstrap_analyze_nan_policy_omit_empty() -> None:
    metric = tea_tasting.metrics.resampling.Bootstrap(
        "value",
        np.mean,
        nan_policy="omit",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [float("nan"), float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert np.isnan(result.control)
    assert np.isnan(result.treatment)
    assert np.isnan(result.effect_size)
    assert np.isnan(result.effect_size_ci_lower)
    assert np.isnan(result.effect_size_ci_upper)
    assert np.isnan(result.rel_effect_size)
    assert np.isnan(result.rel_effect_size_ci_lower)
    assert np.isnan(result.rel_effect_size_ci_upper)


def test_quantile(data_gran: dict[Hashable, pa.Table]) -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "revenue",
        q=0.8,
        alternative="greater",
        confidence_level=0.9,
        n_resamples=100,
        rng=42,
    )
    assert metric.column == "revenue"
    assert metric.q == 0.8
    assert metric.nan_policy == "omit"
    result = metric.analyze(data_gran, 0, 1)
    assert isinstance(result, tea_tasting.metrics.resampling.BootstrapResult)
    assert result.control == pytest.approx(11.972000000000001)
    assert result.treatment == pytest.approx(6.2820000000000045)
    assert result.effect_size == pytest.approx(-5.689999999999997)
    assert result.effect_size_ci_lower == pytest.approx(-10.875800000000003)
    assert result.effect_size_ci_upper == float("inf")
    assert result.rel_effect_size == pytest.approx(-0.47527564316739024)
    assert result.rel_effect_size_ci_lower == pytest.approx(-0.8743329817472134)
    assert result.rel_effect_size_ci_upper == float("inf")


def test_quantile_analyze_nan_policy_propagate() -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "value",
        q=0.5,
        n_resamples=10,
        method="basic",
        rng=42,
        nan_policy="propagate",
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, 6.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert np.isnan(result.control)
    assert result.treatment == 4
    assert np.isnan(result.effect_size)
    assert np.isnan(result.rel_effect_size)


def test_quantile_analyze_nan_policy_omit() -> None:
    metric = tea_tasting.metrics.resampling.Quantile(
        "value",
        q=0.5,
        n_resamples=10,
        method="basic",
        nan_policy="omit",
        rng=42,
    )
    data: dict[Hashable, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan"), 3.0]}),
        1: pa.table({"value": [2.0, 4.0, float("nan")]}),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 2
    assert result.treatment == 3
    assert result.effect_size == 1
    assert result.rel_effect_size == 0.5


def test_bootstrap_random_state_keyword_deprecated() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="'random_state' keyword is deprecated",
    ):
        metric = tea_tasting.metrics.resampling.Bootstrap(
            "a",
            np.mean,
            random_state=42,  # ty:ignore[unknown-argument]
        )
    assert metric.rng == 42


def test_bootstrap_rng_and_random_state_raise() -> None:
    with pytest.raises(
        TypeError,
        match="both 'rng' and deprecated keyword 'random_state'",
    ):
        tea_tasting.metrics.resampling.Bootstrap(
            "a",
            np.mean,
            rng=1,
            random_state=42,  # ty:ignore[unknown-argument]
        )
