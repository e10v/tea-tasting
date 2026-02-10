from __future__ import annotations

from typing import NamedTuple
import unittest.mock

import numpy as np
import pyarrow as pa
import pytest

import tea_tasting.config
import tea_tasting.metrics.base
import tea_tasting.metrics.nonparametric


@pytest.fixture
def data_arrow() -> pa.Table:
    return pa.table({
        "variant": [0, 0, 0, 0, 1, 1, 1, 1],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })


@pytest.fixture
def data_gran(data_arrow: pa.Table) -> dict[object, pa.Table]:
    return tea_tasting.metrics.base.read_granular(
        data_arrow,
        ("value",),
        variant="variant",
    )


def test_mann_whitney_u_init_default() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value")
    assert metric.column == "value"
    assert metric.alternative == tea_tasting.config.get_config("alternative")
    assert metric.correction == tea_tasting.config.get_config("correction")
    assert metric.method == "auto"
    assert metric.nan_policy == "propagate"


def test_mann_whitney_u_init_custom() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU(
        "value",
        alternative="greater",
        correction=False,
        method="exact",
        nan_policy="omit",
    )
    assert metric.column == "value"
    assert metric.alternative == "greater"
    assert metric.correction is False
    assert metric.method == "exact"
    assert metric.nan_policy == "omit"


def test_mann_whitney_u_init_config() -> None:
    with tea_tasting.config.config_context(alternative="less", correction=True):
        metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value")
    assert metric.column == "value"
    assert metric.alternative == "less"
    assert metric.correction is True


def test_mann_whitney_u_cols() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value")
    assert metric.cols == ("value",)


def test_mann_whitney_u_analyze_frame(data_arrow: pa.Table) -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value")
    result = metric.analyze(data_arrow, 0, 1, variant="variant")
    assert isinstance(result, tea_tasting.metrics.nonparametric.MannWhitneyUResult)


def test_mann_whitney_u_analyze(data_gran: dict[object, pa.Table]) -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value", method="exact")
    result = metric.analyze(data_gran, 0, 1)
    assert result.control == 0
    assert result.treatment == 1
    assert result.effect_size == 1
    assert result.control + result.treatment == 1
    assert result.pvalue == pytest.approx(0.02857142857142857)
    assert result.statistic == 16


def test_mann_whitney_u_analyze_uses_treatment_order(
    data_gran: dict[object, pa.Table],
) -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU(
        "value",
        alternative="greater",
        correction=False,
        method="asymptotic",
        nan_policy="omit",
    )
    with unittest.mock.patch("scipy.stats.mannwhitneyu") as mock:
        mock.return_value = NamedTuple(
            "Result",
            (("statistic", float), ("pvalue", float)),
        )(statistic=16, pvalue=0.1)
        metric.analyze(data_gran, 0, 1)
        mock.assert_called_once()
        assert mock.call_args.kwargs == {
            "alternative": "greater",
            "use_continuity": False,
            "method": "asymptotic",
        }
        assert np.array_equal(
            mock.call_args.args[0],
            np.array([5, 6, 7, 8], dtype=float),
        )
        assert np.array_equal(
            mock.call_args.args[1],
            np.array([1, 2, 3, 4], dtype=float),
        )


def test_mann_whitney_u_analyze_ties() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU("value")
    data: dict[object, pa.Table] = {
        0: pa.table({"value": [0.0, 0.0]}),
        1: pa.table({"value": [0.0, 0.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert result.control == 0.5
    assert result.treatment == 0.5
    assert result.effect_size == 0
    assert result.control + result.treatment == 1


def test_mann_whitney_u_analyze_nan_policy_raise() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU(
        "value",
        nan_policy="raise",
    )
    data: dict[object, pa.Table] = {
        0: pa.table({"value": [1.0, float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    with pytest.raises(ValueError, match=r"Input contains nan\."):
        metric.analyze(data, 0, 1)


def test_mann_whitney_u_analyze_nan_policy_omit_empty() -> None:
    metric = tea_tasting.metrics.nonparametric.MannWhitneyU(
        "value",
        nan_policy="omit",
    )
    data: dict[object, pa.Table] = {
        0: pa.table({"value": [float("nan"), float("nan")]}),
        1: pa.table({"value": [2.0, 3.0]}),
    }
    result = metric.analyze(data, 0, 1)
    assert np.isnan(result.control)
    assert np.isnan(result.treatment)
    assert np.isnan(result.effect_size)
    assert np.isnan(result.pvalue)
    assert np.isnan(result.statistic)
