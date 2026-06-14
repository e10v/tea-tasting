from __future__ import annotations

import numpy as np
import pytest

import tea_tasting.utils.checks


def test_check_scalar_typ() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, typ=int) == 1
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.check_scalar(1, typ=str)

def test_check_scalar_ge() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, ge=1) == 1
    with pytest.raises(ValueError, match="must be >="):
        tea_tasting.utils.checks.check_scalar(1, ge=2)

def test_check_scalar_gt() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, gt=0) == 1
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.check_scalar(1, gt=1)

def test_check_scalar_le() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, le=1) == 1
    with pytest.raises(ValueError, match="must be <="):
        tea_tasting.utils.checks.check_scalar(1, le=0)

def test_check_scalar_lt() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, lt=2) == 1
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.checks.check_scalar(1, lt=1)

def test_check_scalar_ne() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, ne=2) == 1
    with pytest.raises(ValueError, match="must be !="):
        tea_tasting.utils.checks.check_scalar(1, ne=1)

def test_check_scalar_is_in() -> None:
    assert tea_tasting.utils.checks.check_scalar(1, in_={0, 1}) == 1
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.checks.check_scalar(1, in_={0, 2})


def test_auto_check_alpha() -> None:
    assert tea_tasting.utils.checks.auto_check(0.05, "alpha") == 0.05
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "alpha")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(0.0, "alpha")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.checks.auto_check(1.0, "alpha")

def test_auto_check_alternative() -> None:
    assert (
        tea_tasting.utils.checks.auto_check("two-sided", "alternative")
        == "two-sided"
    )
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(2, "alternative")
    with pytest.raises(ValueError, match="must be in"):
        tea_tasting.utils.checks.auto_check("2s", "alternative")

def test_auto_check_confidence_level() -> None:
    assert tea_tasting.utils.checks.auto_check(0.95, "confidence_level") == 0.95
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "confidence_level")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(0.0, "confidence_level")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.checks.auto_check(1.0, "confidence_level")

def test_auto_check_correction() -> None:
    assert tea_tasting.utils.checks.auto_check(True, "correction") is True
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "correction")

def test_auto_check_equal_var() -> None:
    assert tea_tasting.utils.checks.auto_check(True, "equal_var") is True
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "equal_var")

def test_auto_check_n_obs() -> None:
    assert tea_tasting.utils.checks.auto_check(2, "n_obs") == 2
    assert tea_tasting.utils.checks.auto_check((2, 3), "n_obs") == (2, 3)
    assert tea_tasting.utils.checks.auto_check(None, "n_obs") is None
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0.5, "n_obs")
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check((0.5, 2), "n_obs")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(1, "n_obs")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check((1, 2), "n_obs")

def test_auto_check_n_resamples() -> None:
    assert tea_tasting.utils.checks.auto_check(1, "n_resamples") == 1
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(0, "n_resamples")
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0.5, "n_resamples")

def test_auto_check_power() -> None:
    assert tea_tasting.utils.checks.auto_check(0.8, "power") == 0.8
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "power")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(0.0, "power")
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting.utils.checks.auto_check(1.0, "power")

def test_auto_check_ratio() -> None:
    assert tea_tasting.utils.checks.auto_check(1.5, "ratio") == 1.5
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check("str", "ratio")
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting.utils.checks.auto_check(0.0, "ratio")

def test_auto_check_rng() -> None:
    generator = np.random.default_rng(42)
    seed_sequence = np.random.SeedSequence(42)
    assert tea_tasting.utils.checks.auto_check(42, "rng") == 42
    assert tea_tasting.utils.checks.auto_check(generator, "rng") is generator
    assert tea_tasting.utils.checks.auto_check(seed_sequence, "rng") is seed_sequence
    assert tea_tasting.utils.checks.auto_check(None, "rng") is None
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0.5, "rng")

def test_auto_check_use_t() -> None:
    assert tea_tasting.utils.checks.auto_check(False, "use_t") is False
    with pytest.raises(TypeError):
        tea_tasting.utils.checks.auto_check(0, "use_t")
