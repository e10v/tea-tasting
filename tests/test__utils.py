from __future__ import annotations

import pytest

import tea_tasting._utils


def test_check_scalar_typ():
    tea_tasting._utils.check_scalar(1, typ=int)
    with pytest.raises(TypeError):
        tea_tasting._utils.check_scalar(1, typ=str)

def test_check_scalar_ge():
    tea_tasting._utils.check_scalar(1, ge=1)
    with pytest.raises(ValueError, match="must be >="):
        tea_tasting._utils.check_scalar(1, ge=2)

def test_check_scalar_gt():
    tea_tasting._utils.check_scalar(1, gt=0)
    with pytest.raises(ValueError, match="must be >"):
        tea_tasting._utils.check_scalar(1, gt=1)

def test_check_scalar_le():
    tea_tasting._utils.check_scalar(1, le=1)
    with pytest.raises(ValueError, match="must be <="):
        tea_tasting._utils.check_scalar(1, le=0)

def test_check_scalar_lt():
    tea_tasting._utils.check_scalar(1, lt=2)
    with pytest.raises(ValueError, match="must be <"):
        tea_tasting._utils.check_scalar(1, lt=1)
