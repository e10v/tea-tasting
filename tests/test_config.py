from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import tea_tasting.config


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def reset_config() -> Generator[None, object, None]:
    old_config = tea_tasting.config._global_config.copy()

    try:
        yield
    finally:
        tea_tasting.config._global_config = old_config


@pytest.mark.usefixtures("reset_config")
def test_get_config():
    config = tea_tasting.config.get_config()
    assert config == tea_tasting.config._global_config
    config["equal_var"] = not config["equal_var"]
    assert config != tea_tasting.config._global_config

    assert (
        tea_tasting.config.get_config("equal_var") ==
        tea_tasting.config._global_config["equal_var"]
    )


@pytest.mark.usefixtures("reset_config")
def test_set_config():
    tea_tasting.config.set_config(equal_var=True)
    assert tea_tasting.config._global_config["equal_var"] is True

    tea_tasting.config.set_config(equal_var=False)
    assert tea_tasting.config._global_config["equal_var"] is False


@pytest.mark.usefixtures("reset_config")
def test_config_context():
    old_equal_var = tea_tasting.config._global_config["equal_var"]

    with tea_tasting.config.config_context(equal_var=not old_equal_var):
        assert tea_tasting.config._global_config["equal_var"] is not old_equal_var

    assert tea_tasting.config._global_config["equal_var"] is old_equal_var
