"""Global config."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal


_global_config = {
    "alternative": "two-sided",
    "confidence_level": 0.95,
    "equal_var": False,
    "use_t": True,
}


def get_config(option: str | None = None) -> Any:
    """Get global configuration.

    Args:
        option: The option name.

    Returns:
        The value of the option if it's not None,
        or a dictionary with all options otherwise.
    """
    if option is not None:
        return _global_config[option]
    return _global_config.copy()


def set_config(
    *,
    alternative: Literal["two-sided", "greater", "less"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    use_t: bool | None = None,
    **kwargs: Any,
) -> None:
    """Set global configuration.

    Args:
        alternative: Default alternative hypothesis. Default is "two-sided".
        confidence_level: Default confidence level for the confidence interval.
            Default is 0.95.
        equal_var: Defines whether equal variance is assumed. If True,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is False.
        use_t: Defines whether to use the Student's t-distribution (True) or
            the Normal distribution (False) by default. Default is True.
        kwargs: User-defined global parameters.
    """
    params = {k: v for k, v in locals().items() if k != "kwargs"} | kwargs
    for name, value in params.items():
        if value is not None:
            _global_config[name] = tea_tasting.utils.auto_check(value, name)


@contextlib.contextmanager
def config_context(
    *,
    alternative: Literal["two-sided", "greater", "less"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    use_t: bool | None = None,
    **kwargs: Any,
) -> Generator[None, Any, None]:
    """Context manager for configuration.

    Args:
        alternative: Default alternative hypothesis. Default is "two-sided".
        confidence_level: Default confidence level for the confidence interval.
            Default is 0.95.
        equal_var: Defines whether equal variance is assumed. If True,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is False.
        use_t: Defines whether to use the Student's t-distribution (True) or
            the Normal distribution (False) by default. Default is True.
        kwargs: User-defined global parameters.
    """
    new_config = locals()
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
