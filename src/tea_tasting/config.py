"""Global config."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal


_global_config = {
    "alternative": "two-sided",
    "confidence_level": 0.95,
    "equal_var": False,
    "use_t": True,
}


def get_config(option: str | None = None) -> Any:  # noqa: ANN401
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
    alternative: Literal["two-sided", "less", "greater"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    use_t: bool | None = None,
) -> None:
    """Set global configuration.

    Args:
        alternative: Default alternative hypothesis.
        confidence_level: Default confidence level for the confidence interval.
        equal_var: Defines whether to use the Welch's t-test (`False`) or the standard
            Student's t-test (`True`) by default. The standard Student's t-test assumes
            equal population variances, while Welch's t-test doesn't. Applicable only if
            `use_t` is `True`.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default.
    """
    for param, value in locals().items():
        if value is not None:
            _global_config[param] = value


@contextlib.contextmanager
def config_context(
    alternative: Literal["two-sided", "less", "greater"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    use_t: bool | None = None,
) -> Generator[None, Any, None]:
    """Context manager for configuration.

    Args:
        alternative: Default alternative hypothesis.
        confidence_level: Default confidence level for the confidence interval.
        equal_var: Defines whether to use the Welch's t-test (`False`) or the standard
            Student's t-test (`True`) by default. The standard Student's t-test assumes
            equal population variances, while Welch's t-test doesn't. Applicable only if
            `use_t` is `True`.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default.
    """
    new_config = locals()
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
