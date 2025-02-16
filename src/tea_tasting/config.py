"""Global configuration."""
# ruff: noqa: PLR0913

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, overload

import tea_tasting.utils


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Literal


_global_config: dict[str, object] = {
    "alpha": 0.05,
    "alternative": "two-sided",
    "confidence_level": 0.95,
    "equal_var": False,
    "n_obs": None,
    "n_resamples": 10_000,
    "power": 0.8,
    "ratio": 1,
    "use_t": True,
}


@overload
def get_config(option: Literal["alpha"]) -> float:
    ...

@overload
def get_config(option: Literal["alternative"]) -> str:
    ...

@overload
def get_config(option: Literal["confidence_level"]) -> float:
    ...

@overload
def get_config(option: Literal["equal_var"]) -> bool:
    ...

@overload
def get_config(option: Literal["n_obs"]) -> int | Sequence[int] | None:
    ...

@overload
def get_config(option: Literal["n_resamples"]) -> str:
    ...

@overload
def get_config(option: Literal["power"]) -> float:
    ...

@overload
def get_config(option: Literal["ratio"]) -> float | int:
    ...

@overload
def get_config(option: Literal["use_t"]) -> bool:
    ...

@overload
def get_config(option: str) -> object:
    ...

@overload
def get_config(option: None = None) -> dict[str, object]:
    ...

def get_config(option: str | None = None) -> object:
    """Retrieve the current settings of the global configuration.

    Args:
        option: The option name.

    Returns:
        The specified option value if its name is provided,
            or a dictionary containing all options otherwise.

    Examples:
        ```pycon
        >>> import tea_tasting as tt

        >>> print(tt.get_config("equal_var"))
        False

        ```
    """
    if option is not None:
        return _global_config[option]
    return _global_config.copy()


def set_config(
    *,
    alpha: float | None = None,
    alternative: Literal["two-sided", "greater", "less"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    n_obs: int | Sequence[int] | None = None,
    n_resamples: int | None = None,
    power: float | None = None,
    ratio: float | int | None = None,
    use_t: bool | None = None,
    **kwargs: object,
) -> None:
    """Update the global configuration with specified settings.

    Args:
        alpha: Significance level. Default is 0.05.
        alternative: Alternative hypothesis:

            - `"two-sided"`: the means are unequal,
            - `"greater"`: the mean in the treatment variant is greater than the mean
                in the control variant,
            - `"less"`: the mean in the treatment variant is less than the mean
                in the control variant.

            Default is `"two-sided"`.

        confidence_level: Confidence level for the confidence interval.
            Default is `0.95`.
        equal_var: Defines whether equal variance is assumed. If `True`,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is `False`.
        n_obs: Number of observations in the control and in the treatment together.
            Default is `None`.
        n_resamples: The number of resamples performed to form the bootstrap
            distribution of a statistic. Default is `10_000`.
        power: Statistical power. Default is 0.8.
        ratio: Ratio of the number of observations in the treatment
            relative to the control. Default is 1.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default. Default is `True`.
        **kwargs: User-defined global parameters.

    Examples:
        ```pycon
        >>> import tea_tasting as tt

        >>> tt.set_config(equal_var=True, use_t=False)
        >>> experiment = tt.Experiment(
        ...     sessions_per_user=tt.Mean("sessions"),
        ...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        ...     orders_per_user=tt.Mean("orders"),
        ...     revenue_per_user=tt.Mean("revenue"),
        ... )
        >>> tt.set_config(equal_var=False, use_t=True)
        >>> print(experiment.metrics["orders_per_user"])
        Mean(value='orders', covariate=None, alternative='two-sided', confidence_level=0.95, equal_var=True, use_t=False, alpha=0.05, ratio=1, power=0.8, effect_size=None, rel_effect_size=None, n_obs=None)

        ```
    """  # noqa: E501
    params = {k: v for k, v in locals().items() if k != "kwargs"} | kwargs
    for name, value in params.items():
        if value is not None:
            _global_config[name] = tea_tasting.utils.auto_check(value, name)


@contextlib.contextmanager
def config_context(
    *,
    alpha: float | None = None,
    alternative: Literal["two-sided", "greater", "less"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    n_obs: int | Sequence[int] | None = None,
    n_resamples: int | None = None,
    power: float | None = None,
    ratio: float | int | None = None,
    use_t: bool | None = None,
    **kwargs: object,
) -> Iterator[object]:
    """A context manager that temporarily modifies the global configuration.

    Args:
        alpha: Significance level. Default is 0.05.
        alternative: Alternative hypothesis:

            - `"two-sided"`: the means are unequal,
            - `"greater"`: the mean in the treatment variant is greater than the mean
                in the control variant,
            - `"less"`: the mean in the treatment variant is less than the mean
                in the control variant.

            Default is `"two-sided"`.

        confidence_level: Confidence level for the confidence interval.
            Default is `0.95`.
        equal_var: Defines whether equal variance is assumed. If `True`,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is `False`.
        n_obs: Number of observations in the control and in the treatment together.
            Default is `None`.
        n_resamples: The number of resamples performed to form the bootstrap
            distribution of a statistic. Default is `10_000`.
        power: Statistical power. Default is 0.8.
        ratio: Ratio of the number of observations in the treatment
            relative to the control. Default is 1.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default. Default is `True`.
        **kwargs: User-defined global parameters.

    Examples:
        ```pycon
        >>> import tea_tasting as tt

        >>> with tt.config_context(equal_var=True, use_t=False):
        ...     experiment = tt.Experiment(
        ...         sessions_per_user=tt.Mean("sessions"),
        ...         orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        ...         orders_per_user=tt.Mean("orders"),
        ...         revenue_per_user=tt.Mean("revenue"),
        ...     )
        >>> print(experiment.metrics["orders_per_user"])
        Mean(value='orders', covariate=None, alternative='two-sided', confidence_level=0.95, equal_var=True, use_t=False, alpha=0.05, ratio=1, power=0.8, effect_size=None, rel_effect_size=None, n_obs=None)

        ```
    """  # noqa: E501
    new_config = {k: v for k, v in locals().items() if k != "kwargs"} | kwargs
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        _global_config.update(**old_config)
