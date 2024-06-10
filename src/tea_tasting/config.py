"""Global configuration."""

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
    "n_resamples": 10_000,
    "use_t": True,
}


def get_config(option: str | None = None) -> Any:
    """Retrieve the current settings of the global configuration.

    Args:
        option: The option name.

    Returns:
        The specified option value if its name is provided,
            or a dictionary containing all options otherwise.

    Examples:
        ```python
        import tea_tasting as tt


        tt.get_config("equal_var")
        #> False
        ```
    """
    if option is not None:
        return _global_config[option]
    return _global_config.copy()


def set_config(
    *,
    alternative: Literal["two-sided", "greater", "less"] | None = None,
    confidence_level: float | None = None,
    equal_var: bool | None = None,
    n_resamples: int | None = None,
    use_t: bool | None = None,
    **kwargs: Any,
) -> None:
    """Update the global configuration with specified settings.

    Args:
        alternative: Default alternative hypothesis. Default is `"two-sided"`.
        confidence_level: Default confidence level for the confidence interval.
            Default is `0.95`.
        equal_var: Defines whether equal variance is assumed. If `True`,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is `False`.
        n_resamples: The number of resamples performed to form the bootstrap
            distribution of a statistic. Default is `10_000`.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default. Default is `True`.
        kwargs: User-defined global parameters.

    Alternative hypothesis options:
        - `"two-sided"`: the means are unequal,
        - `"greater"`: the mean in the treatment variant is greater than the mean
            in the control variant,
        - `"less"`: the mean in the treatment variant is less than the mean
            in the control variant.

    Examples:
        ```python
        import tea_tasting as tt


        tt.set_config(equal_var=True, use_t=False)

        experiment = tt.Experiment(
            sessions_per_user=tt.Mean("sessions"),
            orders_per_session=tt.RatioOfMeans("orders", "sessions"),
            orders_per_user=tt.Mean("orders"),
            revenue_per_user=tt.Mean("revenue"),
        )

        experiment.metrics["orders_per_user"]
        #> Mean(value='orders', covariate=None, alternative='two-sided',
        #> confidence_level=0.95, equal_var=True, use_t=False)
        ```
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
    n_resamples: int | None = None,
    use_t: bool | None = None,
    **kwargs: Any,
) -> Generator[None, Any, None]:
    """A context manager that temporarily modifies the global configuration.

    Args:
        alternative: Default alternative hypothesis. Default is `"two-sided"`.
        confidence_level: Default confidence level for the confidence interval.
            Default is `0.95`.
        equal_var: Defines whether equal variance is assumed. If `True`,
            pooled variance is used for the calculation of the standard error
            of the difference between two means. Default is `False`.
        n_resamples: The number of resamples performed to form the bootstrap
            distribution of a statistic. Default is `10_000`.
        use_t: Defines whether to use the Student's t-distribution (`True`) or
            the Normal distribution (`False`) by default. Default is `True`.
        kwargs: User-defined global parameters.

    Alternative hypothesis options:
        - `"two-sided"`: the means are unequal,
        - `"greater"`: the mean in the treatment variant is greater than the mean
            in the control variant,
        - `"less"`: the mean in the treatment variant is less than the mean
            in the control variant.

    Examples:
        ```python
        import tea_tasting as tt


        with tt.config_context(equal_var=True, use_t=False):
            experiment = tt.Experiment(
                sessions_per_user=tt.Mean("sessions"),
                orders_per_session=tt.RatioOfMeans("orders", "sessions"),
                orders_per_user=tt.Mean("orders"),
                revenue_per_user=tt.Mean("revenue"),
            )

        experiment.metrics["orders_per_user"]
        #> Mean(value='orders', covariate=None, alternative='two-sided',
        #> confidence_level=0.95, equal_var=True, use_t=False)
        ```
    """
    new_config = {k: v for k, v in locals().items() if k != "kwargs"} | kwargs
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
