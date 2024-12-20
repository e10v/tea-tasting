"""Example datasets."""
# ruff: noqa: PLR0913

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


def make_users_data(
    *,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    sessions_uplift: float | int = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_sessions: float | int = 2,
    avg_orders_per_session: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> pd.DataFrame:
    """Generate simulated data for A/B testing scenarios.

    Data mimics what you might encounter in an A/B test for an online store,
    with a user-level randomization. Each row represents an individual user
    with information about:

    - `user`: User identifier.
    - `variant`: Variant of the test. 0 is control, 1 is treatment.
    - `sessions`: Number of user's sessions.
    - `orders`: Number of user's orders.
    - `revenue`: Revenue generated by the user.

    Optionally, pre-experimental data can be generated as well:

    - `sessions_covariate`: Number of user's sessions
        before the experiment.
    - `orders_covariate`: Number of user's orders before the experiment.
    - `revenue_covariate`: Revenue generated by the user
        before the experiment.

    Args:
        covariates: If `True`, generates pre-experimental data as the covariates
            in addition to default columns.
        seed: Random seed.
        n_users: Number of users.
        ratio: Ratio of the number of users in treatment relative to control.
        sessions_uplift: Sessions uplift in the treatment variant, relative to control.
        orders_uplift: Orders uplift in the treatment variant, relative to control.
        revenue_uplift: Revenue uplift in the treatment variant, relative to control.
        avg_sessions: Average number of sessions per user.
        avg_orders_per_session: Average number of orders per session.
            Should be less than `1`.
        avg_revenue_per_order: Average revenue per order.

    Returns:
        Simulated data for A/B testing scenarios.

    Examples:
        ```python
        import tea_tasting as tt


        data = tt.make_users_data(seed=42)
        data
        #>       user  variant  sessions  orders    revenue
        #> 0        0        1         2       1   9.166147
        #> 1        1        0         2       1   6.434079
        #> 2        2        1         2       1   7.943873
        #> 3        3        1         2       1  15.928675
        #> 4        4        0         1       1   7.136917
        #> ...    ...      ...       ...     ...        ...
        #> 3995  3995        0         2       0   0.000000
        #> 3996  3996        0         2       0   0.000000
        #> 3997  3997        0         3       0   0.000000
        #> 3998  3998        0         1       0   0.000000
        #> 3999  3999        0         5       2  17.162459
        #>
        #> [4000 rows x 5 columns]
        ```

        With covariates:

        ```python
        data = tt.make_users_data(seed=42, covariates=True)
        data
        #>       user  variant  sessions  orders    revenue  sessions_covariate  orders_covariate  revenue_covariate
        #> 0        0        1         2       1   9.166147                   3                 2          19.191712
        #> 1        1        0         2       1   6.434079                   4                 1           2.770749
        #> 2        2        1         2       1   7.943873                   4                 2          22.568422
        #> 3        3        1         2       1  15.928675                   1                 0           0.000000
        #> 4        4        0         1       1   7.136917                   1                 1          13.683796
        #> ...    ...      ...       ...     ...        ...                 ...               ...                ...
        #> 3995  3995        0         2       0   0.000000                   1                 0           0.000000
        #> 3996  3996        0         2       0   0.000000                   3                 1          13.517967
        #> 3997  3997        0         3       0   0.000000                   2                 0           0.000000
        #> 3998  3998        0         1       0   0.000000                   1                 0           0.000000
        #> 3999  3999        0         5       2  17.162459                   5                 0           0.000000
        #>
        #> [4000 rows x 8 columns]
        ```
    """  # noqa: E501
    return _make_data(
        covariates=covariates,
        seed=seed,
        n_users=n_users,
        ratio=ratio,
        sessions_uplift=sessions_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_sessions=avg_sessions,
        avg_orders_per_session=avg_orders_per_session,
        avg_revenue_per_order=avg_revenue_per_order,
        explode_sessions=False,
    )


def make_sessions_data(
    *,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    sessions_uplift: float | int = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_sessions: float | int = 2,
    avg_orders_per_session: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> pd.DataFrame:
    """Generate simulated user data for A/B testing scenarios.

    Data mimics what you might encounter in an A/B test for an online store,
    with a user-level randomization. Each row represents a user's session
    with information about:

    - `user`: User identifier.
    - `variant`: Variant of the test. 0 is control, 1 is treatment.
    - `sessions`: Number of user's sessions.
    - `orders`: Number of user's orders.
    - `revenue`: Revenue generated by the user.

    Optionally, pre-experimental data can be generated as well:

    - `sessions_covariate`: Number of user's sessions
        before the experiment.
    - `orders_covariate`: Number of user's orders before the experiment.
    - `revenue_covariate`: Revenue generated by the user
        before the experiment.

    Args:
        covariates: If `True`, generates pre-experimental data as the covariates
            in addition to default columns.
        seed: Random seed.
        n_users: Number of users.
        ratio: Ratio of the number of users in treatment relative to control.
        sessions_uplift: Sessions uplift in the treatment variant, relative to control.
        orders_uplift: Orders uplift in the treatment variant, relative to control.
        revenue_uplift: Revenue uplift in the treatment variant, relative to control.
        avg_sessions: Average number of sessions per user.
        avg_orders_per_session: Average number of orders per session.
            Should be less than `1`.
        avg_revenue_per_order: Average revenue per order.

    Returns:
        Simulated data for A/B testing scenarios.

    Examples:
        ```python
        import tea_tasting as tt


        data = tt.make_sessions_data(seed=42)
        data
        #>       user  variant  sessions  orders    revenue
        #> 0        0        1         1       1   5.887178
        #> 1        0        1         1       1   6.131080
        #> 2        1        0         1       1   2.614675
        #> 3        1        0         1       1  12.296075
        #> 4        2        1         1       1  11.573409
        #> ...    ...      ...       ...     ...        ...
        #> 7953  3999        0         1       1  23.634941
        #> 7954  3999        0         1       0   0.000000
        #> 7955  3999        0         1       1   2.396078
        #> 7956  3999        0         1       1  24.538111
        #> 7957  3999        0         1       0   0.000000
        #>
        #> [7958 rows x 5 columns]
        ```

        With covariates:

        ```python
        data = tt.make_sessions_data(seed=42, covariates=True)
        data
        #>       user  variant  sessions  orders    revenue  sessions_covariate  orders_covariate  revenue_covariate
        #> 0        0        1         1       1   5.887178                 1.5               0.5           1.236732
        #> 1        0        1         1       1   6.131080                 1.5               0.5           1.236732
        #> 2        1        0         1       1   2.614675                 0.0               0.0           0.000000
        #> 3        1        0         1       1  12.296075                 0.0               0.0           0.000000
        #> 4        2        1         1       1  11.573409                 1.5               1.5          12.324434
        #> ...    ...      ...       ...     ...        ...                 ...               ...                ...
        #> 7953  3999        0         1       1  23.634941                 0.2               0.0           0.000000
        #> 7954  3999        0         1       0   0.000000                 0.2               0.0           0.000000
        #> 7955  3999        0         1       1   2.396078                 0.2               0.0           0.000000
        #> 7956  3999        0         1       1  24.538111                 0.2               0.0           0.000000
        #> 7957  3999        0         1       0   0.000000                 0.2               0.0           0.000000
        #>
        #> [7958 rows x 8 columns]
        ```
    """  # noqa: E501
    return _make_data(
        covariates=covariates,
        seed=seed,
        n_users=n_users,
        ratio=ratio,
        sessions_uplift=sessions_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_sessions=avg_sessions,
        avg_orders_per_session=avg_orders_per_session,
        avg_revenue_per_order=avg_revenue_per_order,
        explode_sessions=True,
    )


def _make_data(
    *,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    sessions_uplift: float | int = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_sessions: float | int = 2,
    avg_orders_per_session: float = 0.25,
    avg_revenue_per_order: float | int = 10,
    explode_sessions: bool = False,
) -> pd.DataFrame:
    _check_params(
        n_users=n_users,
        ratio=ratio,
        sessions_uplift=sessions_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_sessions=avg_sessions,
        avg_orders_per_session=avg_orders_per_session,
        avg_revenue_per_order=avg_revenue_per_order,
    )

    rng = np.random.default_rng(seed=seed)
    user = np.arange(n_users)
    variant = rng.binomial(n=1, p=ratio / (1 + ratio), size=n_users)
    sessions_mult = 1 + sessions_uplift*variant
    sessions = 1 + rng.poisson(lam=avg_sessions*sessions_mult - 1, size=n_users)
    size = n_users
    orders_per_sessions_sample_size = 1  # Parameter of Beta distribution (alpha+beta).
    revenue_log_scale = 0.5  # Parameter of log-normal distribution.

    if explode_sessions:
        user = np.repeat(user, sessions)
        sessions = 1
        size = len(user)
        revenue_log_scale = np.sqrt(np.log(
            1 + avg_sessions*(np.exp(revenue_log_scale**2) - 1)))

    orders_per_sessions_mult = (1 + orders_uplift*variant) / (
        1 + sessions_uplift*variant)
    orders_per_sessions = rng.beta(
        a=avg_orders_per_session * orders_per_sessions_mult
            * orders_per_sessions_sample_size,
        b=(1 - avg_orders_per_session*orders_per_sessions_mult)
            * orders_per_sessions_sample_size,
        size=n_users,
    )
    orders = rng.binomial(n=sessions, p=orders_per_sessions[user], size=size)

    revenue_per_order_mult = (1 + revenue_uplift*variant) / (1 + orders_uplift*variant)
    revenue_per_order = rng.lognormal(
        mean=(
            np.log(avg_revenue_per_order * revenue_per_order_mult[user])
            - revenue_log_scale * revenue_log_scale / 2
        ),
        sigma=revenue_log_scale,
        size=size,
    )

    revenue = orders * revenue_per_order

    data = pd.DataFrame({
        "user": user,
        "variant": variant[user].astype(np.uint8),
        "sessions": sessions,
        "orders": orders,
        "revenue": revenue,
    })

    if covariates:
        sessions_covariate = rng.poisson(lam=sessions / sessions_mult[user], size=size)
        orders_per_sessions_covariate = orders_per_sessions / orders_per_sessions_mult

        orders_covariate = rng.binomial(
            n=sessions_covariate,
            p=orders_per_sessions_covariate[user],
            size=size,
        )

        revenue_per_order_covariate = rng.lognormal(
            mean=(
                np.log(revenue_per_order / revenue_per_order_mult[user])
                - revenue_log_scale * revenue_log_scale / 2
            ),
            sigma=revenue_log_scale,
            size=size,
        )

        revenue_covariate = orders_covariate * revenue_per_order_covariate

        if explode_sessions:
            sessions_covariate = _avg_by_groups(sessions_covariate, user)
            orders_covariate = _avg_by_groups(orders_covariate, user)
            revenue_covariate = _avg_by_groups(revenue_covariate, user)

        data = data.assign(
            sessions_covariate=sessions_covariate,
            orders_covariate=orders_covariate,
            revenue_covariate=revenue_covariate,
        )

    return data


def _check_params(
    *,
    n_users: int,
    ratio: float | int,
    sessions_uplift: float | int,
    orders_uplift: float,
    revenue_uplift: float,
    avg_sessions: float | int,
    avg_orders_per_session: float,
    avg_revenue_per_order: float | int,
) -> None:
    tea_tasting.utils.check_scalar(n_users, name="n_users", typ=int, ge=10)
    tea_tasting.utils.check_scalar(ratio, name="ratio", typ=float | int, gt=0)
    tea_tasting.utils.check_scalar(
        sessions_uplift, name="sessions_uplift", typ=float | int, gt=1/avg_sessions - 1)
    tea_tasting.utils.check_scalar(
        orders_uplift,
        name="orders_uplift",
        typ=float | int,
        gt=-1,
        lt=(1 + sessions_uplift)/avg_orders_per_session - 1,
    )
    tea_tasting.utils.check_scalar(
        revenue_uplift, name="revenue_uplift", typ=float | int, gt=-1)
    tea_tasting.utils.check_scalar(
        avg_sessions, name="avg_sessions", typ=float | int, gt=1)
    tea_tasting.utils.check_scalar(
        avg_orders_per_session, name="avg_orders_per_session", typ=float, gt=0, lt=1)
    tea_tasting.utils.check_scalar(
        avg_revenue_per_order, name="avg_revenue_per_order", typ=float | int, gt=0)


def _avg_by_groups(
    values: npt.NDArray[np.number[Any]],
    groups: npt.NDArray[np.number[Any]],
) -> npt.NDArray[np.number[Any]]:
    return np.concatenate([
        np.full(v.shape, v.mean())
        for v in np.split(values, np.unique(groups, return_index=True)[1])
        if len(v) > 0
    ])
