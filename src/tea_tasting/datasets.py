"""Generates a sample of data for examples."""
# ruff: noqa: PLR0913

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import ibis
import numpy as np
import pandas as pd

import tea_tasting.utils


if TYPE_CHECKING:
    from typing import Any, Literal

    import ibis.expr.types  # noqa: TCH004
    import numpy.typing as npt


@overload
def make_users_data(
    *,
    to_pandas: Literal[False] = False,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> ibis.expr.types.Table:
    ...

@overload
def make_users_data(
    *,
    to_pandas: Literal[True] = True,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> pd.DataFrame:
    ...

def make_users_data(
    *,
    to_pandas: bool = False,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> ibis.expr.types.Table | pd.DataFrame:
    """Generates a sample of data for examples.

    Data mimics what you might encounter in an A/B test for an online store,
    with user-level randomization. Each row represents an individual user
    with information about:

    - user identifier,
    - variant of the test,
    - number of visits by the user,
    - number of orders made by the user,
    - revenue generated from user's orders.

    Optionally, pre-experimental data can be generated as well.

    Args:
        to_pandas: If True, return Pandas DataFrame instead if Ibis Table.
        covariates: If True, generates pre-experimental data as the covariates
            in addition to default columns.
        seed: Random seed.
        n_users: Number of users.
        ratio: Ratio of treatment observations to control observations.
        visits_uplift: Relative visits uplift in the treatment variant.
        orders_uplift: Relative orders uplift in the treatment variant.
        revenue_uplift: Relative revenue uplift in the treatment variant.
        avg_visits: Average number of visits per user.
        avg_orders_per_visit: Average number of orders per visit. Should be less than 1.
        avg_revenue_per_order: Average revenue per order.

    Returns:
        An Ibis Table or a Pandas DataFrame with the following columns:
            user: User identifier.
            variant: Variant of the test. 0 is control, 1 is treatment.
            visits: Number of visits.
            orders: Number of orders.
            revenue: Revenue.
            visits_covariate (optional): Number of visits before the experiment.
            orders_covariate (optional): Number of orders before the experiment.
            revenue_covariate (optional): Revenue before the experiment.
    """
    return _make_data(
        to_pandas=to_pandas,
        covariates=covariates,
        seed=seed,
        n_users=n_users,
        ratio=ratio,
        visits_uplift=visits_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_visits=avg_visits,
        avg_orders_per_visit=avg_orders_per_visit,
        avg_revenue_per_order=avg_revenue_per_order,
        explode_visits=False,
    )


@overload
def make_visits_data(
    *,
    to_pandas: Literal[False] = False,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> ibis.expr.types.Table:
    ...

@overload
def make_visits_data(
    *,
    to_pandas: Literal[True] = True,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> pd.DataFrame:
    ...

def make_visits_data(
    *,
    to_pandas: bool = False,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> ibis.expr.types.Table | pd.DataFrame:
    """Generates a sample of data for examples.

    Data mimics what you might encounter in an A/B test for an online store,
    with user-level randomization. Each row represents a user's visit
    with information about:

    - user identifier,
    - variant of the test,
    - number of visits by the user,
    - number of orders made by the user,
    - revenue generated from user's orders.

    Optionally, pre-experimental data can be generated as well.

    Args:
        to_pandas: If True, return Pandas DataFrame instead if Ibis Table.
        covariates: If True, generates pre-experimental data as the covariates
            in addition to default columns.
        seed: Random seed.
        n_users: Number of users.
        ratio: Ratio of treatment observations to control observations.
        visits_uplift: Relative visits uplift in the treatment variant.
        orders_uplift: Relative orders uplift in the treatment variant.
        revenue_uplift: Relative revenue uplift in the treatment variant.
        avg_visits: Average number of visits per user.
        avg_orders_per_visit: Average number of orders per visit. Should be less than 1.
        avg_revenue_per_order: Average revenue per order.

    Returns:
        An Ibis Table with the following columns:
            user: User identifier.
            variant: Variant of the test. 0 is control, 1 is treatment.
            visits: Number of visits.
            orders: Number of orders.
            revenue: Revenue.
            visits_covariate (optional): Number of visits before the experiment.
            orders_covariate (optional): Number of orders before the experiment.
            revenue_covariate (optional): Revenue before the experiment.
    """
    return _make_data(
        to_pandas=to_pandas,
        covariates=covariates,
        seed=seed,
        n_users=n_users,
        ratio=ratio,
        visits_uplift=visits_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_visits=avg_visits,
        avg_orders_per_visit=avg_orders_per_visit,
        avg_revenue_per_order=avg_revenue_per_order,
        explode_visits=True,
    )


def _make_data(
    to_pandas: bool = False,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    n_users: int = 4000,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
    explode_visits: bool = False,
) -> ibis.expr.types.Table | pd.DataFrame:
    _check_params(
        n_users=n_users,
        ratio=ratio,
        visits_uplift=visits_uplift,
        orders_uplift=orders_uplift,
        revenue_uplift=revenue_uplift,
        avg_visits=avg_visits,
        avg_orders_per_visit=avg_orders_per_visit,
        avg_revenue_per_order=avg_revenue_per_order,
    )

    rng = np.random.default_rng(seed=seed)
    user = np.arange(n_users)
    variant = rng.binomial(n=1, p=ratio / (1 + ratio), size=n_users)
    visits_mult = 1 + visits_uplift*variant
    visits = 1 + rng.poisson(lam=avg_visits*visits_mult - 1, size=n_users)
    size = n_users
    orders_per_visits_sample_size = 1  # Parameter of Beta distribution (alpha + beta).
    revenue_log_scale = 0.5  # Parameter of log-normal distribution.

    if explode_visits:
        user = np.repeat(user, visits)
        visits = 1
        size = len(user)
        revenue_log_scale = np.sqrt(np.log(
            1 + avg_visits*(np.exp(revenue_log_scale**2) - 1)))

    orders_per_visits_mult = (1 + orders_uplift*variant) / (1 + visits_uplift*variant)
    orders_per_visits = rng.beta(
        a=avg_orders_per_visit * orders_per_visits_mult
            * orders_per_visits_sample_size,
        b=(1 - avg_orders_per_visit*orders_per_visits_mult)
            * orders_per_visits_sample_size,
        size=n_users,
    )
    orders = rng.binomial(n=visits, p=orders_per_visits[user], size=size)

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
        "visits": visits,
        "orders": orders,
        "revenue": revenue,
    })

    if covariates:
        visits_covariate = rng.poisson(lam=visits / visits_mult[user], size=size)
        orders_per_visits_covariate = orders_per_visits / orders_per_visits_mult

        orders_covariate = rng.binomial(
            n=visits_covariate,
            p=orders_per_visits_covariate[user],
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

        if explode_visits:
            visits_covariate = _avg_by_groups(visits_covariate, user)
            orders_covariate = _avg_by_groups(orders_covariate, user)
            revenue_covariate = _avg_by_groups(revenue_covariate, user)

        data = data.assign(
            visits_covariate=visits_covariate,
            orders_covariate=orders_covariate,
            revenue_covariate=revenue_covariate,
        )

    if to_pandas:
        return data

    con = ibis.pandas.connect()
    return con.create_table("users_data", data)


def _check_params(
    n_users: int,
    ratio: float | int,
    visits_uplift: float,
    orders_uplift: float,
    revenue_uplift: float,
    avg_visits: float | int,
    avg_orders_per_visit: float,
    avg_revenue_per_order: float | int,
) -> None:
    tea_tasting.utils.check_scalar(n_users, name="n_users", typ=int, ge=10)
    tea_tasting.utils.check_scalar(ratio, name="ratio", typ=float | int, gt=0)
    tea_tasting.utils.check_scalar(
        visits_uplift, name="visits_uplift", typ=float, gt=1/avg_visits - 1)
    tea_tasting.utils.check_scalar(
        orders_uplift,
        name="orders_uplift",
        typ=float,
        gt=-1,
        lt=(1 + visits_uplift)/avg_orders_per_visit - 1,
    )
    tea_tasting.utils.check_scalar(
        revenue_uplift, name="revenue_uplift", typ=float, gt=-1)
    tea_tasting.utils.check_scalar(
        avg_visits, name="avg_visits", typ=float | int, gt=1)
    tea_tasting.utils.check_scalar(
        avg_orders_per_visit, name="avg_orders_per_visit", typ=float, gt=0, lt=1)
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
