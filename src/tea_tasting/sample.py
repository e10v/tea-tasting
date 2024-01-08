"""Generates a sample of data for examples."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import numpy as np
import pandas as pd

import tea_tasting._utils


if TYPE_CHECKING:
    from ibis.expr.types import Table


def sample_users_data(  # noqa: PLR0913
    size: int = 10000,
    covariates: bool = False,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    ratio: float | int = 1,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float | int = 2,
    avg_orders_per_visit: float = 0.25,
    avg_revenue_per_order: float | int = 10,
) -> Table:
    """Generates a sample of data for examples.

    Data mimics what you might encounter in an A/B test for an online store. Each row
    represents an individual user with information about:

    - variant of the test,
    - number of visits by the user,
    - number of orders made by the user,
    - revenue generated from user's orders.

    Optionally, pre-experimantal data can be generated as well.

    Args:
        size: Sample size.
        covariates: If True, genarates pre-experimantal data as the covariates
            in addition to default columns.
        seed: Random seed.
        ratio: Ratio of treatment observations to control observations.
        visits_uplift: Relative visits uplift in the treatment variant.
        orders_uplift: Relative orders uplift in the treatment variant.
        revenue_uplift: Relative revenue uplift in the treatment variant.
        avg_visits: Average number of visits per user.
        avg_orders_per_visit: Average number of orders per visit. Should be less than 1.
        avg_revenue_per_order: Average revenue per order.

    Returns:
        An Ibis Table with the following columns:
            user: User indentificator.
            variant: Variant of the test. 0 is control, 1 is treatment.
            visits: Number of visits.
            orders: Number of orders.
            revenue: Revenue.
            visits_covariate (optional): Number of visits before the experiment.
            orders_covariate (optional): Number of orders before the experiment.
            revenue_covariate (optional): Revenue before the experiment.
    """
    tea_tasting._utils.check_scalar(size, name="size", type_=int, ge=10)
    tea_tasting._utils.check_scalar(ratio, name="ratio", type_=float | int, gt=0)
    tea_tasting._utils.check_scalar(
        visits_uplift, name="visits_uplift", type_=float, gt=-1)
    tea_tasting._utils.check_scalar(
        orders_uplift,
        name="orders_uplift",
        type_=float,
        gt=-1,
        lt=(1 + max(visits_uplift, 0))/avg_orders_per_visit - 1,
    )
    tea_tasting._utils.check_scalar(
        revenue_uplift, name="revenue_uplift", type_=float, gt=-1)
    tea_tasting._utils.check_scalar(
        avg_visits, name="avg_visits", type_=float | int, gt=1)
    tea_tasting._utils.check_scalar(
        avg_orders_per_visit, name="avg_orders_per_visit", type_=float, gt=0, lt=1)
    tea_tasting._utils.check_scalar(
        avg_revenue_per_order, name="avg_revenue_per_order", type_=float, gt=0)

    rng = np.random.default_rng(seed=seed)
    treat = rng.binomial(n=1, p=ratio / (1 + ratio), size=size)

    visits_mult = 1 + visits_uplift*treat
    visits = 1 + rng.poisson(lam=avg_visits*visits_mult - 1, size=size)

    orders_per_visits_mult = (1 + orders_uplift*treat) / (1 + visits_uplift*treat)
    orders_per_visits = rng.beta(
        a=avg_orders_per_visit*orders_per_visits_mult,
        b=1 - avg_orders_per_visit*orders_per_visits_mult,
        size=size,
    )

    orders = rng.binomial(n=visits, p=orders_per_visits, size=size)

    revenue_per_order_mult = (1 + revenue_uplift*treat) / (1 + orders_uplift*treat)
    revenue_per_order = rng.lognormal(
        mean=np.log(avg_revenue_per_order * revenue_per_order_mult) - 0.5,
        size=size,
    )

    revenue = orders * revenue_per_order

    data = pd.DataFrame({
        "user": np.arange(size),
        "variant": treat,
        "visits": visits,
        "orders": orders,
        "revenue": revenue,
    })

    if covariates:
        visits_covariate = rng.poisson(lam=visits / visits_mult, size=size)
        orders_per_visits_covariate = orders_per_visits / orders_per_visits_mult

        orders_covariate = rng.binomial(
            n=visits_covariate,
            p=orders_per_visits_covariate,
            size=size,
        )

        revenue_per_order_covariate = rng.lognormal(
            mean=np.log(revenue_per_order/revenue_per_order_mult) - 0.125,
            sigma=0.5,
            size=size,
        )

        revenue_covariate = orders_covariate * revenue_per_order_covariate

        data = data.assign(
            visits_covariate=visits_covariate,
            orders_covariate=orders_covariate,
            revenue_covariate=revenue_covariate,
        )

    return ibis.memtable(data)
