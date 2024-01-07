"""Generates sample data for examples."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from ibis.expr.types import Table


def sample_users_data(  # noqa: PLR0913
    size: int = 5000,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    ratio: float = 1.0,
    visits_uplift: float = 0.0,
    orders_uplift: float = 0.1,
    revenue_uplift: float = 0.1,
    avg_visits: float = 2.0,
    avg_orders_per_visit: float = 0.5,
    avg_revenue_per_order: float = 10.0,
) -> Table:
    rng = np.random.default_rng(seed=seed)

    treat = rng.binomial(
        n=1,
        p=ratio / (1 + ratio),
        size=size,
    )

    visits_mult = 1 + visits_uplift*treat
    visits = 1 + rng.poisson(
        lam=avg_visits*visits_mult - 1,
        size=size,
    )

    orders_per_visits_mult = (1 + orders_uplift*treat) / (1 + visits_uplift*treat)
    orders_per_visits = rng.beta(
        a=avg_orders_per_visit*orders_per_visits_mult,
        b=1 - avg_orders_per_visit*orders_per_visits_mult,
        size=size,
    )

    orders = rng.binomial(
        n=visits,
        p=orders_per_visits,
        size=size,
    )

    revenue_per_order_mult = (1 + revenue_uplift*treat) / (1 + orders_uplift*treat)
    revenue_per_order = rng.lognormal(
        mean=np.log(avg_revenue_per_order * revenue_per_order_mult) - 0.5,
        size=size,
    )

    revenue = orders * revenue_per_order

    return ibis.memtable(pd.DataFrame({
        "user": np.arange(size),
        "variant": treat,
        "visits": visits,
        "orders": orders,
        "revenue": revenue,
    }))
