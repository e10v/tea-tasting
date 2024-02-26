from __future__ import annotations

import ibis.expr.types
import pandas as pd

import tea_tasting.datasets


def test_make_users_data_default():
    n_users = 100
    users_data = tea_tasting.datasets.make_users_data(seed=42, n_users=n_users)
    assert isinstance(users_data, ibis.expr.types.Table)
    assert users_data.columns == ["user", "variant", "visits", "orders", "revenue"]

    data = users_data.to_pandas()
    assert len(data) == n_users
    assert data["user"].drop_duplicates().count() == n_users
    assert data["variant"].drop_duplicates().count() == 2
    assert data["visits"].min() > 0
    assert data["orders"].min() >= 0
    assert data["orders"].sub(data["visits"]).min() <= 0
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_make_users_data_covariates():
    n_users = 100
    users_data = tea_tasting.datasets.make_users_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(users_data, ibis.expr.types.Table)
    assert users_data.columns == [
        "user", "variant", "visits", "orders", "revenue",
        "visits_covariate", "orders_covariate", "revenue_covariate",
    ]

    data = users_data.to_pandas()
    assert data["visits_covariate"].min() >= 0
    assert data["orders_covariate"].min() >= 0
    assert data["orders_covariate"].sub(data["visits_covariate"]).min() <= 0
    assert data["revenue_covariate"].min() >= 0
    assert (
        data["revenue_covariate"].gt(0)
        .eq(data["orders_covariate"].gt(0))
        .astype(int).min()
    ) == 1


def test_make_users_data_pandas():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(
        to_pandas=True, seed=42, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == ["user", "variant", "visits", "orders", "revenue"]


def test_make_visits_data_default():
    n_users = 100
    visits_data = tea_tasting.datasets.make_visits_data(seed=42, n_users=n_users)
    assert isinstance(visits_data, ibis.expr.types.Table)
    assert visits_data.columns == ["user", "variant", "visits", "orders", "revenue"]

    data = visits_data.to_pandas()
    assert len(data) > n_users
    assert data["user"].drop_duplicates().count() == n_users
    assert data["variant"].drop_duplicates().count() == 2
    assert data["visits"].min() == 1
    assert data["visits"].max() == 1
    assert data["orders"].min() >= 0
    assert data["orders"].sub(data["visits"]).min() <= 0
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_make_visits_data_covariates():
    n_users = 100
    visits_data = tea_tasting.datasets.make_visits_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(visits_data, ibis.expr.types.Table)
    assert visits_data.columns == [
        "user", "variant", "visits", "orders", "revenue",
        "visits_covariate", "orders_covariate", "revenue_covariate",
    ]

    data = visits_data.to_pandas()
    assert data["visits_covariate"].min() >= 0
    assert data["orders_covariate"].min() >= 0
    assert data["orders_covariate"].sub(data["visits_covariate"]).min() <= 0
    assert data["revenue_covariate"].min() >= 0
    assert (
        data["revenue_covariate"].gt(0)
        .eq(data["orders_covariate"].gt(0))
        .astype(int).min()
    ) == 1


def test_make_visits_data_pandas():
    n_users = 100
    data = tea_tasting.datasets.make_visits_data(
        to_pandas=True, seed=42, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == ["user", "variant", "visits", "orders", "revenue"]
