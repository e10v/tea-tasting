from __future__ import annotations

import ibis.expr.types
import pandas as pd

import tea_tasting.datasets


def test_make_users_data_default():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(seed=42, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert len(data) == n_users
    assert data["user"].drop_duplicates().count() == n_users
    assert data["variant"].drop_duplicates().count() == 2
    assert data["sessions"].min() > 0
    assert data["orders"].min() >= 0
    assert data["orders"].sub(data["sessions"]).min() <= 0
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_make_users_data_covariates():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    ]
    assert data["sessions_covariate"].min() >= 0
    assert data["orders_covariate"].min() >= 0
    assert data["orders_covariate"].sub(data["sessions_covariate"]).min() <= 0
    assert data["revenue_covariate"].min() >= 0
    assert (
        data["revenue_covariate"].gt(0)
        .eq(data["orders_covariate"].gt(0))
        .astype(int).min()
    ) == 1


def test_make_users_data_ibis():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(seed=42, n_users=n_users, to_ibis=True)
    assert isinstance(data, ibis.expr.types.Table)
    assert data.columns == ["user", "variant", "sessions", "orders", "revenue"]


def test_make_sessions_data_default():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(seed=42, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert len(data) > n_users
    assert data["user"].drop_duplicates().count() == n_users
    assert data["variant"].drop_duplicates().count() == 2
    assert data["sessions"].min() == 1
    assert data["sessions"].max() == 1
    assert data["orders"].min() >= 0
    assert data["orders"].sub(data["sessions"]).min() <= 0
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_make_sessions_data_covariates():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    ]
    assert data["sessions_covariate"].min() >= 0
    assert data["orders_covariate"].min() >= 0
    assert data["orders_covariate"].sub(data["sessions_covariate"]).min() <= 0
    assert data["revenue_covariate"].min() >= 0
    assert (
        data["revenue_covariate"].gt(0)
        .eq(data["orders_covariate"].gt(0))
        .astype(int).min()
    ) == 1


def test_make_sessions_data_ibis():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(
        seed=42, n_users=n_users, to_ibis=True)
    assert isinstance(data, ibis.expr.types.Table)
    assert data.columns == ["user", "variant", "sessions", "orders", "revenue"]
