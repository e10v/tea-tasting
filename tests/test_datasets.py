# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import tea_tasting.datasets


def test_make_users_data_default():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(seed=42, n_users=n_users)
    assert isinstance(data, pa.Table)
    assert data.column_names == ["user", "variant", "sessions", "orders", "revenue"]
    assert data.num_rows == n_users
    assert pc.count_distinct(data["user"]).as_py() == n_users
    assert pc.count_distinct(data["variant"]).as_py() == 2
    assert pc.min(data["sessions"]).as_py() > 0
    assert pc.min(data["orders"]).as_py() >= 0
    assert pc.min(data["revenue"]).as_py() >= 0
    assert pc.min(pc.subtract(data["orders"], data["sessions"])).as_py() <= 0
    assert int(pc.min(pc.equal(
        pc.greater_equal(data["revenue"], 0),
        pc.greater_equal(data["orders"], 0),
    )).as_py()) == 1

def test_make_users_data_pandas():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(
        seed=42, n_users=n_users, return_type="pandas")
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert data.shape[0] == n_users

def test_make_users_data_polars():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(
        seed=42, n_users=n_users, return_type="polars")
    assert isinstance(data, pl.DataFrame)
    assert data.columns == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert data.shape[0] == n_users


def test_make_users_data_covariates():
    n_users = 100
    data = tea_tasting.datasets.make_users_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(data, pa.Table)
    assert data.column_names == [
        "user", "variant", "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    ]
    assert pc.min(data["sessions_covariate"]).as_py() >= 0
    assert pc.min(data["orders_covariate"]).as_py() >= 0
    assert pc.min(data["revenue_covariate"]).as_py() >= 0
    assert pc.min(pc.subtract(
        data["orders_covariate"],
        data["sessions_covariate"],
    )).as_py() <= 0
    assert int(pc.min(pc.equal(
        pc.greater_equal(data["revenue_covariate"], 0),
        pc.greater_equal(data["orders_covariate"], 0),
    )).as_py()) == 1


def test_make_sessions_data_default():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(seed=42, n_users=n_users)
    assert isinstance(data, pa.Table)
    assert data.column_names == ["user", "variant", "sessions", "orders", "revenue"]
    assert data.num_rows > n_users
    assert pc.count_distinct(data["user"]).as_py() == n_users
    assert pc.count_distinct(data["variant"]).as_py() == 2
    assert pc.min(data["sessions"]).as_py() == 1
    assert pc.max(data["sessions"]).as_py() == 1
    assert pc.min(data["orders"]).as_py() >= 0
    assert pc.min(data["revenue"]).as_py() >= 0
    assert pc.min(pc.subtract(data["orders"], data["sessions"])).as_py() <= 0
    assert int(pc.min(pc.equal(
        pc.greater_equal(data["revenue"], 0),
        pc.greater_equal(data["orders"], 0),
    )).as_py()) == 1

def test_make_sessions_data_pandas():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(
        seed=42, n_users=n_users, return_type="pandas")
    assert isinstance(data, pd.DataFrame)
    assert data.columns.to_list() == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert data.shape[0] > n_users

def test_make_sessions_data_polars():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(
        seed=42, n_users=n_users, return_type="polars")
    assert isinstance(data, pl.DataFrame)
    assert data.columns == [
        "user", "variant", "sessions", "orders", "revenue"]
    assert data.shape[0] > n_users


def test_make_sessions_data_covariates():
    n_users = 100
    data = tea_tasting.datasets.make_sessions_data(
        seed=42, covariates=True, n_users=n_users)
    assert isinstance(data, pa.Table)
    assert data.column_names == [
        "user", "variant", "sessions", "orders", "revenue",
        "sessions_covariate", "orders_covariate", "revenue_covariate",
    ]
    assert pc.min(data["sessions_covariate"]).as_py() >= 0
    assert pc.min(data["orders_covariate"]).as_py() >= 0
    assert pc.min(data["revenue_covariate"]).as_py() >= 0
    assert pc.min(pc.subtract(
        data["orders_covariate"],
        data["sessions_covariate"],
    )).as_py() <= 0
    assert int(pc.min(pc.equal(
        pc.greater_equal(data["revenue_covariate"], 0),
        pc.greater_equal(data["orders_covariate"], 0),
    )).as_py()) == 1
