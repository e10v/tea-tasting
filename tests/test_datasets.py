# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import tea_tasting.datasets


def test_sample_users_data_default():
    users_data = tea_tasting.datasets.sample_users_data(size=100, seed=42)
    assert users_data.columns == ["user", "variant", "visits", "orders", "revenue"]

    data = users_data.to_pandas()
    assert len(data) == 100
    assert data["user"].count() == data["user"].drop_duplicates().count()
    assert data["variant"].drop_duplicates().count() == 2
    assert data["visits"].min() > 0
    assert data["orders"].min() >= 0
    assert data["orders"].le(data["visits"]).astype(int).min() == 1
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_sample_users_data_covariates():
    users_data = tea_tasting.datasets.sample_users_data(
        size=100, seed=42, covariates=True)
    assert users_data.columns == [
        "user", "variant", "visits", "orders", "revenue",
        "visits_covariate", "orders_covariate", "revenue_covariate",
    ]

    data = users_data.to_pandas()
    assert data["visits_covariate"].min() >= 0
    assert data["orders_covariate"].min() >= 0
    assert data["orders_covariate"].le(data["visits_covariate"]).astype(int).min() == 1
    assert data["revenue_covariate"].min() >= 0
    assert (
        data["revenue_covariate"].gt(0)
        .eq(data["orders_covariate"].gt(0))
        .astype(int).min()
    ) == 1
