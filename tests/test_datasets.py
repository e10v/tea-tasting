from __future__ import annotations

import tea_tasting.datasets


def test_make_users_data_default():
    size = 100
    users_data = tea_tasting.datasets.make_users_data(size=size, seed=42)
    assert users_data.columns == ["user", "variant", "visits", "orders", "revenue"]

    data = users_data.to_pandas()
    assert len(data) == size
    assert data["user"].drop_duplicates().count() == size
    assert data["variant"].drop_duplicates().count() == 2
    assert data["visits"].min() > 0
    assert data["orders"].min() >= 0
    assert data["orders"].sub(data["visits"]).min() <= 0
    assert data["revenue"].min() >= 0
    assert data["revenue"].gt(0).eq(data["orders"].gt(0)).astype(int).min() == 1


def test_make_users_data_covariates():
    size = 100
    users_data = tea_tasting.datasets.make_users_data(
        size=size, seed=42, covariates=True)
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
