# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
#     "tea-tasting",
# ]
# [tool.marimo.display]
# cell_output = "below"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiple testing

    ## Multiple hypothesis testing problem

    /// admonition | Note

    This guide uses [Polars](https://github.com/pola-rs/polars) as an example data backend. Install Polars in addition to tea-tasting to reproduce the examples:

    ```bash
    uv pip install polars
    ```

    ///

    The [multiple hypothesis testing problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) arises when there is more than one success metric or more than one treatment variant in an A/B test.

    tea-tasting provides the following methods for multiple testing correction:

    - [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate) (FDR) controlling procedures:
        - Benjamini-Hochberg procedure, assuming non-negative correlation between hypotheses.
        - Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses.
    - [Family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate) (FWER) controlling procedures:
        - Hochberg's step-up procedure, assuming non-negative correlation between hypotheses.
        - Holm's step-down procedure, assuming arbitrary dependence between hypotheses.

    As an example, consider an experiment with three variants, a control and two treatments:
    """)
    return


@app.cell
def _():
    import polars as pl
    import tea_tasting as tt

    data = pl.concat((
        tt.make_users_data(
            seed=42,
            orders_uplift=0.10,
            revenue_uplift=0.15,
            return_type="polars",
        ),
        tt.make_users_data(
            seed=21,
            orders_uplift=0.15,
            revenue_uplift=0.20,
            return_type="polars",
        )
            .filter(pl.col("variant").eq(1))
            .with_columns(variant=pl.lit(2, pl.Int64)),
    ))
    data
    return data, tt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's calculate the experiment results:
    """)
    return


@app.cell
def _(data, tt):
    experiment = tt.Experiment(
        sessions_per_user=tt.Mean("sessions"),
        orders_per_session=tt.RatioOfMeans("orders", "sessions"),
        orders_per_user=tt.Mean("orders"),
        revenue_per_user=tt.Mean("revenue"),
    )
    results = experiment.analyze(data, control=0, all_variants=True)
    results
    return experiment, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Suppose only the two metrics `orders_per_user` and `revenue_per_user` are considered as success metrics, while the other two metrics `sessions_per_user` and `orders_per_session` are second-order diagnostic metrics.
    """)
    return


@app.cell
def _():
    metrics = {"orders_per_user", "revenue_per_user"}
    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With two treatment variants and two success metrics, there are four hypotheses in total, which increases the probability of false positives (also called "false discoveries"). It is recommended to adjust the p-values or the significance level (alpha) in this case. Let's explore the correction methods provided by tea-tasting.

    ## False discovery rate

    False discovery rate (FDR) is the expected value of the proportion of false discoveries among the discoveries (rejections of the null hypothesis). To control for FDR, use the [`adjust_fdr`](https://tea-tasting.e10v.me/api/multiplicity/#tea_tasting.multiplicity.adjust_fdr) method:
    """)
    return


@app.cell
def _(metrics, results, tt):
    adjusted_results_fdr = tt.adjust_fdr(results, metrics)
    adjusted_results_fdr
    return (adjusted_results_fdr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The method adjusts p-values and saves them as `pvalue_adj`. Compare these values to the desired significance level alpha to determine if the null hypotheses can be rejected.

    The method also adjusts the significance level alpha and saves it as `alpha_adj`. Compare non-adjusted p-values (`pvalue`) to the `alpha_adj` to determine if the null hypotheses can be rejected:
    """)
    return


@app.cell
def _(adjusted_results_fdr):
    adjusted_results_fdr.with_keys((
        "comparison",
        "metric",
        "control",
        "treatment",
        "rel_effect_size",
        "pvalue",
        "alpha_adj",
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Benjamini-Hochberg procedure. To perform the Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses, set the `arbitrary_dependence` parameter to `True`:
    """)
    return


@app.cell
def _(metrics, results, tt):
    tt.adjust_fdr(results, metrics, arbitrary_dependence=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Family-wise error rate

    Family-wise error rate (FWER) is the probability of making at least one type I error. To control for FWER, use the [`adjust_fwer`](https://tea-tasting.e10v.me/api/multiplicity/#tea_tasting.multiplicity.adjust_fwer) method:
    """)
    return


@app.cell
def _(metrics, results, tt):
    tt.adjust_fwer(results, metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Hochberg's step-up procedure with the Šidák correction, which is slightly more powerful than the Bonferroni correction.

    To perform Holm's step-down procedure, assuming arbitrary dependence between hypotheses, set the `arbitrary_dependence` parameter to `True`. In this case, it is recommended to use the Bonferroni correction, since the Šidák correction assumes non-negative correlation between hypotheses:
    """)
    return


@app.cell
def _(metrics, results, tt):
    tt.adjust_fwer(
        results,
        metrics,
        arbitrary_dependence=True,
        method="bonferroni",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other inputs

    In the examples above, the methods `adjust_fdr` and `adjust_fwer` received results from a *single experiment* with *more than two variants*. They can also accept the results from *multiple experiments* with *two variants* in each:
    """)
    return


@app.cell
def _(experiment, metrics, tt):
    data1 = tt.make_users_data(seed=42, orders_uplift=0.10, revenue_uplift=0.15)
    data2 = tt.make_users_data(seed=21, orders_uplift=0.15, revenue_uplift=0.20)
    result1 = experiment.analyze(data1)
    result2 = experiment.analyze(data2)
    tt.adjust_fdr(
        {"Experiment 1": result1, "Experiment 2": result2},
        metrics,
    )
    return (result2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The methods `adjust_fdr` and `adjust_fwer` can also accept the result of *a single experiment with two variants*:
    """)
    return


@app.cell
def _(metrics, result2, tt):
    tt.adjust_fwer(result2, metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
