# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars",
#     "tea-tasting",
# ]
# ///

import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1 id="multiple-testing">Multiple testing</h1>
        <h2 id="multiple-hypothesis-testing-problem">Multiple hypothesis testing problem</h2>
        <details class="note" open="open">
        <summary>Note</summary>
        <p>This guide uses <a href="https://github.com/pola-rs/polars" target="_blank">Polars</a> as an example data backend. Install Polars in addition to tea-tasting to reproduce the examples:</p>
        <div class="highlight"><pre><span></span><code>pip<span class="w"> </span>install<span class="w"> </span>polars
        </code></pre></div>
        </details>
        <p>The <a href="https://en.wikipedia.org/wiki/Multiple_comparisons_problem" target="_blank">multiple hypothesis testing problem</a> arises when there is more than one success metric or more than one treatment variant in an A/B test.</p>
        <p>tea-tasting provides the following methods for multiple testing correction:</p>
        <ul>
        <li><a href="https://en.wikipedia.org/wiki/False_discovery_rate" target="_blank">False discovery rate</a> (FDR) controlling procedures:<ul>
        <li>Benjamini-Hochberg procedure, assuming non-negative correlation between hypotheses.</li>
        <li>Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses.</li>
        </ul>
        </li>
        <li><a href="https://en.wikipedia.org/wiki/Family-wise_error_rate" target="_blank">Family-wise error rate</a> (FWER) controlling procedures:<ul>
        <li>Hochberg's step-up procedure, assuming non-negative correlation between hypotheses.</li>
        <li>Holm's step-down procedure, assuming arbitrary dependence between hypotheses.</li>
        </ul>
        </li>
        </ul>
        <p>As an example, consider an experiment with three variants, a control and two treatments:</p>
        """
    )
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
    mo.md(
        r"""
        <p>Let's calculate the experiment results:</p>
        """
    )
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
    mo.md(
        r"""
        <p>Suppose only the two metrics <code>orders_per_user</code> and <code>revenue_per_user</code> are considered as success metrics, while the other two metrics <code>sessions_per_user</code> and <code>orders_per_session</code> are second-order diagnostic metrics.</p>
        """
    )
    return


@app.cell
def _():
    metrics = {"orders_per_user", "revenue_per_user"}
    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>With two treatment variants and two success metrics, there are four hypotheses in total, which increases the probability of false positives (also called "false discoveries"). It's recommended to adjust the p-values or the significance level (alpha) in this case. Let's explore the correction methods provided by tea-tasting.</p>
        <h2 id="false-discovery-rate">False discovery rate</h2>
        <p>False discovery rate (FDR) is the expected value of the proportion of false discoveries among the discoveries (rejections of the null hypothesis). To control for FDR, use the <a href="https://tea-tasting.e10v.me/api/multiplicity/#tea_tasting.multiplicity.adjust_fdr" target="_blank"><code>adjust_fdr</code></a> method:</p>
        """
    )
    return


@app.cell
def _(metrics, results, tt):
    adjusted_results_fdr = tt.adjust_fdr(results, metrics)
    adjusted_results_fdr
    return (adjusted_results_fdr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>The method adjusts p-values and saves them as <code>pvalue_adj</code>. Compare these values to the desired significance level alpha to determine if the null hypotheses can be rejected.</p>
        <p>The method also adjusts the significance level alpha and saves it as <code>alpha_adj</code>. Compare non-adjusted p-values (<code>pvalue</code>) to the <code>alpha_adj</code> to determine if the null hypotheses can be rejected:</p>
        """
    )
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
    mo.md(
        r"""
        <p>By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Benjamini-Hochberg procedure. To perform the Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses, set the <code>arbitrary_dependence</code> parameter to <code>True</code>:</p>
        """
    )
    return


@app.cell
def _(metrics, results, tt):
    tt.adjust_fdr(results, metrics, arbitrary_dependence=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h2 id="family-wise-error-rate">Family-wise error rate</h2>
        <p>Family-wise error rate (FWER) is the probability of making at least one type I error. To control for FWER, use the <a href="https://tea-tasting.e10v.me/api/multiplicity/#tea_tasting.multiplicity.adjust_fwer" target="_blank"><code>adjust_fwer</code></a> method:</p>
        """
    )
    return


@app.cell
def _(metrics, results, tt):
    tt.adjust_fwer(results, metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Hochberg's step-up procedure with the Šidák correction, which is slightly more powerful than the Bonferroni correction.</p>
        <p>To perform the Holm's step-down procedure, assuming arbitrary dependence between hypotheses, set the <code>arbitrary_dependence</code> parameter to <code>True</code>. In this case, it's recommended to use the Bonferroni correction, since the Šidák correction assumes non-negative correlation between hypotheses:</p>
        """
    )
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
    mo.md(
        r"""
        <h2 id="other-inputs">Other inputs</h2>
        <p>In the examples above, the methods <code>adjust_fdr</code> and <code>adjust_fwer</code> received results from a <em>single experiment</em> with <em>more than two variants</em>. They can also accept the results from <em>multiple experiments</em> with <em>two variants</em> in each:</p>
        """
    )
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
    mo.md(
        r"""
        <p>The methods <code>adjust_fdr</code> and <code>adjust_fwer</code> can also accept the result of <em>a single experiment with two variants</em>:</p>
        """
    )
    return


@app.cell
def _(metrics, result2, tt):
    tt.adjust_fwer(result2, metrics)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


if __name__ == "__main__":
    app.run()
