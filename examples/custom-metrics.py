# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
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
        <h1 id="custom-metrics">Custom metrics</h1>
        <h2 id="intro">Intro</h2>
        <p>tea-tasting supports Student's t-test, Z-test, and <a href="https://tea-tasting.e10v.me/api/metrics/index/" target="_blank">some other statistical tests</a> out of the box. However, you might want to analyze an experiment using other statistical criteria. In this case, you can define a custom metric with a statistical test of your choice.</p>
        <p>In tea-tasting, there are two types of metrics:</p>
        <ul>
        <li>Metrics that require only aggregated statistics for the analysis.</li>
        <li>Metrics that require granular data for the analysis.</li>
        </ul>
        <p>This guide explains how to define a custom metric for each type.</p>
        <p>First, let's import all the required modules and prepare the data:</p>
        """
    )
    return


@app.cell
def _():
    from typing import Literal, NamedTuple
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import scipy.stats
    import tea_tasting as tt
    import tea_tasting.aggr
    import tea_tasting.config
    import tea_tasting.metrics
    import tea_tasting.utils

    data = tt.make_users_data(seed=42)
    data = data.append_column(
        "has_order",
        pc.greater(data["orders"], 0).cast(pa.int64()),
    )
    data
    return Literal, NamedTuple, data, np, pa, scipy, tea_tasting, tt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>This guide uses PyArrow as the data backend, but it's valid for other backends as well. See the <a href="https://tea-tasting.e10v.me/data-backends/" target="_blank">guide on data backends</a> for more details.</p>
        <h2 id="metrics-based-on-aggregated-statistics">Metrics based on aggregated statistics</h2>
        <p>Let's define a metric that performs a proportion test, <a href="https://en.wikipedia.org/wiki/G-test" target="_blank">G-test</a> or <a href="https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test" target="_blank">Pearson's chi-squared test</a>, on a binary column (with values <code>0</code> or <code>1</code>).</p>
        <p>The first step is defining a result class. It should be a named tuple or a dictionary.</p>
        """
    )
    return


@app.cell
def _(NamedTuple):
    class ProportionResult(NamedTuple):
        control: float
        treatment: float
        effect_size: float
        rel_effect_size: float
        pvalue: float
        statistic: float
    return (ProportionResult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>The second step is defining the metric class itself. A metric based on aggregated statistics should be a subclass of <a href="https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricBaseAggregated" target="_blank"><code>MetricBaseAggregated</code></a>. <code>MetricBaseAggregated</code> is a generic class with the result class as a type variable.</p>
        <p>The metric should have the following methods and properties defined:</p>
        <ul>
        <li>Method <code>__init__</code> checks and saves metric parameters.</li>
        <li>Property <code>aggr_cols</code> returns columns to be aggregated for analysis for each type of statistic.</li>
        <li>Method <code>analyze_aggregates</code> analyzes the metric using aggregated statistics.</li>
        </ul>
        <p>Let's define the metric and discuss each method in details:</p>
        """
    )
    return


@app.cell
def _(Literal, ProportionResult, np, scipy, tea_tasting):
    class Proportion(tea_tasting.metrics.MetricBaseAggregated[ProportionResult]):
        def __init__(
            self,
            column: str,
            *,
            correction: bool = True,
            method: Literal["g-test", "pearson"] = "g-test",
        ) -> None:
            self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
            self.correction = tea_tasting.utils.auto_check(correction, "correction")
            self.method = tea_tasting.utils.check_scalar(
                method, "method", typ=str, in_={"g-test", "pearson"})
        @property
        def aggr_cols(self) -> tea_tasting.metrics.AggrCols:
            return tea_tasting.metrics.AggrCols(
                has_count=True,
                mean_cols=(self.column,),
            )
        def analyze_aggregates(
            self,
            control: tea_tasting.aggr.Aggregates,
            treatment: tea_tasting.aggr.Aggregates,
        ) -> ProportionResult:
            observed = np.empty(shape=(2, 2), dtype=np.int64)
            observed[0, 0] = round(control.count() * control.mean(self.column))
            observed[1, 0] = control.count() - observed[0, 0]
            observed[0, 1] = round(treatment.count() * treatment.mean(self.column))
            observed[1, 1] = treatment.count() - observed[0, 1]
            res = scipy.stats.chi2_contingency(
                observed=observed,
                correction=self.correction,
                lambda_=int(self.method == "pearson"),
            )
            return ProportionResult(
                control=control.mean(self.column),
                treatment=treatment.mean(self.column),
                effect_size=treatment.mean(self.column) - control.mean(self.column),
                rel_effect_size=treatment.mean(self.column)/control.mean(self.column) - 1,
                pvalue=res.pvalue,
                statistic=res.statistic,
            )
    return (Proportion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>Method <code>__init__</code> saves metric parameters to be used in the analysis. You can use utility functions <a href="https://tea-tasting.e10v.me/api/utils/#tea_tasting.utils.check_scalar" target="_blank"><code>check_scalar</code></a> and <a href="https://tea-tasting.e10v.me/api/utils/#tea_tasting.utils.auto_check" target="_blank"><code>auto_check</code></a> to check parameter values.</p>
        <p>Property <code>aggr_cols</code> returns an instance of <a href="https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.AggrCols" target="_blank"><code>AggrCols</code></a>. Analysis of proportion requires the number of rows (<code>has_count=True</code>) and the average value for the column of interest (<code>mean_cols=(self.column,)</code>) for each variant.</p>
        <p>Method <code>analyze_aggregates</code> accepts two parameters: <code>control</code> and <code>treatment</code> data as instances of class <a href="https://tea-tasting.e10v.me/api/aggr/#tea_tasting.aggr.Aggregates" target="_blank"><code>Aggregates</code></a>. They contain values for statistics and columns specified in <code>aggr_cols</code>.</p>
        <p>Method <code>analyze_aggregates</code> returns an instance of <code>ProportionResult</code>, defined earlier, with the analysis result.</p>
        <p>Now we can analyze the proportion of users who created at least one order during the experiment. For comparison, let's also add a metric that performs a Z-test on the same column.</p>
        """
    )
    return


@app.cell
def _(Proportion, data, tt):
    experiment_prop = tt.Experiment(
        prop_users_with_orders=Proportion("has_order"),
        mean_users_with_orders=tt.Mean("has_order", use_t=False),
    )
    experiment_prop.analyze(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h2 id="metrics-based-on-granular-data">Metrics based on granular data</h2>
        <p>Now let's define a metric that performs the Mann-Whitney U test. While it's possible to use the aggregated sum of ranks for the test, this example uses granular data for analysis.</p>
        <p>The result class:</p>
        """
    )
    return


@app.cell
def _(NamedTuple):
    class MannWhitneyUResult(NamedTuple):
        pvalue: float
        statistic: float
    return (MannWhitneyUResult,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>A metric that analyzes granular data should be a subclass of <a href="https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricBaseGranular" target="_blank"><code>MetricBaseGranular</code></a>. <code>MetricBaseGranular</code> is a generic class with the result class as a type variable.</p>
        <p>Metric should have the following methods and properties defined:</p>
        <ul>
        <li>Method <code>__init__</code> checks and saves metric parameters.</li>
        <li>Property <code>cols</code> returns columns to be fetched for an analysis.</li>
        <li>Method <code>analyze_granular</code> analyzes the metric using granular data.</li>
        </ul>
        """
    )
    return


@app.cell
def _(Literal, MannWhitneyUResult, pa, scipy, tea_tasting):
    class MannWhitneyU(tea_tasting.metrics.MetricBaseGranular[MannWhitneyUResult]):
        def __init__(
            self,
            column: str,
            *,
            correction: bool = True,
            alternative: Literal["two-sided", "less", "greater"] | None = None,
        ) -> None:
            self.column = tea_tasting.utils.check_scalar(column, "column", typ=str)
            self.correction = tea_tasting.utils.auto_check(correction, "correction")
            self.alternative = (
                tea_tasting.utils.auto_check(alternative, "alternative")
                if alternative is not None
                else tea_tasting.config.get_config("alternative")
            )
        @property
        def cols(self) -> tuple[str]:
            return (self.column,)
        def analyze_granular(
            self,
            control: pa.Table,
            treatment: pa.Table,
        ) -> MannWhitneyUResult:
            res = scipy.stats.mannwhitneyu(
                treatment[self.column].combine_chunks().to_numpy(zero_copy_only=False),
                control[self.column].combine_chunks().to_numpy(zero_copy_only=False),
                use_continuity=self.correction,
                alternative=self.alternative,
            )
            return MannWhitneyUResult(
                pvalue=res.pvalue,
                statistic=res.statistic,
            )
    return (MannWhitneyU,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>Property <code>cols</code> should return a sequence of strings.</p>
        <p>Method <code>analyze_granular</code> accepts two parameters: control and treatment data as PyArrow Tables. Even with <a href="https://tea-tasting.e10v.me/data-backends/" target="_blank">data backend</a> different from PyArrow, tea-tasting will retrieve the data and transform into a PyArrow Table.</p>
        <p>Method <code>analyze_granular</code> returns an instance of <code>MannWhitneyUResult</code>, defined earlier, with analysis result.</p>
        <p>Now we can perform the Mann-Whitney U test:</p>
        """
    )
    return


@app.cell
def _(MannWhitneyU, data, tt):
    experiment_mwu = tt.Experiment(
        mwu_orders=MannWhitneyU("orders"),
        mwu_revenue=MannWhitneyU("revenue"),
    )
    result_mwu = experiment_mwu.analyze(data)
    result_mwu.with_keys(("metric", "pvalue", "statistic"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h2 id="analyzing-two-types-of-metrics-together">Analyzing two types of metrics together</h2>
        <p>It's also possible to analyze two types of metrics in one experiment:</p>
        """
    )
    return


@app.cell
def _(MannWhitneyU, Proportion, data, tt):
    experiment = tt.Experiment(
        prop_users_with_orders=Proportion("has_order"),
        mean_users_with_orders=tt.Mean("has_order"),
        mwu_orders=MannWhitneyU("orders"),
        mwu_revenue=MannWhitneyU("revenue"),
    )
    experiment.analyze(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p>In this case, tea-tasting performs two queries on the experimental data:</p>
        <ul>
        <li>With aggregated statistics required for analysis of metrics of type <code>MetricBaseAggregated</code>.</li>
        <li>With detailed data with columns required for analysis of metrics of type <code>MetricBaseGranular</code>.</li>
        </ul>
        <h2 id="recommendations">Recommendations</h2>
        <p>Follow these recommendations when defining custom metrics:</p>
        <ul>
        <li>Use parameter and attribute names consistent with the ones that are already defined in tea-tasting. For example, use <code>pvalue</code> instead of <code>p_value</code> or <code>correction</code> instead of <code>use_continuity</code>.</li>
        <li>End confidence interval boundary names with <code>"_ci_lower"</code> and <code>"_ci_upper"</code>.</li>
        <li>During initialization, save parameter values in metric attributes using the same names. For example, use <code>self.correction = correction</code> instead of <code>self.use_continuity = correction</code>.</li>
        <li>Use global settings as default values for standard parameters, such as <code>alternative</code> or <code>confidence_level</code>. See the <a href="https://tea-tasting.e10v.me/api/config/#tea_tasting.config.config_context" target="_blank">reference</a> for the full list of standard parameters. You can also define and use your own global parameters.</li>
        </ul>
        """
    )
    return


if __name__ == "__main__":
    app.run()
