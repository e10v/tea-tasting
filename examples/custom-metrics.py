# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "tea-tasting",
# ]
# [tool.marimo.display]
# cell_output = "below"
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Custom metrics

        ## Intro

        tea-tasting supports Student's t-test, Z-test, and [some other statistical tests](https://tea-tasting.e10v.me/api/metrics/index/) out of the box. However, you might want to analyze an experiment using other statistical criteria. In this case, you can define a custom metric with a statistical test of your choice.

        In tea-tasting, there are two types of metrics:

        - Metrics that require only aggregated statistics for the analysis.
        - Metrics that require granular data for the analysis.

        This guide explains how to define a custom metric for each type.

        First, let's import all the required modules and prepare the data:
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
        This guide uses PyArrow as the data backend, but it's valid for other backends as well. See the [guide on data backends](https://tea-tasting.e10v.me/data-backends/) for more details.

        ## Metrics based on aggregated statistics

        Let's define a metric that performs a proportion test, [G-test](https://en.wikipedia.org/wiki/G-test) or [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test), on a binary column (with values `0` or `1`).

        The first step is defining a result class. It should be a named tuple or a dictionary.
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
        The second step is defining the metric class itself. A metric based on aggregated statistics should be a subclass of [`MetricBaseAggregated`](https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricBaseAggregated). `MetricBaseAggregated` is a generic class with the result class as a type variable.

        The metric should have the following methods and properties defined:

        - Method `__init__` checks and saves metric parameters.
        - Property `aggr_cols` returns columns to be aggregated for analysis for each type of statistic.
        - Method `analyze_aggregates` analyzes the metric using aggregated statistics.

        Let's define the metric and discuss each method in details:
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
        Method `__init__` saves metric parameters to be used in the analysis. You can use utility functions [`check_scalar`](https://tea-tasting.e10v.me/api/utils/#tea_tasting.utils.check_scalar) and [`auto_check`](https://tea-tasting.e10v.me/api/utils/#tea_tasting.utils.auto_check) to check parameter values.

        Property `aggr_cols` returns an instance of [`AggrCols`](https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.AggrCols). Analysis of proportion requires the number of rows (`has_count=True`) and the average value for the column of interest (`mean_cols=(self.column,)`) for each variant.

        Method `analyze_aggregates` accepts two parameters: `control` and `treatment` data as instances of class [`Aggregates`](https://tea-tasting.e10v.me/api/aggr/#tea_tasting.aggr.Aggregates). They contain values for statistics and columns specified in `aggr_cols`.

        Method `analyze_aggregates` returns an instance of `ProportionResult`, defined earlier, with the analysis result.

        Now we can analyze the proportion of users who created at least one order during the experiment. For comparison, let's also add a metric that performs a Z-test on the same column.
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
        ## Metrics based on granular data

        Now let's define a metric that performs the Mann-Whitney U test. While it's possible to use the aggregated sum of ranks for the test, this example uses granular data for analysis.

        The result class:
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
        A metric that analyzes granular data should be a subclass of [`MetricBaseGranular`](https://tea-tasting.e10v.me/api/metrics/base/#tea_tasting.metrics.base.MetricBaseGranular). `MetricBaseGranular` is a generic class with the result class as a type variable.

        Metric should have the following methods and properties defined:

        - Method `__init__` checks and saves metric parameters.
        - Property `cols` returns columns to be fetched for an analysis.
        - Method `analyze_granular` analyzes the metric using granular data.
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
        Property `cols` should return a sequence of strings.

        Method `analyze_granular` accepts two parameters: control and treatment data as PyArrow Tables. Even with [data backend](https://tea-tasting.e10v.me/data-backends/) different from PyArrow, tea-tasting will retrieve the data and transform into a PyArrow Table.

        Method `analyze_granular` returns an instance of `MannWhitneyUResult`, defined earlier, with analysis result.

        Now we can perform the Mann-Whitney U test:
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
        ## Analyzing two types of metrics together

        It's also possible to analyze two types of metrics in one experiment:
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
        In this case, tea-tasting performs two queries on the experimental data:

        - With aggregated statistics required for analysis of metrics of type `MetricBaseAggregated`.
        - With detailed data with columns required for analysis of metrics of type `MetricBaseGranular`.

        ## Recommendations

        Follow these recommendations when defining custom metrics:

        - Use parameter and attribute names consistent with the ones that are already defined in tea-tasting. For example, use `pvalue` instead of `p_value` or `correction` instead of `use_continuity`.
        - End confidence interval boundary names with `"_ci_lower"` and `"_ci_upper"`.
        - During initialization, save parameter values in metric attributes using the same names. For example, use `self.correction = correction` instead of `self.use_continuity = correction`.
        - Use global settings as default values for standard parameters, such as `alternative` or `confidence_level`. See the [reference](https://tea-tasting.e10v.me/api/config/#tea_tasting.config.config_context) for the full list of standard parameters. You can also define and use your own global parameters.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
