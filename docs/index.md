# tea-tasting: statistical analysis of A/B tests

[![CI](https://github.com/e10v/tea-tasting/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/e10v/tea-tasting/actions/workflows/ci.yml)
[![Docs](https://github.com/e10v/tea-tasting/actions/workflows/docs.yml/badge.svg)](https://tea-tasting.e10v.me/)
[![Coverage](https://codecov.io/github/e10v/tea-tasting/coverage.svg?branch=main)](https://codecov.io/gh/e10v/tea-tasting)
[![License](https://img.shields.io/github/license/e10v/tea-tasting)](https://github.com/e10v/tea-tasting/blob/main/LICENSE)
[![Package Status](https://img.shields.io/pypi/status/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)
[![Version](https://img.shields.io/pypi/v/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/tea-tasting.svg)](https://pypi.org/project/tea-tasting/)

tea-tasting is a Python package for the statistical analysis of A/B tests featuring:

- Student's t-test, Z-test, bootstrap, and quantile metrics out of the box.
- Extensible API that lets you define and use statistical tests of your choice.
- [Delta method](https://alexdeng.github.io/public/files/kdd2018-dm.pdf) for ratio metrics.
- Variance reduction using [CUPED](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)/[CUPAC](https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/), which can be combined with the Delta method for ratio metrics.
- Confidence intervals for both absolute and percentage changes.
- Checks for sample-ratio mismatches.
- Power analysis.
- Multiple hypothesis testing (family-wise error rate and false discovery rate).
- Simulated experiments, including A/A tests.

tea-tasting calculates statistics directly within data backends such as BigQuery, ClickHouse, DuckDB, PostgreSQL, Snowflake, Spark, and many other backends supported by [Ibis](https://github.com/ibis-project/ibis). This approach eliminates the need to import granular data into a Python environment.

tea-tasting also accepts dataframes supported by [Narwhals](https://github.com/narwhals-dev/narwhals): cuDF, Dask, Modin, pandas, Polars, PyArrow.

## Installation

```bash
uv pip install tea-tasting
```

## Basic example

```pycon
>>> import tea_tasting as tt

>>> data = tt.make_users_data(seed=42)
>>> experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
... )
>>> result = experiment.analyze(data)
>>> result
            metric control treatment rel_effect_size rel_effect_size_ci pvalue
 sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]  0.674
orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%] 0.0762
   orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]  0.118
  revenue_per_user    5.24      5.73            9.3%       [-2.4%, 22%]  0.123

```

Learn more in the detailed [user guide](https://tea-tasting.e10v.me/user-guide/). Additionally, see the guides on more specific topics:

- [Data backends](https://tea-tasting.e10v.me/data-backends/).
- [Power analysis](https://tea-tasting.e10v.me/power-analysis/).
- [Multiple hypothesis testing](https://tea-tasting.e10v.me/multiple-testing/).
- [Custom metrics](https://tea-tasting.e10v.me/custom-metrics/).
- [Simulated experiments](https://tea-tasting.e10v.me/simulated-experiments/).

## Examples

The tea-tasting repository includes [examples](https://github.com/e10v/tea-tasting/tree/main/examples) as copies of the guides in the [marimo](https://github.com/marimo-team/marimo) notebook format. You can either download them from GitHub and run in your local environment, or you can run them as WASM notebooks in the online playground.

### Run in a local environment

To run the examples in your local environment, clone the repository and change the directory:

```bash
git clone git@github.com:e10v/tea-tasting.git && cd tea-tasting
```

Install marimo, tea-tasting, and other packages used in the examples:

```bash
uv venv && uv pip install marimo tea-tasting polars ibis-framework[duckdb]
```

Launch the notebook server:

```bash
uv run marimo edit examples
```

Now you can choose and run the example notebooks.

### Run in the online playground

To run the examples as WASM notebooks in the online playground, open the following links:

- [User guide](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fuser-guide.py&embed=true).
- [Data backends](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fdata-backends.py&embed=true).
- [Power analysis](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fpower-analysis.py&embed=true).
- [Multiple hypothesis testing](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fmultiple-testing.py&embed=true).
- [Custom metrics](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fcustom-metrics.py&embed=true).
- [Simulated experiments](https://marimo.app/gh/e10v/tea-tasting/main?entrypoint=examples%2Fsimulated-experiments.py&embed=true).

[WASM notebooks](https://docs.marimo.io/guides/wasm/) run entirely in the browser on [Pyodide](https://github.com/pyodide/pyodide) and thus have some limitations. In particular:

- Tables and dataframes render less attractively because Pyodide doesn't always include the latest [packages versions](https://pyodide.org/en/stable/usage/packages-in-pyodide.html).
- You can't simulate experiments [in parallel](https://tea-tasting.e10v.me/simulated-experiments/#parallel-execution) because Pyodide currently [doesn't support multiprocessing](https://pyodide.org/en/stable/usage/wasm-constraints.html#included-but-not-working-modules).
- Other unpredictable issues may arise, such as the inability to use duckdb with ibis.

## Package name

The package name "tea-tasting" is a play on words that refers to two subjects:

- [Lady tasting tea](https://en.wikipedia.org/wiki/Lady_tasting_tea) is a famous experiment which was devised by Ronald Fisher. In this experiment, Fisher developed the null hypothesis significance testing framework to analyze a lady's claim that she could discern whether the tea or the milk was added first to the cup.
- "tea-tasting" phonetically resembles "t-testing", referencing Student's t-test, a statistical method developed by William Gosset.
