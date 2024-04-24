# Overview

## About

**tea-tasting** is a Python package for statistical analysis of A/B tests that features:

- Student's t-test and Z-test out of the box.
- Extensible API: Define and use statistical tests of your choice.
- [Delta method](https://alexdeng.github.io/public/files/kdd2018-dm.pdf) for ratio metrics.
- Variance reduction with [CUPED](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)/[CUPAC](https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/) (also in combination with delta method for ratio metrics).
- Confidence interval for both absolute and percent change.
- Sample ratio mismatch check.

**tea-tasting** calculates statistics within data backends such as BigQuery, ClickHouse, PostgreSQL, Snowflake, Spark, and other of 20+ backends supported by [Ibis](https://ibis-project.org/). This approach eliminates the need to import granular data into a Python environment, though Pandas DataFrames are also supported.

**tea-tasting** is still in alpha, but already includes all the features listed above. The following features are coming soon:

- More statistical tests:
  - Bootstrap.
  - Quantile test (using Bootstrap).
  - Asymptotic and exact tests for frequency data.
  - Mann–Whitney U test.
- Power analysis.
- A/A tests and simulations.
- Pretty output for experiment results (round etc.).
- More documentation and examples.

## Package name

The package name "tea-tasting" is a play of words which refers to two subjects:

- [Lady tasting tea](https://en.wikipedia.org/wiki/Lady_tasting_tea) is a famous experiment which was devised by Ronald Fisher. In this experiment, Fisher developed the null hypothesis significance testing framework to analyze a lady's claim that she could discern whether the tea or the milk was added first to a cup.
- "tea-tasting" phonetically resembles "t-testing" or Student's t-test, a statistical test developed by William Gosset.
