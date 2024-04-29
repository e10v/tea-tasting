"""This module provides built-in metrics used to analyze experimental data.

All metric classes can be imported from `tea_tasting.metrics` module.
For convenience, the API reference is provided by submodules of `tea_tasting.metrics`:

- `tea_tasting.metrics.base`: Base classes for metrics.
- `tea_tasting.metrics.mean`: Metrics for the analysis of means.
- `tea_tasting.metrics.proportion`: Metrics for the analysis of proportions.
"""
# pyright: reportUnusedImport=false

from tea_tasting.metrics.base import (
    AggrCols,
    MetricBase,
    MetricBaseAggregated,
    MetricBaseGranular,
    MetricResult,
    aggregate_by_variants,
    read_dataframes,
)
from tea_tasting.metrics.mean import Mean, RatioOfMeans
from tea_tasting.metrics.proportion import SampleRatio
