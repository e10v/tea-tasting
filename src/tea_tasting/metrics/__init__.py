"""This module provides built-in metrics used to analyze experimental data.

All metric classes can be imported from `tea_tasting.metrics` module.
For convenience, the API reference is provided by submodules of `tea_tasting.metrics`:

- `tea_tasting.metrics.base`: Base classes for metrics.
- `tea_tasting.metrics.mean`: Metrics for the analysis of means.
- `tea_tasting.metrics.proportion`: Metrics for the analysis of proportions.
- `tea_tasting.metrics.resampling`: Metrics analyzed using resampling methods.
"""
# pyright: reportUnusedImport=false

from tea_tasting.metrics.base import (
    AggrCols,
    MetricBase,
    MetricBaseAggregated,
    MetricBaseGranular,
    MetricPowerResults,
    MetricResult,
    PowerBase,
    PowerBaseAggregated,
    aggregate_by_variants,
    read_granular,
)
from tea_tasting.metrics.mean import Mean, RatioOfMeans
from tea_tasting.metrics.proportion import SampleRatio
from tea_tasting.metrics.resampling import Bootstrap, Quantile
