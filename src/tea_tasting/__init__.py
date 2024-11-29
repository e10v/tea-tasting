"""A Python package for the statistical analysis of A/B tests.

All classes and functions for the analysis of the experiments can be imported
from the root `tea_tasting` module.

There are functions and classes for advanced use cases such as defining custom metrics.
They can be imported from submodules of `tea_tasting`.

For convenience, the API reference is provided by submodules:

- `tea_tasting.metrics`: Built-in metrics.
- `tea_tasting.experiment`: Experiment and experiment result.
- `tea_tasting.multiplicity`: Multiple hypothesis testing.
- `tea_tasting.datasets`: Example datasets.
- `tea_tasting.config`: Global configuration.
- `tea_tasting.aggr`: Module for working with aggregated statistics.
- `tea_tasting.utils`: Useful functions and classes.
"""
# pyright: reportUnusedImport=false

from tea_tasting.config import config_context, get_config, set_config
from tea_tasting.datasets import make_sessions_data, make_users_data
from tea_tasting.experiment import Experiment
from tea_tasting.metrics import Bootstrap, Mean, Quantile, RatioOfMeans, SampleRatio
from tea_tasting.multiplicity import adjust_fdr, adjust_fwer
from tea_tasting.version import __version__
