"""API reference is auto-generated based on docstrings.

All classes and functions for the analysis of the experiments can be imported
from the root `tea_tasting` module. For convenience, the API reference is provided
by submodules of `tea_tasting`:

- `tea_tasting.metrics`: Built-in metrics.
- `tea_tasting.experiment`: Experiment and experiment result.
- `tea_tasting.datasets`: Example datasets.
- `tea_tasting.config`: Global configuration.
- `tea_tasting.aggr`: Module for working with aggregated statistics.
- `tea_tasting.utils`: Useful functions and classes.
"""
# pyright: reportUnusedImport=false

from tea_tasting.config import config_context, get_config, set_config
from tea_tasting.datasets import make_sessions_data, make_users_data
from tea_tasting.experiment import Experiment
from tea_tasting.metrics import Mean, RatioOfMeans, SampleRatio
from tea_tasting.version import __version__
