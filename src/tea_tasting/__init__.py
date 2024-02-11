"""A Python package for statistical analysis of A/B tests."""
# pyright: reportUnusedImport=false

from __future__ import annotations

from tea_tasting.config import config_context, get_config, set_config
from tea_tasting.datasets import make_users_data
from tea_tasting.version import __version__
