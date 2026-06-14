"""Data backend adapters."""

from tea_tasting.backends.base import BaseTable, BaseTableGroupBy
from tea_tasting.backends.ibis import IbisTable, IbisTableGroupBy
from tea_tasting.backends.narwhals import NarwhalsFrame, NarwhalsFrameGroupBy
