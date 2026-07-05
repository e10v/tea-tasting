"""Data backend adapters."""

from tea_tasting.backends.base import BaseTable, BaseTableGroupBy
from tea_tasting.backends.ibis import IbisTable
from tea_tasting.backends.narwhals import NarwhalsFrame
from tea_tasting.backends.sql import SQLQuery
