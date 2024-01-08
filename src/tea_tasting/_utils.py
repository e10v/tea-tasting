"""Useful functions."""

from __future__ import annotations

from typing import Any


def check_scalar(
    value: Any,
    name: str = "Value",
    type_: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
) -> None:
    if type_ is not None and  not isinstance(value, type_):
        raise TypeError(f"{name} must be an instance of {type_}.")

    if ge is not None and value < ge:
        raise ValueError(f"{name} == {value}, must be >= {ge}.")

    if gt is not None and value <= gt:
        raise ValueError(f"{name} == {value}, must be > {gt}.")

    if le is not None and value > le:
        raise ValueError(f"{name} == {value}, must be <= {le}.")

    if lt is not None and value >= lt:
        raise ValueError(f"{name} == {value}, must be < {lt}.")
