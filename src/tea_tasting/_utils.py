"""Useful functions."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any


def check_scalar(
    value: Any,
    name: str = "Value",
    typ: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
) -> None:
    if typ is not None and not isinstance(value, typ):
        raise TypeError(f"{name} must be an instance of {typ}.")

    if ge is not None and value < ge:
        raise ValueError(f"{name} == {value}, must be >= {ge}.")

    if gt is not None and value <= gt:
        raise ValueError(f"{name} == {value}, must be > {gt}.")

    if le is not None and value > le:
        raise ValueError(f"{name} == {value}, must be <= {le}.")

    if lt is not None and value >= lt:
        raise ValueError(f"{name} == {value}, must be < {lt}.")


def sorted_tuple(left: str, right: str) -> tuple[str, str]:
    if right < left:
        return right, left
    return left, right
