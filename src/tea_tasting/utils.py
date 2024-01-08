"""Useful functions."""

from typing import Any


def check_value(
    value: Any,
    name: str = "Value",
    type_: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
) -> None:
    """Validate parameter type and value.

    Args:
        value: Parameter value.
        name: Parameter name.
        type_: If not `None`, parameter value must be an instance of `type_`.
        ge: If not `None`, parameter value must be `>= ge`.
        gt: If not `None`, parameter value must be `> gt`.
        le: If not `None`, parameter value must be `<= le`.
        lt: If not `None`, parameter value must be `< lt`.

    Raises:
        TypeError: Parameter value is not an instance of `type_`.
        ValueError: any of the cases:
        - Parameter value `< ge`.
        - Parameter value `<= gt`.
        - Parameter value `> le`.
        - Parameter value `>= lt`.
    """
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
