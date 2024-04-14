"""Useful functions and classes."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, TypeVar

    R = TypeVar("R")


def check_scalar(
    value: R,
    name: str = "value",
    *,
    typ: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
    is_in: Any = None,
) -> R:
    """Check scalar parameter's type and value.

    Args:
        value: Parameter value.
        name: Parameter name.
        typ: Acceptable data types.
        ge: If not None, check that the parameter value is greater than or equal to ge.
        gt: If not None, check that the parameter value is greater than to gt.
        le: If not None, check that the parameter value is less than or equal to le.
        lt: If not None, check that the parameter value is less than to gt.
        is_in: If not None, check that the parameter value is in is_in.

    Returns:
        Parameter value.
    """
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
    if is_in is not None and value not in is_in:
        raise ValueError(f"{name} == {value}, must be in {is_in}.")
    return value


def auto_check(value: R, name: str) -> R:
    """Check parameter's type and value based in it's name.

    Args:
        value: Parameter value.
        name: Parameter name.

    Returns:
        Parameter value.
    """
    if name == "alternative":
        check_scalar(value, name, typ=str, is_in={"two-sided", "greater", "less"})
    elif name == "confidence_level":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    elif name in {"equal_var", "use_t"}:
        check_scalar(value, name, typ=bool)
    return value


class ReprMixin:
    """Mixin class for object representation.

    Representation string is generated based on parameters values saved in attributes.
    """
    @classmethod
    def _get_param_names(cls) -> Iterator[str]:
        if cls.__init__ is object.__init__:
            return
        init_signature = inspect.signature(cls.__init__)

        for p in init_signature.parameters.values():
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "There should not be positional arguments in the __init__.")
            if p.name != "self" and p.kind != p.VAR_KEYWORD:
                yield p.name

    def __repr__(self) -> str:
        """Object representation."""
        params = {p: getattr(self, p) for p in self._get_param_names()}
        params_repr = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_repr})"
