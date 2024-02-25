"""Useful functions and classes."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any


def check_scalar(
    value: Any,
    name: str = "value",
    typ: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
    is_in: Any = None,
) -> None:
    """Validate scalar parameter's type and value.

    Args:
        value: Parameter to validate.
        name: Parameter name.
        typ: Acceptable data types.
        ge: If not None, check that the parameter value is greater than or equal to ge.
        gt: If not None, check that the parameter value is greater than to gt.
        le: If not None, check that the parameter value is less than or equal to le.
        lt: If not None, check that the parameter value is less than to gt.
        is_in: If not None, check that the parameter value is in is_in.
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


class ReprMixin:
    """Mixin class for object representation.

    Representation string is generated based on parameters values saved in attributes.
    Attributes names in priority order:

    - "_{parameter name}",
    - "{parameter name}_",
    - "{parameter name}".
    """
    @classmethod
    def _get_param_names(cls: type[ReprMixin]) -> tuple[str, ...]:
        if cls.__init__ is object.__init__:
            return ()
        init_signature = inspect.signature(cls.__init__)
        params = tuple(
            p for p in init_signature.parameters.values() if p.name != "self")
        for p in params:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "There should not be positional arguments in the __init__.")
        return tuple(p.name for p in params)

    def _get_param_value(self, param_name: str) -> Any:
        if hasattr(self, "_" + param_name):
            return getattr(self, "_" + param_name)
        if hasattr(self, param_name + "_"):
            return getattr(self, param_name + "_")
        return getattr(self, param_name)

    def __repr__(self) -> str:
        """Object representation."""
        params = {p: self._get_param_value(p) for p in self._get_param_names()}
        params_repr = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_repr})"
