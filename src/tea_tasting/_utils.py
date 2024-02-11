"""Useful functions and classes."""

from __future__ import annotations

import inspect
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


class ReprMixin:
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

    def _get_param_value(self: ReprMixin, param_name: str) -> Any:
        if hasattr(self, "_" + param_name):
            return getattr(self, "_" + param_name)
        if hasattr(self, param_name + "_"):
            return getattr(self, param_name + "_")
        return getattr(self, param_name)

    def __repr__(self: ReprMixin) -> str:
        params = {p: self._get_param_value(p) for p in self._get_param_names()}
        params_repr = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_repr})"
