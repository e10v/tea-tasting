"""Useful functions and classes."""
# ruff: noqa: SIM114

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal, TypeVar

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
        gt: If not None, check that the parameter value is greater than gt.
        le: If not None, check that the parameter value is less than or equal to le.
        lt: If not None, check that the parameter value is less than gt.
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
    elif name == "correction":
        check_scalar(value, name, typ=bool)
    elif name == "equal_var":
        check_scalar(value, name, typ=bool)
    elif name == "ratio":
        check_scalar(value, name, typ=float | int, gt=0)
    elif name == "use_t":
        check_scalar(value, name, typ=bool)
    return value


def div(
    numer: float | int,
    denom: float | int,
    zero_div: float | int | Literal["auto"] = "auto",
) -> float |int:
    """Handle division by zero.

    Args:
        numer: Numerator.
        denom: Denominator.
        zero_div: Result if denominator is zero. If "auto", return:
            nan if numer == 0,
            inf if numer > 0,
            -inf if numer < 0.

    Returns:
        Result of the division.
    """
    if denom != 0:
        return numer / denom
    if zero_div != "auto":
        return zero_div
    if numer == 0:
        return float("nan")
    return float("inf") if numer > 0 else float("-inf")


class _NumericBase:
    value: Any
    zero_div: float | int | Literal["auto"] = "auto"

    def __add__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x + y, self.zero_div)

    def __sub__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x - y, self.zero_div)

    def __mul__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x * y, self.zero_div)

    def __truediv__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(div(x, y, self.zero_div), self.zero_div)

    def __floordiv__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x // y, self.zero_div)

    def __mod__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x % y, self.zero_div)

    def __divmod__(self, other: Any) -> tuple[Numeric, Numeric]:
        x, y = self.value, getattr(other, "value", other)
        d, m = divmod(x, y)
        return numeric(d, self.zero_div), numeric(m, self.zero_div)

    def __pow__(self, other: Any, mod: Any = None) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        z = getattr(mod, "value", mod)
        return numeric(pow(x, y, z), self.zero_div)

    def __radd__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x + y, self.zero_div)

    def __rsub__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x - y, self.zero_div)

    def __rmul__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x * y, self.zero_div)

    def __rtruediv__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(div(x, y, self.zero_div), self.zero_div)

    def __rfloordiv__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x // y, self.zero_div)

    def __rmod__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x % y, self.zero_div)

    def __rdivmod__(self, other: Any) -> tuple[Numeric, Numeric]:
        y, x = self.value, getattr(other, "value", other)
        d, m = divmod(x, y)
        return numeric(d, self.zero_div), numeric(m, self.zero_div)

    def __rpow__(self, other: Any, mod: Any = None) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        z = getattr(mod, "value", mod)
        return numeric(pow(x, y, z), self.zero_div)

    def __neg__(self) -> Numeric:
        return numeric(-self.value, self.zero_div)

    def __pos__(self) -> Numeric:
        return numeric(self)

    def __abs__(self) -> Numeric:
        return numeric(abs(self.value), self.zero_div)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __round__(self, ndigits: int | None = None) -> Numeric:
        return numeric(round(self.value, ndigits), self.zero_div)

    def __trunc__(self) -> Numeric:
        return numeric(math.trunc(self.value), self.zero_div)

    def __floor__(self) -> Numeric:
        return numeric(math.floor(self.value), self.zero_div)

    def __ceil__(self) -> Numeric:
        return numeric(math.ceil(self.value), self.zero_div)


class Float(_NumericBase, float):
    """Float, which doesn't raise an error on division by zero."""
    def __new__(
        cls,
        value: Any,
        zero_div: float | int | Literal["auto"] = "auto",
    ) -> Float:
        """Float, which doesn't raise an error on division by zero."""
        instance = float.__new__(cls, value)
        instance.value = float(value)
        instance.zero_div = zero_div
        return instance

class Int(_NumericBase, int):
    """Integer, which doesn't raise an error on division by zero."""
    def __new__(
        cls,
        value: Any,
        zero_div: float | int | Literal["auto"] = "auto",
    ) -> Int:
        """Integer, which doesn't raise an error on division by zero."""
        instance = int.__new__(cls, value)
        instance.value = int(value)
        instance.zero_div = zero_div
        return instance

Numeric = Float | Int


def numeric(
    value: Any,
    zero_div: float | int | Literal["auto"] = "auto",
) -> Numeric:
    """Float or integer, which doesn't raise an error on division by zero."""
    if isinstance(value, int):
        return Int(value, zero_div)
    if isinstance(value, float):
        return Float(value, zero_div)
    try:
        return Int(value, zero_div)
    except ValueError:
        return Float(value, zero_div)


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
