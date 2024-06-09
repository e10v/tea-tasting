"""Useful functions and classes."""
# ruff: noqa: SIM114

from __future__ import annotations

import inspect
import locale
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
    in_: Any = None,
) -> R:
    """Check if a scalar parameter meets specified type and value constraints.

    Args:
        value: Parameter value.
        name: Parameter name.
        typ: Acceptable data types.
        ge: If not `None`, check that the parameter value is greater than
            or equal to `ge`.
        gt: If not `None`, check that the parameter value is greater than `gt`.
        le: If not `None`, check that the parameter value is less than or equal to `le`.
        lt: If not `None`, check that the parameter value is less than `lt`.
        in_: If not `None`, check that the parameter value is in `in_`.

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
    if in_ is not None and value not in in_:
        raise ValueError(f"{name} == {value}, must be in {in_}.")
    return value


def auto_check(value: R, name: str) -> R:
    """Automatically check a parameter's type and value based on its name.

    Args:
        value: Parameter value.
        name: Parameter name.

    Returns:
        Parameter value.
    """
    if name == "alternative":
        check_scalar(value, name, typ=str, in_={"two-sided", "greater", "less"})
    elif name == "confidence_level":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    elif name == "correction":
        check_scalar(value, name, typ=bool)
    elif name == "equal_var":
        check_scalar(value, name, typ=bool)
    elif name == "n_resamples":
        check_scalar(value, name, typ=int, gt=0)
    elif name == "ratio":
        check_scalar(value, name, typ=float | int, gt=0)
    elif name == "use_t":
        check_scalar(value, name, typ=bool)
    return value


def format_num(
    val: float | int | None,
    sig: int = 3,
    pct: bool = False,
    nan: str = "-",
    inf: str = "∞",
    fixed_point_limit: float = 0.001,
    thousands_sep: str | None = None,
    decimal_point: str | None = None,
) -> str:
    """Format a number according to specified formatting rules.

    Args:
        val: Number to format.
        sig: Number of significant digits.
        pct: If `True`, format as a percentage.
        nan: Replacement for `None` and `nan` values.
        inf: Replacement for infinite values.
        fixed_point_limit: Limit, below which number is formatted as exponential.
        thousands_sep: Thousands separator. If `None`, the value from locales is used.
        decimal_point: Decimal point symbol. If `None`, the value from locales is used.

    Returns:
        Formatted number.
    """
    if val is None or math.isnan(val):
        return nan

    if math.isinf(val):
        return inf if val > 0 else "-" + inf

    if pct:
        val = val * 100

    if abs(val) < fixed_point_limit:
        precision = max(0, sig - 1)
        typ = "e" if val != 0 else "f"
    else:
        precision = max(0, sig - 1 - int(math.floor(math.log10(abs(val)))))
        val = round(val, precision)
        # Repeat in order to format 99.999 as "100", not "100.0".
        precision = max(0, sig - 1 - int(math.floor(math.log10(abs(val)))))
        typ = "f"

    result = format(val, f"_.{precision}{typ}")

    if thousands_sep is None:
        thousands_sep = locale.localeconv().get("thousands_sep", "_")  # type: ignore
    if thousands_sep is not None and thousands_sep != "_":
        result = result.replace("_", thousands_sep)

    if decimal_point is None:
        decimal_point = locale.localeconv().get("decimal_point", ".")  # type: ignore
    if decimal_point is not None and decimal_point != ".":
        result = result.replace(".", decimal_point)

    if pct:
        return result + "%"

    return result


def div(
    numer: float | int,
    denom: float | int,
    fill_zero_div: float | int | Literal["auto"] = "auto",
) -> float |int:
    """Perform division, providing specified results for cases of division by zero.

    Args:
        numer: Numerator.
        denom: Denominator.
        fill_zero_div: Result if denominator is zero.

    Returns:
        Result of the division.

    If `fill_zero_div` is equal `"auto"`, return:

    - `nan` if numerator is equal to `0`,
    - `inf` if numerator is greater than `0`,
    - `-inf` if numerator is less than `0`.
    """
    if denom != 0:
        return numer / denom
    if fill_zero_div != "auto":
        return fill_zero_div
    if numer == 0:
        return float("nan")
    return float("inf") if numer > 0 else float("-inf")


class _NumericBase:
    value: Any
    fill_zero_div: float | int | Literal["auto"] = "auto"

    def __add__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x + y, self.fill_zero_div)

    def __sub__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x - y, self.fill_zero_div)

    def __mul__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x * y, self.fill_zero_div)

    def __truediv__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(div(x, y, self.fill_zero_div), self.fill_zero_div)

    def __floordiv__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x // y, self.fill_zero_div)

    def __mod__(self, other: Any) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        return numeric(x % y, self.fill_zero_div)

    def __divmod__(self, other: Any) -> tuple[Numeric, Numeric]:
        x, y = self.value, getattr(other, "value", other)
        d, m = divmod(x, y)
        return numeric(d, self.fill_zero_div), numeric(m, self.fill_zero_div)

    def __pow__(self, other: Any, mod: Any = None) -> Numeric:
        x, y = self.value, getattr(other, "value", other)
        z = getattr(mod, "value", mod)
        return numeric(pow(x, y, z), self.fill_zero_div)

    def __radd__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x + y, self.fill_zero_div)

    def __rsub__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x - y, self.fill_zero_div)

    def __rmul__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x * y, self.fill_zero_div)

    def __rtruediv__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(div(x, y, self.fill_zero_div), self.fill_zero_div)

    def __rfloordiv__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x // y, self.fill_zero_div)

    def __rmod__(self, other: Any) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        return numeric(x % y, self.fill_zero_div)

    def __rdivmod__(self, other: Any) -> tuple[Numeric, Numeric]:
        y, x = self.value, getattr(other, "value", other)
        d, m = divmod(x, y)
        return numeric(d, self.fill_zero_div), numeric(m, self.fill_zero_div)

    def __rpow__(self, other: Any, mod: Any = None) -> Numeric:
        y, x = self.value, getattr(other, "value", other)
        z = getattr(mod, "value", mod)
        return numeric(pow(x, y, z), self.fill_zero_div)

    def __neg__(self) -> Numeric:
        return numeric(-self.value, self.fill_zero_div)

    def __pos__(self) -> Numeric:
        return numeric(self)

    def __abs__(self) -> Numeric:
        return numeric(abs(self.value), self.fill_zero_div)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __round__(self, ndigits: int | None = None) -> Numeric:
        return numeric(round(self.value, ndigits), self.fill_zero_div)

    def __trunc__(self) -> Numeric:
        return numeric(math.trunc(self.value), self.fill_zero_div)

    def __floor__(self) -> Numeric:
        return numeric(math.floor(self.value), self.fill_zero_div)

    def __ceil__(self) -> Numeric:
        return numeric(math.ceil(self.value), self.fill_zero_div)


class Float(_NumericBase, float):
    """Float that gracefully handles division by zero errors."""
    def __new__(
        cls,
        value: Any,
        fill_zero_div: float | int | Literal["auto"] = "auto",
    ) -> Float:
        """Float that gracefully handles division by zero errors."""
        instance = float.__new__(cls, value)
        instance.value = float(value)
        instance.fill_zero_div = fill_zero_div
        return instance

class Int(_NumericBase, int):
    """Integer that gracefully handles division by zero errors."""
    def __new__(
        cls,
        value: Any,
        fill_zero_div: float | int | Literal["auto"] = "auto",
    ) -> Int:
        """Integer that gracefully handles division by zero errors."""
        instance = int.__new__(cls, value)
        instance.value = int(value)
        instance.fill_zero_div = fill_zero_div
        return instance

Numeric = Float | Int


def numeric(
    value: Any,
    fill_zero_div: float | int | Literal["auto"] = "auto",
) -> Numeric:
    """Float or integer that gracefully handles division by zero errors."""
    if isinstance(value, int):
        return Int(value, fill_zero_div)
    if isinstance(value, float):
        return Float(value, fill_zero_div)
    try:
        return Int(value, fill_zero_div)
    except ValueError:
        return Float(value, fill_zero_div)


class ReprMixin:
    """A mixin class that provides a method for generating a string representation.

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
                    "There should not be positional parameters in the __init__.")
            if p.name != "self" and p.kind != p.VAR_KEYWORD:
                yield p.name

    def __repr__(self) -> str:
        """Object representation."""
        params = {p: getattr(self, p) for p in self._get_param_names()}
        params_repr = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_repr})"
