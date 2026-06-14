"""Division-safe numeric helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Literal


def div(
    numer: float | int,
    denom: float | int,
    fill_zero_div: float | int | Literal["auto"] = "auto",
) -> float | int:
    """Perform division, providing specified results for cases of division by zero.

    Args:
        numer: Numerator.
        denom: Denominator.
        fill_zero_div: Result if denominator is zero.

    Returns:
        Result of the division.

    If `fill_zero_div` equals `"auto"`, return:

    - `inf` if numerator is greater than `0`,
    - `nan` if numerator is equal to or less than `0`.
    """
    if denom != 0:
        return numer / denom
    if fill_zero_div != "auto":
        return fill_zero_div
    return float("inf") if numer > 0 else float("nan")


class _NumericBase:
    value: float | int
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

type Numeric = Float | Int


def numeric(
    value: object,
    fill_zero_div: float | int | Literal["auto"] = "auto",
) -> Numeric:
    """Convert an object to a numeric type that gracefully handles division by zero.

    Args:
        value: Object to convert.
        fill_zero_div: Result if denominator is zero.

    If `fill_zero_div` equals `"auto"`, division by zero returns:

    - `inf` if numerator is greater than `0`,
    - `nan` if numerator is equal to or less than `0`.

    Returns:
        Float or integer that gracefully handles division by zero errors.
    """
    if isinstance(value, int):
        return Int(value, fill_zero_div)
    if isinstance(value, float):
        return Float(value, fill_zero_div)
    try:
        return Int(value, fill_zero_div)
    except ValueError:
        return Float(value, fill_zero_div)

