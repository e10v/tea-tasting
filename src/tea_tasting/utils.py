"""Useful functions and classes."""
# ruff: noqa: SIM114

from __future__ import annotations

import abc
from collections.abc import Sequence
import inspect
import locale
import math
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Literal, TypeVar

    R = TypeVar("R")


def check_scalar(  # noqa: PLR0913
    value: R,
    name: str = "value",
    *,
    typ: Any = None,
    ge: Any = None,
    gt: Any = None,
    le: Any = None,
    lt: Any = None,
    ne: Any = None,
    in_: Any = None,
) -> R:
    """Check if a scalar parameter meets specified type and value constraints.

    Args:
        value: Parameter value.
        name: Parameter name.
        typ: Acceptable data types.
        ge: If not `None`, check that parameter value is greater than
            or equal to `ge`.
        gt: If not `None`, check that parameter value is greater than `gt`.
        le: If not `None`, check that parameter value is less than or equal to `le`.
        lt: If not `None`, check that parameter value is less than `lt`.
        ne: If not `None`, check that parameter value is not equal to `ne`.
        in_: If not `None`, check that parameter value is in `in_`.

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
    if ne is not None and value == ne:
        raise ValueError(f"{name} == {value}, must be != {ne}.")
    if in_ is not None and value not in in_:
        raise ValueError(f"{name} == {value}, must be in {in_}.")
    return value


def auto_check(value: R, name: str) -> R:  # noqa: C901, PLR0912
    """Automatically check a parameter's type and value based on its name.

    The following parameter names are supported: `"alpha"`, `"alternative"`,
    `"confidence_level"`, `"correction"`, `"equal_var"`, `"n_obs"`,
    `"n_resamples"`, `"power"`, `"ratio"`, `"use_t"`.

    Args:
        value: Parameter value.
        name: Parameter name.

    Returns:
        Parameter value.
    """
    if name == "alpha":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    if name == "alternative":
        check_scalar(value, name, typ=str, in_={"two-sided", "greater", "less"})
    elif name == "confidence_level":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    elif name == "correction":
        check_scalar(value, name, typ=bool)
    elif name == "equal_var":
        check_scalar(value, name, typ=bool)
    elif name == "n_obs":
        check_scalar(value, name, typ=int | Sequence | None)
        if isinstance(value, int):
            check_scalar(value, name, gt=1)
        if isinstance(value, Sequence):
            for val in value:
                check_scalar(val, name, typ=int, gt=1)
    elif name == "n_resamples":
        check_scalar(value, name, typ=int, gt=0)
    elif name == "power":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    elif name == "ratio":
        check_scalar(value, name, typ=float | int, gt=0)
    elif name == "use_t":
        check_scalar(value, name, typ=bool)
    return value


def format_num(
    val: float | int | None,
    sig: int = 3,
    *,
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
    if thousands_sep != "_":
        result = result.replace("_", thousands_sep)

    if decimal_point is None:
        decimal_point = locale.localeconv().get("decimal_point", ".")  # type: ignore
    if decimal_point != ".":
        result = result.replace(".", decimal_point)

    if pct:
        return result + "%"

    return result


def get_and_format_num(data: dict[str, Any], key: str) -> str:
    """Get and format dictionary value.

    Args:
        data: Dictionary.
        key: Key.

    Returns:
        Formatted value.

    Formatting rules:
        - If a name starts with `"rel_"` or equals to `"power"` consider it
            a percentage value. Round percentage values to 2 significant digits,
            multiply by `100` and add `"%"`.
        - Round other values to 3 significant values.
        - If value is less than `0.001`, format it in exponential presentation.
        - If a name ends with `"_ci"`, consider it a confidence interval.
            Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
            and format the interval as `"[{lower_bound}, {lower_bound}]"`.
    """
    if key.endswith("_ci"):
        ci_lower = get_and_format_num(data, key + "_lower")
        ci_upper = get_and_format_num(data, key + "_upper")
        return f"[{ci_lower}, {ci_upper}]"

    val = data.get(key)
    if not isinstance(val, float | int | None):
        return str(val)

    sig, pct = (2, True) if key.startswith("rel_") or key == "power" else (3, False)
    return format_num(val, sig=sig, pct=pct)


class PrettyDictsMixin(abc.ABC):
    """Pretty representation of a sequence of dictionaries.

    Default formatting rules:
        - If a name starts with `"rel_"` or equals to `"power"` consider it
            a percentage value. Round percentage values to 2 significant digits,
            multiply by `100` and add `"%"`.
        - Round other values to 3 significant values.
        - If value is less than `0.001`, format it in exponential presentation.
        - If a name ends with `"_ci"`, consider it a confidence interval.
            Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
            and format the interval as `"[{lower_bound}, {lower_bound}]"`.
    """
    default_keys: Sequence[str]

    @abc.abstractmethod
    def to_dicts(self) -> Sequence[dict[str, Any]]:
        """Convert the object to a sequence of dictionaries."""

    def to_pandas(self) -> pd.DataFrame:
        """Convert the object to a Pandas DataFrame."""
        return pd.DataFrame.from_records(self.to_dicts())

    def to_pretty(
        self,
        keys: Sequence[str] | None = None,
        formatter: Callable[[dict[str, Any], str], str] = get_and_format_num,
    ) -> pd.DataFrame:
        """Convert the object to a Pandas Dataframe with formatted values.

        Args:
            keys: Keys to convert. If a key is not defined in the dictionary
                it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            Pandas Dataframe with formatted values.

        Default formatting rules:
            - If a name starts with `"rel_"` or equals to `"power"` consider it
                a percentage value. Round percentage values to 2 significant digits,
                multiply by `100` and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.
        """
        if keys is None:
            keys = self.default_keys
        return pd.DataFrame.from_records(
            {key: formatter(data, key) for key in keys}
            for data in self.to_dicts()
        )

    def to_string(
        self,
        keys: Sequence[str] | None = None,
        formatter: Callable[[dict[str, Any], str], str] = get_and_format_num,
    ) -> str:
        """Convert the object to a string.

        Args:
            keys: Keys to convert. If a key is not defined in the dictionary
                it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            A table with results rendered as string.

        Default formatting rules:
            - If a name starts with `"rel_"` or equals to `"power"` consider it
                a percentage value. Round percentage values to 2 significant digits,
                multiply by `100` and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.
        """
        return self.to_pretty(keys, formatter).to_string(index=False)

    def to_html(
        self,
        keys: Sequence[str] | None = None,
        formatter: Callable[[dict[str, Any], str], str] = get_and_format_num,
    ) -> str:
        """Convert the object to HTML.

        Args:
            keys: Keys to convert. If a key is not defined in the dictionary
                it's assumed to be `None`.
            formatter: Custom formatter function. It should accept a dictionary
                of metric result attributes and an attribute name, and return
                a formatted attribute value.

        Returns:
            A table with results rendered as HTML.

        Default formatting rules:
            - If a name starts with `"rel_"` or equals to `"power"` consider it
                a percentage value. Round percentage values to 2 significant digits,
                multiply by `100` and add `"%"`.
            - Round other values to 3 significant values.
            - If value is less than `0.001`, format it in exponential presentation.
            - If a name ends with `"_ci"`, consider it a confidence interval.
                Look up for attributes `"{name}_lower"` and `"{name}_upper"`,
                and format the interval as `"[{lower_bound}, {lower_bound}]"`.
        """
        return self.to_pretty(keys, formatter).to_html(index=False)

    def __str__(self) -> str:
        """Object string representation."""
        return self.to_string()

    def _repr_html_(self) -> str:
        """Object HTML representation."""
        return self.to_html()


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
