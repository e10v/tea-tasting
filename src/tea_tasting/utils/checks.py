"""Parameter validation helpers."""
# ruff: noqa: SIM114

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Literal


def check_scalar[R](  # noqa: PLR0913
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


@overload
def auto_check(value: float, name: Literal["alpha"]) -> float:
    ...

@overload
def auto_check(
    value: str,
    name: Literal["alternative"],
) -> Literal["two-sided", "greater", "less"]:
    ...

@overload
def auto_check(value: float, name: Literal["confidence_level"]) -> float:
    ...

@overload
def auto_check(value: bool, name: Literal["correction"]) -> bool:  # noqa: FBT001
    ...

@overload
def auto_check(value: bool, name: Literal["equal_var"]) -> bool:  # noqa: FBT001
    ...

@overload
def auto_check(
    value: int | Sequence[int] | None,
    name: Literal["n_obs"],
) -> int | Sequence[int] | None:
    ...

@overload
def auto_check(value: int, name: Literal["n_resamples"]) -> int:
    ...

@overload
def auto_check(
    value: str,
    name: Literal["nan_policy"],
) -> Literal["propagate", "omit", "raise"]:
    ...

@overload
def auto_check(value: float, name: Literal["power"]) -> float:
    ...

@overload
def auto_check(value: float | int, name: Literal["ratio"]) -> float | int:
    ...

@overload
def auto_check(
    value: int | np.random.Generator | np.random.SeedSequence | None,
    name: Literal["rng"],
) -> int | np.random.Generator | np.random.SeedSequence | None:
    ...

@overload
def auto_check(value: bool, name: Literal["use_t"]) -> bool:  # noqa: FBT001
    ...

@overload
def auto_check[R](value: R, name: str) -> R:
    ...

def auto_check[R](value: R, name: str) -> R:  # noqa: C901, PLR0912
    """Automatically check a parameter's type and value based on its name.

    The following parameter names are supported: `"alpha"`, `"alternative"`,
    `"confidence_level"`, `"correction"`, `"equal_var"`, `"n_obs"`,
    `"n_resamples"`, `"nan_policy"`, `"power"`, `"ratio"`, `"rng"`, `"use_t"`.

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
    elif name == "nan_policy":
        check_scalar(value, name, typ=str, in_={"propagate", "omit", "raise"})
    elif name == "power":
        check_scalar(value, name, typ=float, gt=0, lt=1)
    elif name == "ratio":
        check_scalar(value, name, typ=float | int, gt=0)
    elif name == "rng":
        check_scalar(
            value,
            name,
            typ=int | np.random.Generator | np.random.SeedSequence | None,
        )
    elif name == "use_t":
        check_scalar(value, name, typ=bool)
    return value
