from typing import Callable

import sympy as sp

X = sp.Symbol("x")
ExpressionType = sp.Expr | str


def parse_univariate_expression(raw: ExpressionType) -> sp.Expr:
    try:
        expr = sp.sympify(raw)
    except (sp.SympifyError, TypeError) as exc:
        raise ValueError("f must be a valid sympy expression") from exc
    if expr.free_symbols - {X}:
        raise ValueError("f must only be expressed in terms of the variable x")
    return expr


def exact_bounds(a: float, b: float) -> tuple[sp.Rational, sp.Rational]:
    return sp.Rational(str(a)), sp.Rational(str(b))


def require_continuous(
    expr: sp.Expr,
    interval: sp.Interval,
    *,
    subject: str,
    interval_label: str,
    property_name: str = "continuous",
    context: str = "",
) -> None:
    suffix = f" for the given {context}" if context else ""
    try:
        domain = sp.calculus.util.continuous_domain(expr, X, interval)
    except NotImplementedError as exc:
        raise ValueError(
            f"cannot determine whether {subject} is {property_name} on {interval_label}{suffix}"
        ) from exc
    if not interval.is_subset(domain):
        raise ValueError(f"{subject} is not {property_name} on {interval_label}")


def function_range(
    expr: sp.Expr,
    interval: sp.Interval,
    *,
    subject: str,
    interval_label: str,
    context: str = "",
) -> sp.Set:
    suffix = f" for the given {context}" if context else ""
    try:
        return sp.calculus.util.function_range(expr, X, interval)
    except NotImplementedError as exc:
        raise ValueError(
            f"cannot determine whether {subject} is continuous on {interval_label}{suffix}"
        ) from exc


def to_numeric_function(expr: sp.Expr) -> Callable[[float], float]:
    return sp.lambdify(X, expr, "math")
