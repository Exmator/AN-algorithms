from dataclasses import dataclass

import sympy as sp

from algorithm_interface import Parameters, Result, Algorithm
from registry import AlgorithmRegistry
from symbolic import (
    ExpressionType,
    X,
    exact_bounds,
    function_range,
    parse_univariate_expression,
    require_continuous,
    to_numeric_function,
)

@dataclass
class FixedPointParams(Parameters):
    f: ExpressionType
    p0: float
    a: float
    b: float
    err: float = 1e-6
    max_iter: int = 100

@dataclass
class FixedPointResult(Result):
    root: float
    iterations: int
    converged: bool

    def __str__(self) -> str:
        if self.converged:
            return f"Root found: {self.root:.6f} (after {self.iterations} iterations)"
        return f"Did not converge after {self.iterations} iterations (last estimate: {self.root:.6f})"

@AlgorithmRegistry.register(
    name="Fixed Point",
    params_cls=FixedPointParams,
    description="Fixed Point method to find the root of f in [a, b]",
)
class FixedPointAlgorithm(Algorithm):
    def _validate(self, p: FixedPointParams) -> None:
        if not isinstance(p, FixedPointParams):
            raise TypeError("parameters must be a FixedPointParams instance")
        if p.a >= p.b:
            raise ValueError("a must be less than b")
        if not (p.a <= p.p0 <= p.b):
            raise ValueError("p0 must lie within [a, b]")
        if p.err <= 0:
            raise ValueError("err must be positive")
        if p.max_iter <= 0:
            raise ValueError("max_iter must be positive")

        f = parse_univariate_expression(p.f)
        g = X - f

        a_exact, b_exact = exact_bounds(p.a, p.b)
        closed_interval = sp.Interval(a_exact, b_exact)
        open_interval = sp.Interval.open(a_exact, b_exact)

        # Existence: g continuous on [a, b] and g([a, b]) subseteq [a, b]
        require_continuous(g, closed_interval, subject="g", interval_label="[a, b]", context="f")
        g_range = function_range(g, closed_interval, subject="g", interval_label="[a, b]", context="f")
        if not g_range.is_subset(closed_interval):
            raise ValueError(
                f"g([a, b]) = {g_range} is not contained in [a, b]: existence of a fixed point is not guaranteed"
            )

        # Uniqueness/convergence: g differentiable on (a, b) with sup|g'| < 1
        dg = sp.diff(g, X)
        require_continuous(
            dg, open_interval, subject="g", interval_label="(a, b)", property_name="differentiable", context="f"
        )
        dg_range = function_range(dg, open_interval, subject="g", interval_label="(a, b)", context="f")
        k = sp.Max(sp.Abs(dg_range.inf), sp.Abs(dg_range.sup))
        if k >= 1:
            raise ValueError(
                f"g is not a contraction on (a, b): sup|g'(x)| = {k} >= 1, "
                "uniqueness of the fixed point is not guaranteed"
            )

    def _execute(self, p: FixedPointParams) -> FixedPointResult:
        f = parse_univariate_expression(p.f)
        g = X - f
        g_func = to_numeric_function(g)

        p_curr = p.p0
        for i in range(p.max_iter):
            p_next = g_func(p_curr)
            if abs(p_next - p_curr) < p.err:
                return FixedPointResult(root=p_next, iterations=i + 1, converged=True)
            p_curr = p_next

        return FixedPointResult(root=p_curr, iterations=p.max_iter, converged=False)
